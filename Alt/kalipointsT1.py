# Version 1.5 (Hard Iron Fixes: Seed-Logik korrigiert & Auto-Zentrierung im Plot)
import math
import numpy as np
import pyvista as pv
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import pandas as pd
import io
from datetime import datetime
import re 
import os

# Zufallsgenerator ohne festen Seed - erzeugt echte Zufallszahlen
rng = np.random.default_rng()

# ============================================================
# KONSTANTEN & KONFIGURATION
# ============================================================
OUTPUT_DIR = "kalipoints_datasets"
DEFAULT_SAMPLES = 1000
DEFAULT_ALPHA = 90
DEFAULT_NOISE = 0.0
DEFAULT_OFFSET = [0.0, 0.0, 0.0]
DEFAULT_DISTORTION = [1.0, 1.0, 1.0]
SIDEBAR_WIDTH = "400px"
MESH_OFFSET_SCALE = 1.02
MESH_GRID_RESOLUTION = 100  # Reduziert von 250 für bessere Performance
CACHED_MESHES = {}  # Cache für berechnete Meshes

def create_soft_iron_matrix(x_scale, y_scale, z_scale, x_rot_deg, y_rot_deg, z_rot_deg):
    """
    Erstellt die Soft Iron Transformationsmatrix (Skalierung + Rotation).
    
    Die Matrix wird als: R_z(ψ) * R_y(θ) * R_x(φ) * Skalierung zusammengestellt.
    
    Args:
        x_scale, y_scale, z_scale: Verzerrungsfaktoren (Diagonale der Skalierungsmatrix)
        x_rot_deg, y_rot_deg, z_rot_deg: Rotationen in Grad (um x, y, z Achse)
    
    Returns:
        3x3 Transformationsmatrix als numpy array
    """
    # Konvertiere Grad zu Radianten
    phi = np.radians(x_rot_deg) if x_rot_deg else 0.0
    theta = np.radians(y_rot_deg) if y_rot_deg else 0.0
    psi = np.radians(z_rot_deg) if z_rot_deg else 0.0
    
    # Skalierungsmatrix (Diagonalmatrix mit Verzerrungsfaktoren)
    S = np.array([
        [x_scale, 0, 0],
        [0, y_scale, 0],
        [0, 0, z_scale]
    ])
    
    # Rotationsmatrix um x-Achse (φ)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    R_x = np.array([
        [1, 0, 0],
        [0, cos_phi, -sin_phi],
        [0, sin_phi, cos_phi]
    ])
    
    # Rotationsmatrix um y-Achse (θ)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R_y = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])
    
    # Rotationsmatrix um z-Achse (ψ)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    R_z = np.array([
        [cos_psi, -sin_psi, 0],
        [sin_psi, cos_psi, 0],
        [0, 0, 1]
    ])
    
    # Kombinierte Transformationsmatrix: R_z * R_y * R_x * S
    # (Zuerst Skalierung, dann Rotation um x, dann y, dann z)
    transform_matrix = R_z @ R_y @ R_x @ S
    
    return transform_matrix

def apply_soft_iron_transformation(points, x_scale, y_scale, z_scale, 
                                   x_rot_deg, y_rot_deg, z_rot_deg):
    """
    Wendet die Soft Iron Transformation auf eine Punktewolke an.
    
    Args:
        points: Nx3 numpy array mit Punkten
        x_scale, y_scale, z_scale: Verzerrungsfaktoren
        x_rot_deg, y_rot_deg, z_rot_deg: Rotationen in Grad
    
    Returns:
        Transformierte Nx3 Punktewolke
    """
    if x_scale == 1.0 and y_scale == 1.0 and z_scale == 1.0 and \
       not x_rot_deg and not y_rot_deg and not z_rot_deg:
        # Keine Transformation nötig
        return points
    
    transform_matrix = create_soft_iron_matrix(x_scale, y_scale, z_scale,
                                              x_rot_deg, y_rot_deg, z_rot_deg)
    
    # Wende Transformation auf alle Punkte an (Nx3 @ 3x3)
    transformed_points = points @ transform_matrix.T
    
    return transformed_points

def create_iron_error_matrix(x_scale, y_scale, z_scale, x_rot_deg, y_rot_deg, z_rot_deg,
                             hi_x, hi_y, hi_z):
    """
    Erstellt die komplette 4x4 Iron Error Matrix aus Soft Iron (SI) und Hard Iron (HI) Parametern.
    
    Die Matrix kombiniert:
    - Die 3x3 Soft Iron Transformationsmatrix (mit Skalierung + Rotation)
    - Die Hard Iron Offset Werte (x, y, z)
    
    Returns: 4x4 numpy array im Format:
    [[ SI_tm11, SI_tm12, SI_tm13, HI-X],
     [ SI_tm21, SI_tm22, SI_tm23, HI-Y],
     [ SI_tm31, SI_tm32, SI_tm33, HI-Z],
     [       0,       0,       0,   1 ]]
    """
    # Berechne die 3x3 Soft Iron Transformationsmatrix
    transform_matrix_3x3 = create_soft_iron_matrix(x_scale, y_scale, z_scale,
                                                   x_rot_deg, y_rot_deg, z_rot_deg)
    
    # Erstelle die 4x4 Matrix mit Hard Iron Offset
    iron_error_matrix = np.eye(4)
    
    # Kopiere die 3x3 Soft Iron Transformationsmatrix in die obere linke Ecke
    iron_error_matrix[:3, :3] = transform_matrix_3x3
    
    # Setze die Hard Iron Offset in der letzten Spalte (Translationskomponente)
    iron_error_matrix[0, 3] = hi_x
    iron_error_matrix[1, 3] = hi_y
    iron_error_matrix[2, 3] = hi_z
    
    # Die letzte Reihe bleibt [0, 0, 0, 1]
    
    return iron_error_matrix

def generate_random_soft_iron_params(max_distortion, max_rotation):
    """
    Generiert zufällige Soft Iron Parameter (Verzerrung und Rotation) ohne Seed.
    
    Args:
        max_distortion: Maximale Verzerrung pro Achse (Standard: 2.0)
        max_rotation: Maximale Rotation pro Achse in Grad (Standard: 10.0)
    
    Returns:
        Tuple: (x_scale, y_scale, z_scale, x_rot, y_rot, z_rot)
    """
    # Defaultwerte, falls None
    max_distortion = max_distortion if max_distortion is not None else 2.0
    max_rotation = max_rotation if max_rotation is not None else 10.0
    
    # Zufällige Verzerrungsfaktoren zwischen 1 und max_distortion
    x_scale = rng.uniform(1.0, float(max_distortion))
    y_scale = rng.uniform(1.0, float(max_distortion))
    z_scale = rng.uniform(1.0, float(max_distortion))
    
    # Zufällige Rotationen zwischen 0 und max_rotation (für jede Achse)
    x_rot = rng.uniform(0.0, float(max_rotation))
    y_rot = rng.uniform(0.0, float(max_rotation))
    z_rot = rng.uniform(0.0, float(max_rotation))
    
    return x_scale, y_scale, z_scale, x_rot, y_rot, z_rot

# ============================================================
# 1. HILFSFUNKTIONEN FÜR SEEDS & DATENVERWALTUNG
# ============================================================
def fibonacci_sphere(generation_mode, noise_value, samples=1000, alpha=90, seed_string="", 
                     xyz0=[0, 0, 0], xyz1=[0, 0, 0], x_rot=0, y_rot=0, z_rot=0):
    """ 
    Generiert Punkte auf einem Kugelstreifen mit optionaler Soft Iron Transformation.
    
    Args:
        generation_mode: 'optimal', 'random', oder 'path'
        noise_value: Rausch-Standardabweichung
        samples: Anzahl Punkte
        alpha: Winkeleinschränkung in Grad
        seed_string: (nicht mehr verwendet, vorhanden für Kompatibilität)
        xyz0: Hard Iron Offset [x, y, z]
        xyz1: Soft Iron Skalierung [x_scale, y_scale, z_scale]
        x_rot, y_rot, z_rot: Rotationen in Grad
    
    Rückgabe: (points_array, used_offset_list)
    """
    # Verwende globalen RNG ohne Seed

    # Hard Iron Offset wird direkt aus den UI-Eingaben verwendet
    current_offset = list(xyz0) # Kopie der manuellen Offsets
    
    # --------------------------------------------------

    alpha_rad = math.pi * alpha / 180.0
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)
    z_limit = math.sin(alpha_rad)
    z_span = z_limit * 2
    sample_divisor = float(samples - 1) if samples > 1 else 1.0

    for i in range(samples):
        fraction = i / sample_divisor
        z = z_limit - (fraction * z_span)
        radius = math.sqrt(1 - z * z)
        theta = phi * i
        if generation_mode == "random": 
            theta = rng.uniform(0, math.pi*2)
        x = math.cos(theta) * radius
        y = math.sin(theta) * radius

        if noise_value != 0.0:
            x += rng.uniform(-noise_value, noise_value)
            y += rng.uniform(-noise_value, noise_value)
            z += rng.uniform(-noise_value, noise_value)

        points.append((x, y, z))

    points_array = np.array(points)
    
    # --- SOFT IRON TRANSFORMATION ---
    # Extrahiere Soft Iron Skalierungsfaktoren
    x_scale = float(xyz1[0]) if len(xyz1) > 0 else 1.0
    y_scale = float(xyz1[1]) if len(xyz1) > 1 else 1.0
    z_scale = float(xyz1[2]) if len(xyz1) > 2 else 1.0
    
    # Wende Soft Iron Transformation an (Rotation + Skalierung)
    if x_scale != 1.0 or y_scale != 1.0 or z_scale != 1.0 or x_rot or y_rot or z_rot:
        points_array = apply_soft_iron_transformation(
            points_array, x_scale, y_scale, z_scale, x_rot, y_rot, z_rot
        )
    
    # --- HARD IRON OFFSET WIRD ZULETZT ADDIERT (nach Transformation) ---
    points_array[:, 0] += current_offset[0]
    points_array[:, 1] += current_offset[1]
    points_array[:, 2] += current_offset[2]

    return points_array, current_offset

def create_sphere_mesh(xyz0 = [0, 0, 0]):
    """ 
    Erzeugt das Mesh für die Kugel mit Caching.
    Bei großen Offsets wird gecacht, um Performance zu sparen.
    """
    # Cache-Key basierend auf Offset (mit Genauigkeit auf 2 Dezimalstellen)
    cache_key = tuple(round(x, 2) for x in xyz0)
    
    if cache_key in CACHED_MESHES:
        return CACHED_MESHES[cache_key]
    
    x0, y0, z0 = xyz0[0], xyz0[1], xyz0[2]
    r = 1
    R = r * MESH_OFFSET_SCALE
    
    def f(x, y, z):
        return (x-x0)**2 + (y-y0)**2 + (z-z0)**2
    
    # Reduzierte Grid-Auflösung für bessere Performance (100 statt 250)
    X, Y, Z = np.mgrid[(-R+x0):(R+x0):MESH_GRID_RESOLUTION*1j, 
                       (-R+y0):(R+y0):MESH_GRID_RESOLUTION*1j, 
                       (-R+z0):(R+z0):MESH_GRID_RESOLUTION*1j]
    
    grid = pv.StructuredGrid(X, Y, Z)
    values = f(X, Y, Z)
    grid.point_data["values"] = values.ravel(order="F")
    isosurf = grid.contour(isosurfaces=[r**2])
    mesh = isosurf.extract_geometry()
    vertices = mesh.points
    triangles = mesh.faces.reshape(-1, 4)
    
    result = (vertices, triangles, np.array(xyz0))
    
    # Cache-Größe begrenzen zur Speicheroptimierung
    if len(CACHED_MESHES) > 20:
        oldest_key = next(iter(CACHED_MESHES))
        del CACHED_MESHES[oldest_key]
    
    CACHED_MESHES[cache_key] = result
    return result

def build_figure_with_points(data_points, center_offset, sphere_vertices, sphere_triangles, 
                             alpha, noise_value, x_distortion, y_distortion, z_distortion, 
                             mesh_opacity, show_mesh=True, point_color='blue'):
    """
    Erstellt eine Plotly Figure mit optionalem Kugel-Mesh und Datenpunkten.
    Die Achsenlängen sind ALLE gleich (keine Skalenverzerrung).
    Die Skala wird so angepasst, dass alle Punkte (Mesh + Datenpunkte) sichtbar sind.
    
    Args:
        show_mesh: Boolean - ob das Mesh angezeigt werden soll (default: True)
        point_color: Farbe der Datenpunkte als String (default: 'blue')
    """
    # Bestimme die Bounding Box
    if show_mesh:
        # Mit Mesh: Berechne BBox mit Mesh + Datenpunkten
        all_points = np.vstack([sphere_vertices, data_points])
    else:
        # Ohne Mesh: Berechne BBox nur mit Datenpunkten
        all_points = data_points
    
    # Finde Min/Max für alle Koordinaten
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    min_z, max_z = np.min(all_points[:, 2]), np.max(all_points[:, 2])
    
    # Berechne die Range in jeder Dimension
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    
    # Der max_range ist der größte Range über alle Dimensionen
    max_range_actual = max(range_x, range_y, range_z)
    
    # Addiere 5% Puffer für bessere Sichtbarkeit
    max_range = max_range_actual * 0.525  # 0.525 = 1.05 / 2 (da wir ±range vom Center gehen)
    # Berechne das Zentrum aller Punkte
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    # Erstelle die Traces (mit oder ohne Mesh)
    data_traces = []
    
    # Füge Mesh nur hinzu, wenn show_mesh=True
    if show_mesh:
        intensity = np.linspace(0, 1, len(sphere_vertices))
        data_traces.append(
            go.Mesh3d(
                x=sphere_vertices[:, 0], 
                y=sphere_vertices[:, 1], 
                z=sphere_vertices[:, 2], 
                colorscale=[[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'magenta']], 
                intensity=intensity, 
                i=sphere_triangles[:, 1], 
                j=sphere_triangles[:, 2], 
                k=sphere_triangles[:, 3], 
                opacity=mesh_opacity, 
                showscale=False, 
                name='Einheitskugel'
            )
        )
    
    # Datenpunkte mit flexibler Farbe
    data_traces.append(
        go.Scatter3d(
            x=data_points[:, 0], 
            y=data_points[:, 1], 
            z=data_points[:, 2], 
            mode='markers', 
            marker=dict(size=2, color=point_color, opacity=1.0), 
            name=f'Gesamtpunkte ({len(data_points)})', 
            hoverinfo='none'
        )
    )
    
    fig = go.Figure(data=data_traces)
    fig.update_layout(
        scene=dict(
            aspectmode='cube',  # Wichtig: gleiche Skala auf allen Achsen
            xaxis_title='X', 
            yaxis_title='Y', 
            zaxis_title='Z',
            # ALLE ACHSEN HABEN DEN GLEICHEN BEREICH (um den gemeinsamen Center)
            xaxis=dict(range=[center_x - max_range, center_x + max_range]), 
            yaxis=dict(range=[center_y - max_range, center_y + max_range]), 
            zaxis=dict(range=[center_z - max_range, center_z + max_range]),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1), 
            center=dict(x=0, y=0, z=0), 
            eye=dict(x=1.25, y=1.25, z=1.25)
        ),
        hovermode=False  # Disable hover für bessere Performance
    )
    
    return fig

def build_results_display(data_points, alpha, generation_mode, noise_value, center_offset, 
                         x_distortion, y_distortion, z_distortion, x_rot=0, y_rot=0, z_rot=0,
                         iron_error_matrix=None, sample_count=None, raw_metadata=None):
    """
    Erstellt die Ergebnis-Anzeige für die rechte Sidebar mit Metadaten und Transformationsmatrix.
    
    Args:
        raw_metadata: Dict mit Original-CSV-Metadaten (können None-Werte enthalten für fehlende Felder)
    """
    p_style = {'margin': '4px 0', 'fontSize': '0.9em', 'lineHeight': '1.2'}
    title_style = {'margin': '8px 0 4px 0', 'fontSize': '1em', 'fontWeight': 'bold'}
    hr_style = {'margin': '8px 0'}
    matrix_style = {'margin': '2px 0', 'fontSize': '12px', 'fontFamily': 'monospace', 'lineHeight': '1.4', 'letterSpacing': '0.02em'}
    
    # Hilfsfunktion für N/A-Handling
    def format_value(val, fmt='{:.4f}'):
        if val is None:
            return 'N/A'
        try:
            return fmt.format(val) if isinstance(fmt, str) else fmt(val)
        except:
            return str(val)
    
    result = [
        html.H4("Metadaten", style=title_style),
    ]
    
    # Nutze raw_metadata wenn vorhanden (z.B. von CSV-Load), sonst Parameter
    # raw_metadata hat Vorrang, weil es die echten Werte mit None-Indikatoren hat
    
    # Verteilung
    if raw_metadata:
        dist_style = raw_metadata.get('DISTRIBUTION_STYLE')
        if dist_style == 'evenly':
            dist_text = 'Gleichmässig'
        elif dist_style == 'randomly':
            dist_text = 'Zufall'
        elif dist_style is None:
            dist_text = 'N/A'
        else:
            dist_text = dist_style
    else:
        dist_text = 'Gleichmässig' if generation_mode == 'optimal' else 'Zufall' if generation_mode == 'random' else 'Pfad'
    result.append(html.P(f"Verteilung: {dist_text}", style=p_style))
    
    # Rauschen
    if raw_metadata and 'NOISE' in raw_metadata:
        noise_val = raw_metadata.get('NOISE')
    else:
        noise_val = noise_value
    result.append(html.P(f"Rauschen: {format_value(noise_val, '{:.4f}')}", style=p_style))
    
    # Anzahl Punkte
    if raw_metadata and 'POINT_AMOUNT' in raw_metadata:
        point_val = raw_metadata.get('POINT_AMOUNT')
        if point_val is None:
            point_display = 'N/A'
        else:
            point_display = str(int(point_val))
    else:
        point_count = sample_count if sample_count is not None else len(data_points)
        point_display = str(point_count)
    result.append(html.P(f"Anzahl Punkte: {point_display}", style=p_style))
    
    # Winkeleinschränkung
    if raw_metadata and 'ANGULAR_CONSTRAINT_DEG' in raw_metadata:
        alpha_val = raw_metadata.get('ANGULAR_CONSTRAINT_DEG')
    else:
        alpha_val = alpha
    result.append(html.P(f"Winkeleinschränkung: {format_value(alpha_val, '{:.1f}')}°", style=p_style))
    
    # Feldlinienwinkel (hardcoded, nicht implementiert)
    result.append(html.P(f"Feldlinienwinkel: 0° (nicht implementiert)", style=p_style))
    
    # Hard Iron Offset
    if raw_metadata and ('HI_OFFSET' in raw_metadata or 'HI_X_Y_Z_OFFSET' in raw_metadata):
        hi = raw_metadata.get('HI_OFFSET') or raw_metadata.get('HI_X_Y_Z_OFFSET')
        if hi is None:
            hi_text = "N/A"
        else:
            hi_text = f"({format_value(hi[0], '{:.4f}')}, {format_value(hi[1], '{:.4f}')}, {format_value(hi[2], '{:.4f}')})"
    else:
        hi_text = f"({format_value(center_offset[0], '{:.4f}')}, {format_value(center_offset[1], '{:.4f}')}, {format_value(center_offset[2], '{:.4f}')})"
    result.append(html.P(f"Hard Iron (X,Y,Z): {hi_text}", style=p_style))
    
    # SI Verzerrung
    if raw_metadata and ('SI_DISTORTION' in raw_metadata or 'SI_X_Y_Z_DISTORTION' in raw_metadata):
        si_dist = raw_metadata.get('SI_DISTORTION') or raw_metadata.get('SI_X_Y_Z_DISTORTION')
        if si_dist is None:
            si_dist_text = "N/A"
        else:
            si_dist_text = f"({format_value(si_dist[0], '{:.4f}')}, {format_value(si_dist[1], '{:.4f}')}, {format_value(si_dist[2], '{:.4f}')})"
    else:
        si_dist_text = f"({format_value(x_distortion, '{:.4f}')}, {format_value(y_distortion, '{:.4f}')}, {format_value(z_distortion, '{:.4f}')})"
    result.append(html.P(f"SI Verzerrung (X,Y,Z): {si_dist_text}", style=p_style))
    
    # SI Rotation
    if raw_metadata and ('SI_ROTATION_DEG' in raw_metadata or 'SI_X_Y_Z_ROTATION_DEG' in raw_metadata):
        si_rot = raw_metadata.get('SI_ROTATION_DEG') or raw_metadata.get('SI_X_Y_Z_ROTATION_DEG')
        if si_rot is None:
            si_rot_text = "N/A"
        else:
            si_rot_text = f"({format_value(si_rot[0], '{:.2f}')}°, {format_value(si_rot[1], '{:.2f}')}°, {format_value(si_rot[2], '{:.2f}')}°)"
    else:
        si_rot_text = f"({format_value(x_rot, '{:.2f}')}°, {format_value(y_rot, '{:.2f}')}°, {format_value(z_rot, '{:.2f}')}°)"
    result.append(html.P(f"SI Rotation (X,Y,Z): {si_rot_text}", style=p_style))
    
    # Transformationsmatrix
    if iron_error_matrix is not None:
        result.append(html.P("Transformationsmatrix (4x4):", style={'margin': '8px 0 2px 0', 'fontSize': '0.9em', 'fontWeight': 'bold'}))
        for row in iron_error_matrix:
            # Formatiere mit 13 Zeichen Breite für bessere Spaltenausrichtung
            row_text = "[" + ", ".join([f"{val:>12.6f}" for val in row]) + " ]"
            result.append(html.P(row_text, style=matrix_style))
    else:
        result.append(html.P("Transformationsmatrix: N/A (keine Daten)", style=p_style))
    
    result.append(html.Hr(style=hr_style))
    return result

def load_csv_data_by_seed(seed_string, suppress_error=False):
    """
    Liest Punktedaten und erweiterte Metadaten aus der CSV.
    Rückgabe: (numpy_array, metadaten_dict, error_message)
    
    metadaten_dict enthält:
    - DISTRIBUTION_STYLE: 'evenly' oder 'randomly'
    - NOISE: Rausch-Standardabweichung (float)
    - POINT_AMOUNT: Anzahl der Punkte (int)
    - ANGULAR_CONSTRAINT_DEG: Winkeleinschränkung in Grad (float)
    - FIELD_LINE_ANGLE_DEG: Feldlinienwinkel in Grad (float) - derzeit nicht implementiert
    - HI_OFFSET: [x, y, z] Hard Iron Offset (floats)
    - SI_DISTORTION: [x, y, z] Soft Iron Verzerrung (floats)
    - SI_ROTATION_DEG: [x, y, z] Rotationen in Grad (floats)
    - IRON_ERROR_MATRIX_RAD: 4x4 Matrix (für neue Formate)
    
    Args:
        seed_string: Der Seed zum Laden
        suppress_error: Wenn True, wird Fehler "Datei nicht gefunden" nicht zurückgegeben
    """
    full_path = os.path.join(OUTPUT_DIR, f"{seed_string}.csv")
    
    if not os.path.exists(full_path):
        if suppress_error:
            return None, None, None  # Leise fehlschlagen wenn suppress=True
        return None, None, f"Fehler: Datei '{seed_string}.csv' nicht gefunden."
        
    try:
        # 1. Metadaten aus den ersten Zeilen lesen (Rückwärtskompatibilität + Neue Formate)
        # WICHTIG: Mit None initialisieren, nicht mit Defaults!
        # So können fehlende Felder erkannt und als "N/A" angezeigt werden
        metadata = {
            'DISTRIBUTION_STYLE': None,
            'NOISE': None,
            'POINT_AMOUNT': None,
            'ANGULAR_CONSTRAINT_DEG': None,
            'FIELD_LINE_ANGLE_DEG': None,
            'HI_OFFSET': None,
            'SI_DISTORTION': None,
            'SI_ROTATION_DEG': None,
            'IRON_ERROR_MATRIX_RAD': None,
            # Alte Kompatibilität
            'HI_X_Y_Z_OFFSET': None,
            'SI_X_Y_Z_DISTORTION': None,
            'SI_X_Y_Z_ROTATION_DEG': None
        }
        
        with open(full_path, 'r') as f:
            for i in range(20):  # Lese bis zu 20 Zeilen für Metadaten
                line = f.readline().strip()
                if not line.startswith("#"):
                    break
                
                # Neue Metadaten-Format
                if line.startswith("# DISTRIBUTION_STYLE:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        metadata['DISTRIBUTION_STYLE'] = value
                elif line.startswith("# NOISE:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        try:
                            metadata['NOISE'] = float(value)
                        except (ValueError, TypeError):
                            metadata['NOISE'] = None
                elif line.startswith("# POINT_AMOUNT:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        try:
                            metadata['POINT_AMOUNT'] = int(value)
                        except (ValueError, TypeError):
                            metadata['POINT_AMOUNT'] = None
                elif line.startswith("# ANGULAR_CONSTRAINT-DEG:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        try:
                            metadata['ANGULAR_CONSTRAINT_DEG'] = float(value)
                        except (ValueError, TypeError):
                            metadata['ANGULAR_CONSTRAINT_DEG'] = None
                elif line.startswith("# FIELD_LINE_ANGLE-DEG:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        try:
                            # Extrahiere nur die erste Zahl (falls Kommentar folgt)
                            metadata['FIELD_LINE_ANGLE_DEG'] = float(value.split()[0])
                        except (ValueError, IndexError, TypeError):
                            metadata['FIELD_LINE_ANGLE_DEG'] = None
                elif line.startswith("# HI-X-Y-Z-OFFSET:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        try:
                            metadata['HI_X_Y_Z_OFFSET'] = [float(x.strip()) for x in value.split(",")]
                            metadata['HI_OFFSET'] = metadata['HI_X_Y_Z_OFFSET'].copy()
                        except (ValueError, TypeError):
                            metadata['HI_OFFSET'] = None
                            metadata['HI_X_Y_Z_OFFSET'] = None
                elif line.startswith("# SI-X-Y-Z-DISTORTION:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        try:
                            metadata['SI_X_Y_Z_DISTORTION'] = [float(x.strip()) for x in value.split(",")]
                            metadata['SI_DISTORTION'] = metadata['SI_X_Y_Z_DISTORTION'].copy()
                        except (ValueError, TypeError):
                            metadata['SI_DISTORTION'] = None
                            metadata['SI_X_Y_Z_DISTORTION'] = None
                elif line.startswith("# SI-X-Y-Z-ROTATION-DEG:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        try:
                            metadata['SI_X_Y_Z_ROTATION_DEG'] = [float(x.strip()) for x in value.split(",")]
                            metadata['SI_ROTATION_DEG'] = metadata['SI_X_Y_Z_ROTATION_DEG'].copy()
                        except (ValueError, TypeError):
                            metadata['SI_ROTATION_DEG'] = None
                            metadata['SI_X_Y_Z_ROTATION_DEG'] = None
                elif line.startswith("# IRON_ERROR_MATRIX_FORMAT-RAD:"):
                    # Parst die 4x4 Matrix (mehrzeilig)
                    # Die Matrix wird gleich in den nächsten Zeilen gelesen
                    pass
                
                # Old format (für Rückwärtskompatibilität)
                if line.startswith("# HI_OFFSET:"):
                    try:
                        content = line.split(":")[1]
                        hi_offset = [float(x) for x in content.split(";")]
                        metadata['HI_OFFSET'] = hi_offset
                        metadata['HI_X_Y_Z_OFFSET'] = hi_offset
                    except (ValueError, TypeError):
                        metadata['HI_OFFSET'] = None
                elif line.startswith("# SI_DISTORTION:"):
                    try:
                        content = line.split(":")[1]
                        si_dist = [float(x) for x in content.split(";")]
                        metadata['SI_DISTORTION'] = si_dist
                        metadata['SI_X_Y_Z_DISTORTION'] = si_dist
                    except (ValueError, TypeError):
                        metadata['SI_DISTORTION'] = None
                elif line.startswith("# SI_ROTATION:"):
                    try:
                        content = line.split(":")[1]
                        si_rot = [float(x) for x in content.split(";")]
                        metadata['SI_ROTATION_DEG'] = si_rot
                        metadata['SI_X_Y_Z_ROTATION_DEG'] = si_rot
                    except (ValueError, TypeError):
                        metadata['SI_ROTATION_DEG'] = None

        # 2. Pandas liest den Rest (ignoriert Zeilen mit #)
        df = pd.read_csv(full_path, sep=';', comment='#')
        
        return df[['X', 'Y', 'Z']].values, metadata, None 
    except Exception as e:
        return None, None, f"Fehler beim Lesen der CSV-Datei: {e}"

def get_available_seeds():
    """Liest alle CSV-Dateien im Output-Ordner und gibt sie als Dropdown-Optionen zurück."""
    options = []
    
    if os.path.exists(OUTPUT_DIR):
        files = os.listdir(OUTPUT_DIR)
        files.sort(reverse=True)
        
        for f in files:
            if f.endswith(".csv"):
                seed_name = f.replace(".csv", "")
                options.append({'label': seed_name, 'value': seed_name})
    
    if not options:
        return [{'label': 'Keine Datensätze gefunden', 'value': 'none', 'disabled': True}]
    
    return options

# ============================================================
# 2. FUNKTIONEN FÜR PUNKT-GENERIERUNG & VISUALISIERUNG
# ============================================================

# ============================================================
# 3. STYLES & LAYOUT KONFIGURATION
# ============================================================

sphere_vertices, sphere_triangles, _ = create_sphere_mesh()

app = Dash(__name__)

SIDEBAR_WIDTH = "400px" 
SECTION_STYLE = {'marginBottom': '20px', 'paddingBottom': '10px', 'borderBottom': '1px dotted #ccc'}
BUTTON_STYLE_INLINE = {'width': 'auto', 'margin': '0 5px', 'flexShrink': 0}
TAB_STYLE = {'padding': '6px 12px', 'fontWeight': 500, 'fontSize': '0.9em', 'border': '1px solid #ddd', 'backgroundColor': '#f0f0f0', 'borderRadius': '4px 4px 0 0', 'marginRight': '2px'}
TAB_SELECTED_STYLE = {'padding': '6px 12px', 'fontWeight': 'bold', 'borderTop': '2px solid #007bff', 'borderLeft': '1px solid #ddd', 'borderRight': '1px solid #ddd', 'borderBottom': '1px solid white', 'backgroundColor': 'white', 'fontSize': '0.9em', 'borderRadius': '4px 4px 0 0'}

SIDEBAR_STYLE = {
    "position": "relative", "width": SIDEBAR_WIDTH, "background-color": "#f8f9fa",
    "padding": "20px", "transition": "all 0.5s", "overflow-y": "auto",     
    "border-right": "1px solid #ddd", "display": "flex", "flex-direction": "column", "box-sizing": "border-box"
}

COLLAPSED_STYLE = {"width": "0px", "padding": "0px", "overflow": "hidden", "transition": "all 0.5s", "border": "none"}

BUTTON_STYLE = {
    "position": "absolute", "top": "50%", "zIndex": "100", "width": "20px", "height": "60px",
    "border": "1px solid #ccc", "backgroundColor": "white", "cursor": "pointer",
    "display": "flex", "alignItems": "center", "justifyContent": "center",
    "fontWeight": "bold", "color": "#555", "borderRadius": "0 5px 5px 0" 
}

app.layout = html.Div([
    dcc.Store(id='left-sidebar-state', data=True),
    dcc.Store(id='right-sidebar-state', data=False),
    dcc.Store(id='point-data-storage', data=None),
    dcc.Store(id='seed-data-storage', data=None),
    dcc.Store(id='loaded-seed-string-store', data=""),  # Speichert den ORIGINAL-Seed der geladen wurde
    dcc.Store(id='current-seed-store', data=""), 
    dcc.Store(id='export-filename-store', data=""),  # Speichert den vereinfachten Dateinamen
    dcc.Store(id='export-success-trigger', data=0),
    dcc.Store(id='magnetic-error-values-store', data=None),
    html.Div(id='download-output', style={'display': 'none'}),
    dcc.Download(id="download-csv"),

    html.Div([
        # --- LINKES FENSTER ---
        html.Div(id='left-sidebar', style=SIDEBAR_STYLE, children=[
            html.H2("Einstellungen", style={'marginBottom': '20px', 'whiteSpace': 'nowrap'}),
            
            # 1. Punktegenerierung & Modus
            html.Div(style=SECTION_STYLE, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
                    html.Label("Verteilung:", style={'fontWeight': 'bold', 'flexShrink': 0}),
                    dcc.RadioItems(
                        id='generation-mode',
                        options=[{'label': ' Gleichmässig ', 'value': 'optimal'}, {'label': 'Zufall', 'value': 'random'}, {'label': ' Pfad', 'value': 'path', 'disabled': True}],
                        value='optimal', inline=True, style={'display': 'flex', 'gap': '10px', 'flexGrow': 1} 
                    ),
                ]),
                html.P("Funktion Pfad ist noch nicht implementiert worden", style={'fontSize': '0.85em', 'color': '#999', 'marginTop': '8px', 'marginBottom': '0'}),
            ]),
            
            # 2. Fehlerabweichung/Rauschen
            html.Div(style=SECTION_STYLE, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[ 
                    html.Label("Rauschen:", style={'fontWeight': 'bold', 'marginRight': '5px', 'whiteSpace': 'nowrap'}),
                    html.Div(style={'flexGrow': 1}, children=[
                        dcc.Input(id='noise-input', type='number', step=0.001, placeholder='Fehlerabweichung (Standard: 0 bzw. 0.001)', style={'width': '100%'})
                    ]),
                ]),
            ]),
            
            # 3. Anzahl Punkte
            html.Div(style=SECTION_STYLE, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                    html.Label("Anzahl Punkte:", style={'fontWeight': 'bold', 'flexShrink': 0}),
                    dcc.Input(id='sample-duration-dropdown', type='number', min=100, max=10000, step=100, placeholder='1000', style={'flexGrow': 1, 'minWidth': '0'}),
                ]),
            ]),
            
            # 4. Winkeleinschränkung 
            html.Div(style=SECTION_STYLE, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                    html.Label("Winkeleinschränkung in Grad:", style={'fontWeight': 'bold', 'flexShrink': 0}),
                    dcc.Input(id='angular-constraint-input', type='number', min=0, max=90, step=1, placeholder='90', style={'flexGrow': 1, 'minWidth': '0'}),
                ])
            ]),
            
            # 4b. Feldlinienwinkel (noch nicht implementiert)
            html.Div(style=SECTION_STYLE, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                    html.Label("Feldlinienwinkel:", style={'fontWeight': 'bold', 'flexShrink': 0}),
                    dcc.Input(id='fieldline-angle-input', type='number', placeholder='Aktuell 0° fixiert', disabled=True, style={'flexGrow': 1, 'minWidth': '0', 'backgroundColor': '#e9ecef', 'color': '#6c757d', 'cursor': 'not-allowed'}),
                ]),
                html.P("Funktion noch nicht implementiert worden", style={'fontSize': '0.85em', 'color': '#999', 'marginTop': '8px', 'marginBottom': '0'}),
            ]),

            # 5. Hard Iron (Versatz) mit Mode-Tabs
            html.Div(style=SECTION_STYLE, children=[
                html.Label("Hard Iron (Versatz):", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                dcc.Tabs(id="hard-iron-mode-tabs", value='hard-iron-random', style={'height': '30px'}, children=[
                    dcc.Tab(label='Manuell', value='hard-iron-manual', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                        html.Div(style={'paddingTop': '10px'}, children=[
                            html.Label("Achsenabweichung:"),
                            html.Div(style={'display': 'flex', 'gap': '5px'}, children=[
                                html.Div(style={'flex': '1'}, children=[html.Div("x", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='x-offset-input', type='number', placeholder='0', step=0.1, style={'width': '100%'})]),
                                html.Div(style={'flex': '1'}, children=[html.Div("y", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='y-offset-input', type='number', placeholder='0', step=0.1, style={'width': '100%'})]),
                                html.Div(style={'flex': '1'}, children=[html.Div("z", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='z-offset-input', type='number', placeholder='0', step=0.1, style={'width': '100%'})]),
                            ]),
                        ]),
                    ]),
                    dcc.Tab(label='Zufällig', value='hard-iron-random', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                        html.Div(style={'paddingTop': '10px'}, children=[
                            dcc.Tabs(id="hard-iron-random-type-tabs", value='hi-random-collective', style={'height': '30px'}, children=[
                                dcc.Tab(label='Kollektiv', value='hi-random-collective', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                                    html.Div(style={'paddingTop': '10px'}, children=[
                                        html.Label("Zufallsbereich:"),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '5px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='offset-rand-min-collective', type='number', placeholder='Minimalwert', step=0.1, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='offset-rand-max-collective', type='number', placeholder='Maximalwert', step=0.1, style={'width': '100%'})]),
                                        ]),
                                    ]),
                                ]),
                                dcc.Tab(label='Achsenspezifisch', value='hi-random-specific', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                                    html.Div(style={'paddingTop': '10px'}, children=[
                                        html.Label("X-Achse:"),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '3px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='offset-rand-min-x', type='number', placeholder='Min', step=0.1, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='offset-rand-max-x', type='number', placeholder='Max', step=0.1, style={'width': '100%'})]),
                                        ]),
                                        html.Label("Y-Achse:", style={'marginTop': '10px'}),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '3px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='offset-rand-min-y', type='number', placeholder='Min', step=0.1, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='offset-rand-max-y', type='number', placeholder='Max', step=0.1, style={'width': '100%'})]),
                                        ]),
                                        html.Label("Z-Achse:", style={'marginTop': '10px'}),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '3px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='offset-rand-min-z', type='number', placeholder='Min', step=0.1, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='offset-rand-max-z', type='number', placeholder='Max', step=0.1, style={'width': '100%'})]),
                                        ]),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
            
            # 6. Soft Iron (Verzerrung) mit Mode-Tabs
            html.Div(style=SECTION_STYLE, children=[
                html.Label("Soft Iron (Verzerrung/Rotation):", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                dcc.Tabs(id="soft-iron-mode-tabs", value='soft-iron-random', style={'height': '30px'}, children=[
                    dcc.Tab(label='Manuell', value='soft-iron-manual', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                        html.Div(style={'paddingTop': '10px'}, children=[
                            html.Label("Verzerrungsfaktor:"),
                            html.Div(style={'display': 'flex', 'gap': '5px'}, children=[
                                html.Div(style={'flex': '1'}, children=[html.Div("x", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='x-distortion-input', type='number', placeholder='1', step=0.01, style={'width': '100%'})]),
                                html.Div(style={'flex': '1'}, children=[html.Div("y", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='y-distortion-input', type='number', placeholder='1', step=0.01, style={'width': '100%'})]),
                                html.Div(style={'flex': '1'}, children=[html.Div("z", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='z-distortion-input', type='number', placeholder='1', step=0.01, style={'width': '100%'})]),
                            ]),
                            html.Label("Rotation: (x, y, z Reihenfolge)", style={'marginTop': '10px'}),
                            html.Div(style={'display': 'flex', 'gap': '5px'}, children=[
                                html.Div(style={'flex': '1'}, children=[html.Div("x", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='x-rotation', placeholder='0 - 360', type='number', step=0.01, style={'width': '100%'})]),
                                html.Div(style={'flex': '1'}, children=[html.Div("y", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='y-rotation', placeholder='0 - 360', type='number', step=0.01, style={'width': '100%'})]),
                                html.Div(style={'flex': '1'}, children=[html.Div("z", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='z-rotation', placeholder='0 - 360', type='number', step=0.01, style={'width': '100%'})]),
                            ]),
                        ]),
                    ]),
                    dcc.Tab(label='Zufällig', value='soft-iron-random', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                        html.Div(style={'paddingTop': '10px'}, children=[
                            dcc.Tabs(id="soft-iron-random-type-tabs", value='si-random-collective', style={'height': '30px'}, children=[
                                dcc.Tab(label='Kollektiv', value='si-random-collective', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                                    html.Div(style={'paddingTop': '10px'}, children=[
                                        html.Label("Zufallsbereich Verzerrung:"),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '5px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-distortion-rand-min-collective', type='number', placeholder='Minimalwert', step=0.01, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-distortion-rand-max-collective', type='number', placeholder='Maximalwert', step=0.01, style={'width': '100%'})]),
                                        ]),
                                        html.Label("Zufallsbereich Rotation (Grad):", style={'marginTop': '10px'}),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '5px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-rotation-rand-min-collective', type='number', placeholder='Minimalwert', step=0.1, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-rotation-rand-max-collective', type='number', placeholder='Maximalwert', step=0.1, style={'width': '100%'})]),
                                        ]),
                                    ]),
                                ]),
                                dcc.Tab(label='Achsenspezifisch', value='si-random-specific', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                                    html.Div(style={'paddingTop': '10px'}, children=[
                                        html.Label("Verzerrung X-Achse:"),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '3px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-distortion-rand-min-x', type='number', placeholder='Min', step=0.01, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-distortion-rand-max-x', type='number', placeholder='Max', step=0.01, style={'width': '100%'})]),
                                        ]),
                                        html.Label("Verzerrung Y-Achse:", style={'marginTop': '10px'}),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '3px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-distortion-rand-min-y', type='number', placeholder='Min', step=0.01, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-distortion-rand-max-y', type='number', placeholder='Max', step=0.01, style={'width': '100%'})]),
                                        ]),
                                        html.Label("Verzerrung Z-Achse:", style={'marginTop': '10px'}),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '3px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-distortion-rand-min-z', type='number', placeholder='Min', step=0.01, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-distortion-rand-max-z', type='number', placeholder='Max', step=0.01, style={'width': '100%'})]),
                                        ]),
                                        html.Hr(style={'marginTop': '15px', 'marginBottom': '15px'}),
                                        html.Label("Rotation X-Achse:", style={'marginTop': '10px'}),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '3px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-rotation-rand-min-x', type='number', placeholder='Min', step=0.1, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-rotation-rand-max-x', type='number', placeholder='Max', step=0.1, style={'width': '100%'})]),
                                        ]),
                                        html.Label("Rotation Y-Achse:", style={'marginTop': '10px'}),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '3px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-rotation-rand-min-y', type='number', placeholder='Min', step=0.1, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-rotation-rand-max-y', type='number', placeholder='Max', step=0.1, style={'width': '100%'})]),
                                        ]),
                                        html.Label("Rotation Z-Achse:", style={'marginTop': '10px'}),
                                        html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '3px'}, children=[
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-rotation-rand-min-z', type='number', placeholder='Min', step=0.1, style={'width': '100%'})]),
                                            html.Div(style={'flex': '1'}, children=[dcc.Input(id='si-rotation-rand-max-z', type='number', placeholder='Max', step=0.1, style={'width': '100%'})]),
                                        ]),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]), 
            
            # 7. Metadaten & Buttons
            html.Div([
                html.Hr(style={'marginTop': '5px'}),
                html.P("Uhrzeit, der Erzeugung der Daten: (hh:mm:ss)", id='display-time', style={'fontSize': '0.8em', 'color': '#555'}),
                
                html.Button('Punkte Erzeugen', id='submit-button', n_clicks=0, style={'width': '100%', 'marginBottom': '15px'}),
                html.Button('Datenset(s) exportieren', id='export-dataset', n_clicks=0, style={'width': '100%', 'marginBottom': '15px'}),
                html.Div(id='export-status', children='', style={'color': 'green', 'textAlign': 'center', 'marginTop': '5px'}),

                # Mesh Opacity Slider
                html.Div(style=SECTION_STYLE, children=[
                    html.Label("Kugeltransparenz:", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='mesh-opacity-slider',
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        value=0.3,
                        marks={i/10: str(round(i/10, 1)) for i in range(0, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ]),
                html.Hr(),
                
            ], style={'marginTop': '10px'}),
        ]),

        # --- MITTLERER BEREICH (PLOT) ---
        html.Div([
            html.Div("❮", id="btn-toggle-left", n_clicks=0, style={**BUTTON_STYLE, "left": "0", "borderRadius": "0 8px 8px 0"}),
            html.Div("❯", id="btn-toggle-right", n_clicks=0, style={**BUTTON_STYLE, "right": "0", "borderRadius": "8px 0 0 8px"}),
            dcc.Graph(id='sphere-plot', style={'width': '100%', 'height': '100vh'}, config={
                'responsive': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'kalipoints_plot',
                    'width': 3840,
                    'height': 2160,
                    'scale': 2
                }
            })
        ], style={'flex': '1', 'position': 'relative', 'height': '100vh', 'overflow': 'hidden'}),

        # --- RECHTES FENSTER (RESULTATE) ---
        html.Div(id='right-sidebar', style={**SIDEBAR_STYLE, "border-right": "none", "border-left": "1px solid #ddd", "width": "400px"}, children=[
            html.H3("Resultate", style={'marginBottom': '20px', 'whiteSpace': 'nowrap'}),
            
            # Datenset Laden Section
            html.Div(style={'marginBottom': '20px', 'paddingBottom': '15px', 'borderBottom': '1px solid #ddd'}, children=[
                html.Label("Datenset laden:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center'}, children=[
                    dcc.Dropdown(
                        id='import-seed', 
                        options=get_available_seeds(), 
                        value=None, 
                        clearable=True, 
                        placeholder="Datensatz auswählen", 
                        style={'flexGrow': 1}
                    ),
                    html.Button('Laden', id='load-dataset-button', n_clicks=0, style=BUTTON_STYLE_INLINE)
                ]),
                html.Div(id='seed-load-status', children='', style={'fontSize': '0.8em', 'textAlign': 'center', 'marginTop': '5px'}),
            ]),
            
            # Results Container
            html.Div(id='results-container', children="Hier erscheinen später die Ergebnisse...", style={'padding': '10px', 'border': '1px dashed #ccc', 'height': '100%'})
        ]),
    ], style={'display': 'flex', 'flexDirection': 'row', 'height': '100vh', 'width': '100%', 'position': 'absolute', 'top': 0, 'left': 0, 'overflow': 'hidden'})
])

# ----------------------------------------------------
# 3. CALLBACKS 
# ----------------------------------------------------

# A. Callback für das Einklappen/Ausklappen
@app.callback(
    [Output('left-sidebar', 'style'), Output('right-sidebar', 'style'), Output('btn-toggle-left', 'children'), Output('btn-toggle-right', 'children'), Output('left-sidebar-state', 'data'), Output('right-sidebar-state', 'data')],
    [Input('btn-toggle-left', 'n_clicks'), Input('btn-toggle-right', 'n_clicks')],
    [State('left-sidebar-state', 'data'), State('right-sidebar-state', 'data')]
)
def toggle_sidebars(n_left, n_right, left_is_open, right_is_open):
    ctx = callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'btn-toggle-left': left_is_open = not left_is_open
        elif button_id == 'btn-toggle-right': right_is_open = not right_is_open
    left_style = SIDEBAR_STYLE.copy() if left_is_open else COLLAPSED_STYLE.copy()
    right_base = {**SIDEBAR_STYLE, "border-right": "none", "border-left": "1px solid #ddd", "width": "400px"}
    right_style = right_base if right_is_open else COLLAPSED_STYLE.copy()
    return left_style, right_style, "❮" if left_is_open else "❯", "❯" if right_is_open else "❮", left_is_open, right_is_open

# ============================================================
# 4. HELPER-FUNKTIONEN FÜR CALLBACKS
# ============================================================

def resolve_input_data(trigger_id, loaded_seed_data, ui_params):
    """
    Bestimmt die Eingabeparameter basierend auf der Quelle (Seed oder UI).
    Unterstützt sowohl alte als auch neue Metadaten-Formate.
    Rückgabe: Alle resolvierten Parameter als Dictionary
    """
    if trigger_id == 'seed-data-storage' and loaded_seed_data is not None:
        # Aus geladenem Seed
        data = loaded_seed_data
        
        # Extrahiere Hard Iron Offset (unterstütze beide Formate)
        hi = data.get('HI_OFFSET') or data.get('HI_X_Y_Z_OFFSET') or data.get('Hard_Iron', [0, 0, 0])
        
        # Extrahiere Soft Iron Distortion (unterstütze beide Formate)
        si_dist = data.get('SI_DISTORTION') or data.get('SI_X_Y_Z_DISTORTION') or data.get('Soft_Iron', [1.0, 1.0, 1.0])
        
        # Konvertiere distribution_style zu generation_mode
        distribution_style = data.get('DISTRIBUTION_STYLE', data.get('Punktegenerierung', 'optimal'))
        if distribution_style == 'evenly':
            generation_mode = 'optimal'
        elif distribution_style == 'randomly':
            generation_mode = 'random'
        else:
            generation_mode = distribution_style
        
        result = {
            'generation_mode': generation_mode,
            'noise_value': float(data.get('NOISE') or data.get('Fehlerabweichung_Start') or DEFAULT_NOISE),
            'sample_count': int(data.get('POINT_AMOUNT') or data.get('Kalibrierdauer/anzahlpunkte') or DEFAULT_SAMPLES),
            'alpha': float(data.get('ANGULAR_CONSTRAINT_DEG') or data.get('Winkeleinschränkung') or DEFAULT_ALPHA),
            'x_offset': float(hi[0]) if hi and len(hi) > 0 else 0.0,
            'y_offset': float(hi[1]) if hi and len(hi) > 1 else 0.0,
            'z_offset': float(hi[2]) if hi and len(hi) > 2 else 0.0,
            'x_distortion': float(si_dist[0]) if si_dist and len(si_dist) > 0 else 1.0,
            'y_distortion': float(si_dist[1]) if si_dist and len(si_dist) > 1 else 1.0,
            'z_distortion': float(si_dist[2]) if si_dist and len(si_dist) > 2 else 1.0,
            'from_seed': True
        }
    else:
        # Aus UI mit Default-Fallbacks
        sample_count_val = ui_params['sample_count'] if ui_params['sample_count'] is not None else DEFAULT_SAMPLES
        # Sanitize sample_count: Ensure it's in valid range and multiple of 100
        try:
            sc = int(sample_count_val)
            sc = max(100, min(10000, sc))  # Clamp to [100, 10000]
            sc = (sc // 100) * 100  # Round down to nearest 100
        except (ValueError, TypeError):
            sc = DEFAULT_SAMPLES
        
        result = {
            'generation_mode': ui_params['generation_mode'],
            'noise_value': ui_params['noise'] if ui_params['noise'] is not None else DEFAULT_NOISE,
            'sample_count': sc,
            'alpha': max(0.0, min(90.0, float(ui_params['alpha'] if ui_params['alpha'] is not None else DEFAULT_ALPHA))),
            'x_offset': ui_params['x_offset'] if ui_params['x_offset'] is not None else DEFAULT_OFFSET[0],
            'y_offset': ui_params['y_offset'] if ui_params['y_offset'] is not None else DEFAULT_OFFSET[1],
            'z_offset': ui_params['z_offset'] if ui_params['z_offset'] is not None else DEFAULT_OFFSET[2],
            'x_distortion': ui_params['x_distortion'] if ui_params['x_distortion'] is not None else DEFAULT_DISTORTION[0],
            'y_distortion': ui_params['y_distortion'] if ui_params['y_distortion'] is not None else DEFAULT_DISTORTION[1],
            'z_distortion': ui_params['z_distortion'] if ui_params['z_distortion'] is not None else DEFAULT_DISTORTION[2],
            'from_seed': False
        }
    
    return result

# C2. Callback: UI-Felder mit geladenen Seed-Daten füllen
@app.callback(
    [Output('x-offset-input', 'value'),
     Output('y-offset-input', 'value'),
     Output('z-offset-input', 'value'),
     Output('x-distortion-input', 'value'),
     Output('y-distortion-input', 'value'),
     Output('z-distortion-input', 'value'),
     Output('x-rotation', 'value'),
     Output('y-rotation', 'value'),
     Output('z-rotation', 'value'),
     Output('generation-mode', 'value'),
     Output('angular-constraint-input', 'value'),
     Output('sample-duration-dropdown', 'value'),
     Output('noise-input', 'value')],
    [Input('seed-data-storage', 'data')],
    prevent_initial_call=True
)
def populate_ui_from_seed(seed_data):
    """
    Füllt die UI-Eingabefelder mit Werten aus dem geladenen Seed.
    Unterstützt sowohl alte als auch neue Metadaten-Formate.
    """
    if seed_data is None or not isinstance(seed_data, dict):
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
    # Hard Iron Offset: Versuche neue und alte Formate
    hi = seed_data.get('HI_OFFSET') or seed_data.get('HI_X_Y_Z_OFFSET') or seed_data.get('Hard_Iron', [0, 0, 0])
    x_offset = hi[0] if hi and len(hi) > 0 else 0
    y_offset = hi[1] if hi and len(hi) > 1 else 0
    z_offset = hi[2] if hi and len(hi) > 2 else 0
    
    # Soft Iron Distortion: Versuche neue und alte Formate
    si_dist = seed_data.get('SI_DISTORTION') or seed_data.get('SI_X_Y_Z_DISTORTION') or seed_data.get('Soft_Iron', [1.0, 1.0, 1.0])
    x_distortion = si_dist[0] if si_dist and len(si_dist) > 0 else 1.0
    y_distortion = si_dist[1] if si_dist and len(si_dist) > 1 else 1.0
    z_distortion = si_dist[2] if si_dist and len(si_dist) > 2 else 1.0
    
    # Soft Iron Rotation: Versuche neue und alte Formate
    si_rot = seed_data.get('SI_ROTATION_DEG') or seed_data.get('SI_X_Y_Z_ROTATION_DEG') or seed_data.get('Soft_Iron_Rotation', [0.0, 0.0, 0.0])
    x_rotation = si_rot[0] if si_rot and len(si_rot) > 0 else 0.0
    y_rotation = si_rot[1] if si_rot and len(si_rot) > 1 else 0.0
    z_rotation = si_rot[2] if si_rot and len(si_rot) > 2 else 0.0
    
    # Weitere Parameter (neue Format unterstützen)
    # Konvertiere distribution_style zu generation_mode
    distribution_style = seed_data.get('DISTRIBUTION_STYLE') or seed_data.get('Punktegenerierung', 'optimal')
    if distribution_style in ('evenly', 'randomly'):
        generation_mode = 'optimal' if distribution_style == 'evenly' else 'random'
    else:
        generation_mode = distribution_style
    
    alpha = seed_data.get('ANGULAR_CONSTRAINT_DEG') or seed_data.get('Winkeleinschränkung') or DEFAULT_ALPHA
    sample_count = seed_data.get('POINT_AMOUNT') or seed_data.get('Kalibrierdauer/anzahlpunkte') or DEFAULT_SAMPLES
    noise = seed_data.get('NOISE') or seed_data.get('Fehlerabweichung_Start') or DEFAULT_NOISE
    
    return x_offset, y_offset, z_offset, x_distortion, y_distortion, z_distortion, x_rotation, y_rotation, z_rotation, generation_mode, alpha, sample_count, noise

# D. Callback für den Graphen (Berechnung)
@app.callback(
    [Output('sphere-plot', 'figure'),
     Output('results-container', 'children'),
     Output('point-data-storage', 'data'),
     Output('display-time', 'children'),
     Output('seed-data-storage', 'data', allow_duplicate=True),
     Output('current-seed-store', 'data'),
     Output('export-filename-store', 'data'),
     Output('magnetic-error-values-store', 'data')],
    [Input('submit-button', 'n_clicks'),
     Input('seed-data-storage', 'data')],
    [State('sample-duration-dropdown', 'value'), 
     State('angular-constraint-input', 'value'),
     State('x-offset-input', 'value'), State('y-offset-input', 'value'), State('z-offset-input', 'value'),
     State('x-distortion-input', 'value'), State('y-distortion-input', 'value'), State('z-distortion-input', 'value'),
     State('x-rotation', 'value'), State('y-rotation', 'value'), State('z-rotation', 'value'),
     State('generation-mode', 'value'),
     State('noise-input', 'value'),
     State('hard-iron-mode-tabs', 'value'),
     State('hard-iron-random-type-tabs', 'value'),
     State('offset-rand-min-collective', 'value'), State('offset-rand-max-collective', 'value'),
     State('offset-rand-min-x', 'value'), State('offset-rand-max-x', 'value'),
     State('offset-rand-min-y', 'value'), State('offset-rand-max-y', 'value'),
     State('offset-rand-min-z', 'value'), State('offset-rand-max-z', 'value'),
     State('soft-iron-mode-tabs', 'value'),
     State('soft-iron-random-type-tabs', 'value'),
     State('si-distortion-rand-min-collective', 'value'), State('si-distortion-rand-max-collective', 'value'),
     State('si-rotation-rand-min-collective', 'value'), State('si-rotation-rand-max-collective', 'value'),
     State('si-distortion-rand-min-x', 'value'), State('si-distortion-rand-max-x', 'value'),
     State('si-distortion-rand-min-y', 'value'), State('si-distortion-rand-max-y', 'value'),
     State('si-distortion-rand-min-z', 'value'), State('si-distortion-rand-max-z', 'value'),
     State('si-rotation-rand-min-x', 'value'), State('si-rotation-rand-max-x', 'value'),
     State('si-rotation-rand-min-y', 'value'), State('si-rotation-rand-max-y', 'value'),
     State('si-rotation-rand-min-z', 'value'), State('si-rotation-rand-max-z', 'value'),
     State('mesh-opacity-slider', 'value'),
     State('loaded-seed-string-store', 'data')], 
    prevent_initial_call=True
)
def update_graph(n_clicks, loaded_seed_data, ui_sample_count, ui_angular_constraint, ui_x_offset, ui_y_offset, 
                 ui_z_offset, ui_x_distortion, ui_y_distortion, ui_z_distortion, x_rotation, y_rotation, 
                 z_rotation, ui_generation_mode, ui_noise, hard_iron_mode, hard_iron_random_type_mode,
                 hi_rand_min_collective, hi_rand_max_collective,
                 hi_rand_min_x, hi_rand_max_x, hi_rand_min_y, hi_rand_max_y, hi_rand_min_z, hi_rand_max_z,
                 soft_iron_mode, soft_iron_random_type_mode,
                 si_dist_rand_min_collective, si_dist_rand_max_collective,
                 si_rot_rand_min_collective, si_rot_rand_max_collective,
                 si_dist_rand_min_x, si_dist_rand_max_x, si_dist_rand_min_y, si_dist_rand_max_y, si_dist_rand_min_z, si_dist_rand_max_z,
                 si_rot_rand_min_x, si_rot_rand_max_x, si_rot_rand_min_y, si_rot_rand_max_y, si_rot_rand_min_z, si_rot_rand_max_z,
                 ui_mesh_opacity, loaded_seed_string):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    current_time = datetime.now()
    current_time_str = current_time.strftime("%H:%M:%S")

    # Resolve all input parameters
    ui_params = {
        'generation_mode': ui_generation_mode,
        'noise': ui_noise,
        'sample_count': ui_sample_count,
        'alpha': ui_angular_constraint,
        'x_offset': ui_x_offset,
        'y_offset': ui_y_offset,
        'z_offset': ui_z_offset,
        'x_distortion': ui_x_distortion,
        'y_distortion': ui_y_distortion,
        'z_distortion': ui_z_distortion
    }
    
    params = resolve_input_data(trigger_id, loaded_seed_data, ui_params)
    
    # --- HARD IRON OFFSETS INITIALISIEREN ---
    # Diese werden entweder manuell gesetzt oder zufällig generiert
    if hard_iron_mode == 'hard-iron-manual':
        # Verwende manuelle Eingaben
        hi_x_offset = params['x_offset']
        hi_y_offset = params['y_offset']
        hi_z_offset = params['z_offset']
    elif hard_iron_mode == 'hard-iron-random':
        # Generiere zufällige Offsets
        if hard_iron_random_type_mode == 'hi-random-collective':
            # Kollektiv: Gleicher Min/Max Range für alle Achsen
            min_val = float(hi_rand_min_collective) if hi_rand_min_collective is not None else 0.0
            max_val = float(hi_rand_max_collective) if hi_rand_max_collective is not None else 0.0
            hi_x_offset = rng.uniform(min_val, max_val)
            hi_y_offset = rng.uniform(min_val, max_val)
            hi_z_offset = rng.uniform(min_val, max_val)
        else:  # hi-random-specific
            # Achsenspezifisch: Unterschiedliche Min/Max für jede Achse
            min_x = float(hi_rand_min_x) if hi_rand_min_x is not None else 0.0
            max_x = float(hi_rand_max_x) if hi_rand_max_x is not None else 0.0
            min_y = float(hi_rand_min_y) if hi_rand_min_y is not None else 0.0
            max_y = float(hi_rand_max_y) if hi_rand_max_y is not None else 0.0
            min_z = float(hi_rand_min_z) if hi_rand_min_z is not None else 0.0
            max_z = float(hi_rand_max_z) if hi_rand_max_z is not None else 0.0
            hi_x_offset = rng.uniform(min_x, max_x)
            hi_y_offset = rng.uniform(min_y, max_y)
            hi_z_offset = rng.uniform(min_z, max_z)
    else:
        # Fallback: Verwende Defaults
        hi_x_offset = 0.0
        hi_y_offset = 0.0
        hi_z_offset = 0.0
    
    # KRITISCH: Unterscheide zwischen zwei Fällen:
    # 1. Seed wurde von außen geladen (trigger_id == 'seed-data-storage') -> LADE PUNKTE AUS CSV!
    # 2. Benutzer klickt "Punkte Erzeugen" (trigger_id == 'submit-button') -> generiere NEUEN SEED
    
    csv_metadata = None  # Track echte CSV-Metadaten für die Anzeige
    
    if trigger_id == 'seed-data-storage' and loaded_seed_string:
        # Seed wurde von außen geladen -> VERWENDE GESPEICHERTE PUNKTE AUS CSV!
        seed_string = loaded_seed_string
        display_time_str = f"Uhrzeit (aus Datensatz): ({current_time_str})"
        
        # Lade die Punkte aus der CSV-Datei
        data_points, csv_metadata, load_error = load_csv_data_by_seed(seed_string)
        
        if load_error or data_points is None:
            # Fallback: Generiere Punkte wenn CSV nicht existiert
            si_x_scale = params['x_distortion']
            si_y_scale = params['y_distortion']
            si_z_scale = params['z_distortion']
            si_x_rot = x_rotation if x_rotation else 0
            si_y_rot = y_rotation if y_rotation else 0
            si_z_rot = z_rotation if z_rotation else 0
            
            data_points, actual_hi_offset = fibonacci_sphere(
                params['generation_mode'], params['noise_value'], params['sample_count'], 
                params['alpha'], seed_string, 
                xyz0=[hi_x_offset, hi_y_offset, hi_z_offset],
                xyz1=[si_x_scale, si_y_scale, si_z_scale],
                x_rot=si_x_rot,
                y_rot=si_y_rot,
                z_rot=si_z_rot
            )
            csv_metadata = None  # Fallback, keine CSV-Metadaten
        else:
            # Nutze die Metadaten aus der CSV-Datei (csv_metadata bleibt mit None-Werten!)
            actual_hi_offset = csv_metadata.get('HI_OFFSET') or csv_metadata.get('HI_X_Y_Z_OFFSET') or [params['x_offset'], params['y_offset'], params['z_offset']]
            si_dist = csv_metadata.get('SI_DISTORTION') or csv_metadata.get('SI_X_Y_Z_DISTORTION') or [params['x_distortion'], params['y_distortion'], params['z_distortion']]
            si_x_scale = si_dist[0] if si_dist and len(si_dist) > 0 else params['x_distortion']
            si_y_scale = si_dist[1] if si_dist and len(si_dist) > 1 else params['y_distortion']
            si_z_scale = si_dist[2] if si_dist and len(si_dist) > 2 else params['z_distortion']
            # Nutze SI_ROTATION_DEG aus den neuen Metadaten, fallback zu SI_ROTATION für alte Formate
            si_rot = csv_metadata.get('SI_ROTATION_DEG') or csv_metadata.get('SI_ROTATION', [0.0, 0.0, 0.0])
            si_x_rot = si_rot[0] if si_rot and len(si_rot) > 0 else 0.0
            si_y_rot = si_rot[1] if si_rot and len(si_rot) > 1 else 0.0
            si_z_rot = si_rot[2] if si_rot and len(si_rot) > 2 else 0.0
    else:
        # Benutzer hat "Punkte Erzeugen" geklickt -> GENERIERE NEUEN SEED
        # Einfacher Seed-String für Anzeige und Kompatibilität
        seed_string = f"{int(params['alpha'])}_{params['generation_mode']}_{current_time.strftime('%y-%m-%d_%H-%M-%S')}"
        display_time_str = f"Uhrzeit: ({current_time_str})"
        
        # --- SOFT IRON RANDOM LOGIC ---
        # Wenn Soft Iron im Zufalls-Modus ist, generiere zufällige Parameter
        if soft_iron_mode == 'soft-iron-manual':
            # Verwende manuelle Eingaben
            si_x_scale = params['x_distortion']
            si_y_scale = params['y_distortion']
            si_z_scale = params['z_distortion']
            si_x_rot = x_rotation if x_rotation else 0
            si_y_rot = y_rotation if y_rotation else 0
            si_z_rot = z_rotation if z_rotation else 0
        elif soft_iron_mode == 'soft-iron-random':
            # Generiere zufällige Soft Iron Parameter
            if soft_iron_random_type_mode == 'si-random-collective':
                # Kollektiv: Gleicher Min/Max Range für alle Achsen (Verzerrung und Rotation)
                dist_min = float(si_dist_rand_min_collective) if si_dist_rand_min_collective is not None else 1.0
                dist_max = float(si_dist_rand_max_collective) if si_dist_rand_max_collective is not None else 1.0
                rot_min = float(si_rot_rand_min_collective) if si_rot_rand_min_collective is not None else 0.0
                rot_max = float(si_rot_rand_max_collective) if si_rot_rand_max_collective is not None else 0.0
                
                si_x_scale = rng.uniform(dist_min, dist_max)
                si_y_scale = rng.uniform(dist_min, dist_max)
                si_z_scale = rng.uniform(dist_min, dist_max)
                si_x_rot = rng.uniform(rot_min, rot_max)
                si_y_rot = rng.uniform(rot_min, rot_max)
                si_z_rot = rng.uniform(rot_min, rot_max)
            else:  # si-random-specific
                # Achsenspezifisch: Unterschiedliche Min/Max für jede Achse und Parameter
                dist_min_x = float(si_dist_rand_min_x) if si_dist_rand_min_x is not None else 1.0
                dist_max_x = float(si_dist_rand_max_x) if si_dist_rand_max_x is not None else 1.0
                dist_min_y = float(si_dist_rand_min_y) if si_dist_rand_min_y is not None else 1.0
                dist_max_y = float(si_dist_rand_max_y) if si_dist_rand_max_y is not None else 1.0
                dist_min_z = float(si_dist_rand_min_z) if si_dist_rand_min_z is not None else 1.0
                dist_max_z = float(si_dist_rand_max_z) if si_dist_rand_max_z is not None else 1.0
                
                rot_min_x = float(si_rot_rand_min_x) if si_rot_rand_min_x is not None else 0.0
                rot_max_x = float(si_rot_rand_max_x) if si_rot_rand_max_x is not None else 0.0
                rot_min_y = float(si_rot_rand_min_y) if si_rot_rand_min_y is not None else 0.0
                rot_max_y = float(si_rot_rand_max_y) if si_rot_rand_max_y is not None else 0.0
                rot_min_z = float(si_rot_rand_min_z) if si_rot_rand_min_z is not None else 0.0
                rot_max_z = float(si_rot_rand_max_z) if si_rot_rand_max_z is not None else 0.0
                
                si_x_scale = rng.uniform(dist_min_x, dist_max_x)
                si_y_scale = rng.uniform(dist_min_y, dist_max_y)
                si_z_scale = rng.uniform(dist_min_z, dist_max_z)
                si_x_rot = rng.uniform(rot_min_x, rot_max_x)
                si_y_rot = rng.uniform(rot_min_y, rot_max_y)
                si_z_rot = rng.uniform(rot_min_z, rot_max_z)
        else:
            # Fallback: Verwende Defaults (keine Verzerrung, keine Rotation)
            si_x_scale = 1.0
            si_y_scale = 1.0
            si_z_scale = 1.0
            si_x_rot = 0.0
            si_y_rot = 0.0
            si_z_rot = 0.0
        
        # Generate points mit Soft Iron Rotation
        data_points, actual_hi_offset = fibonacci_sphere(
            params['generation_mode'], params['noise_value'], params['sample_count'], 
            params['alpha'], seed_string, 
            xyz0=[hi_x_offset, hi_y_offset, hi_z_offset],
            xyz1=[si_x_scale, si_y_scale, si_z_scale],
            x_rot=si_x_rot,
            y_rot=si_y_rot,
            z_rot=si_z_rot
        )

    # Create mesh centered at origin (0,0,0) first
    center_offset = actual_hi_offset
    sphere_vertices, sphere_triangles, _ = create_sphere_mesh(xyz0=[0, 0, 0])
    
    # --- APPLY SOFT IRON TRANSFORMATION TO MESH VERTICES ---
    # Das Mesh wird mit den gleichen Transformationen verformt wie die Datenpunkte
    # Transformation erfolgt um den Ursprung, BEVOR der Offset addiert wird
    transformed_vertices = sphere_vertices.copy()
    if si_x_scale != 1.0 or si_y_scale != 1.0 or si_z_scale != 1.0 or \
       si_x_rot or si_y_rot or si_z_rot:
        # Wende Soft Iron Transformation auf die Mesh-Vertices an
        transformed_vertices = apply_soft_iron_transformation(
            sphere_vertices, 
            si_x_scale, si_y_scale, si_z_scale,
            si_x_rot, si_y_rot, si_z_rot
        )
    
    # Addiere den Hard Iron Offset zum transformierten Mesh
    transformed_vertices[:, 0] += center_offset[0]
    transformed_vertices[:, 1] += center_offset[1]
    transformed_vertices[:, 2] += center_offset[2]
    
    # Bestimme, ob das Mesh angezeigt werden soll und welche Farbe die Punkte haben
    # - Wenn Punkte geladen wurden: Kein Mesh, rote Punkte
    # - Wenn Punkte erzeugt wurden: Mesh angezeigen, blaue Punkte
    show_mesh = False
    point_color = 'darkred'
    
    if trigger_id == 'submit-button' or (trigger_id == 'seed-data-storage' and not loaded_seed_string):
        # Punkte wurden erzeugt (nicht geladen)
        show_mesh = True
        point_color = 'blue'
    
    # Build figure mit transformierten Mesh-Vertices
    fig = build_figure_with_points(
        data_points, center_offset, transformed_vertices, sphere_triangles, 
        params['alpha'], params['noise_value'], si_x_scale, 
        si_y_scale, si_z_scale, ui_mesh_opacity, 
        show_mesh=show_mesh, point_color=point_color
    )
    
    # Berechne die 4x4 Iron Error Matrix EARLY (für Results-Display)
    iron_error_matrix = create_iron_error_matrix(
        si_x_scale, si_y_scale, si_z_scale,
        si_x_rot, si_y_rot, si_z_rot,
        center_offset[0], center_offset[1], center_offset[2]
    )
    
    # Build results display mit vollständigen Metadaten
    # Nutze csv_metadata wenn vorhanden (enthält None für fehlende Felder)
    result_text = build_results_display(
        data_points, params['alpha'], params['generation_mode'], 
        params['noise_value'], center_offset, si_x_scale, 
        si_y_scale, si_z_scale, si_x_rot, si_y_rot, si_z_rot,
        iron_error_matrix=iron_error_matrix, sample_count=params['sample_count'],
        raw_metadata=csv_metadata
    )
    
    # Export data
    export_points_xyz = data_points[:, [0, 1, 2]]
    points_data = pd.DataFrame(export_points_xyz, columns=['X', 'Y', 'Z']).to_json(orient='split')
    seed_display = f"Seed: {seed_string}"
    
    # Speichere die tatsächlich verwendeten magnetischen Fehlerwerte + erweiterte Metadaten
    # Konvertiere die Matrix zu einer Liste von Listen für JSON-Serialisierung
    iron_error_matrix_list = iron_error_matrix.tolist()
    
    magnetic_errors = {
        'DISTRIBUTION_STYLE': params['generation_mode'],
        'NOISE': params['noise_value'],
        'POINT_AMOUNT': params['sample_count'],
        'ANGULAR_CONSTRAINT_DEG': params['alpha'],
        'HI': [center_offset[0], center_offset[1], center_offset[2]],
        'HI_X_Y_Z_OFFSET': [center_offset[0], center_offset[1], center_offset[2]],
        'SI_DISTORTION': [si_x_scale, si_y_scale, si_z_scale],
        'SI_X_Y_Z_DISTORTION': [si_x_scale, si_y_scale, si_z_scale],
        'SI_ROTATION': [si_x_rot, si_y_rot, si_z_rot],
        'SI_ROTATION_DEG': [si_x_rot, si_y_rot, si_z_rot],
        'SI_X_Y_Z_ROTATION_DEG': [si_x_rot, si_y_rot, si_z_rot],
        'IRON_ERROR_MATRIX_RAD': iron_error_matrix_list
    }
    
    # Generiere einfachen Dateinamen: {alpha}_{TT-MM-JJ}_{HH-MM}.csv
    export_filename = f"{int(params['alpha'])}_{current_time.strftime('%d-%m-%y')}_{current_time.strftime('%H-%M')}.csv"
    
    return fig, result_text, points_data, display_time_str, None, None, export_filename, magnetic_errors

# E. Callback für den Export der Punktedaten
@app.callback(
    [Output("download-csv", "data"),
     Output('export-status', 'children'),
     Output('export-success-trigger', 'data')],
    [Input("export-dataset", "n_clicks")],
    [State('point-data-storage', 'data'),
     State('export-filename-store', 'data'),
     State('magnetic-error-values-store', 'data')]
)
def export_dataset(n_clicks, json_data, export_filename, magnetic_errors):
    if n_clicks > 0 and json_data is not None and export_filename:
        df = pd.read_json(json_data, orient='split')
        full_path = os.path.join(OUTPUT_DIR, export_filename)
        
        # Nutze die tatsächlich generierten Werte aus dem Store
        if magnetic_errors:
            distribution_style = magnetic_errors.get('DISTRIBUTION_STYLE', 'evenly')
            noise = magnetic_errors.get('NOISE', 0.0)
            point_amount = magnetic_errors.get('POINT_AMOUNT', 1000)
            angular_constraint_deg = magnetic_errors.get('ANGULAR_CONSTRAINT_DEG', 90.0)
            
            hi = magnetic_errors.get('HI', [0, 0, 0])
            si_dist = magnetic_errors.get('SI_DISTORTION', [1.0, 1.0, 1.0])
            si_rot = magnetic_errors.get('SI_ROTATION', [0.0, 0.0, 0.0])
            
            ox, oy, oz = hi[0], hi[1], hi[2]
            x_distortion, y_distortion, z_distortion = si_dist[0], si_dist[1], si_dist[2]
            x_rotation, y_rotation, z_rotation = si_rot[0], si_rot[1], si_rot[2]
            
            iron_error_matrix = magnetic_errors.get('IRON_ERROR_MATRIX_RAD', None)
        else:
            # Fallback auf Defaults
            distribution_style = 'evenly'
            noise = 0.0
            point_amount = 1000
            angular_constraint_deg = 90.0
            
            ox, oy, oz = 0, 0, 0
            x_distortion, y_distortion, z_distortion = 1.0, 1.0, 1.0
            x_rotation, y_rotation, z_rotation = 0.0, 0.0, 0.0
            iron_error_matrix = None

        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            with open(full_path, 'w', newline='') as f:
                # Schreibe die erweiterten Metadaten in die richtige Reihenfolge
                f.write(f"# DISTRIBUTION_STYLE: {distribution_style}\n")
                f.write(f"# NOISE: {noise}\n")
                f.write(f"# POINT_AMOUNT: {point_amount}\n")
                f.write(f"# ANGULAR_CONSTRAINT-DEG: {angular_constraint_deg}\n")
                f.write(f"# FIELD_LINE_ANGLE-DEG: 0 (-> nicht implementiert)\n")
                f.write(f"# HI-X-Y-Z-OFFSET: {ox},{oy},{oz}\n")
                f.write(f"# SI-X-Y-Z-DISTORTION: {x_distortion},{y_distortion},{z_distortion}\n")
                f.write(f"# SI-X-Y-Z-ROTATION-DEG: {x_rotation},{y_rotation},{z_rotation}\n")
                
                # Schreibe die 4x4 Iron Error Matrix, falls vorhanden
                if iron_error_matrix is not None:
                    f.write("# IRON_ERROR_MATRIX_FORMAT-RAD:\n")
                    for row in iron_error_matrix:
                        f.write(f"# [{', '.join([f'{val:.8f}' for val in row])}]\n")
                
                # Schreibe die Datenpunkte
                df.to_csv(f, index=False, sep=';', float_format='%.8f')
            
            return None, f"✓ Export erfolgreich: {export_filename}", n_clicks

        except Exception as e:
            return None, html.Span(f"✗ Fehler: {e}", style={'color': 'red'}), no_update
        
    return None, "", no_update

# F. Callback: Dropdown-Liste aktualisieren
@app.callback(
    Output('import-seed', 'options'),
    [Input('export-success-trigger', 'data')], 
    prevent_initial_call=True
)
def update_seed_dropdown(trigger_value):
    return get_available_seeds()

# F2. Callback: Datensatz laden (WICHTIG: Triggeriert den Graph-Callback über seed-data-storage)
@app.callback(
    [Output('seed-data-storage', 'data'),
     Output('loaded-seed-string-store', 'data'),
     Output('seed-load-status', 'children')],
    [Input('load-dataset-button', 'n_clicks')],
    [State('import-seed', 'value')],
    prevent_initial_call=True
)
def load_dataset(n_clicks, selected_seed):
    """
    Lädt ein Datensatz aus der CSV-Datei und speichert die Metadaten im Store.
    Dies triggeriert dann automatisch den update_graph Callback.
    """
    if n_clicks == 0 or selected_seed is None:
        return no_update, no_update, ""
    
    # Lade die Datei
    data_points, metadata, load_error = load_csv_data_by_seed(selected_seed)
    
    if load_error:
        return no_update, no_update, html.Span(f"✗ {load_error}", style={'color': 'red', 'fontSize': '0.8em'})
    
    if data_points is None or metadata is None:
        return no_update, no_update, html.Span(f"✗ Fehler beim Laden: Datei konnte nicht korrekt gelesen werden", style={'color': 'red', 'fontSize': '0.8em'})
    
    # Speichere die Metadaten im Store (triggeriert update_graph Callback)
    return metadata, selected_seed, html.Span(f"✓ Datensatz geladen: {selected_seed}", style={'color': 'green', 'fontSize': '0.8em'})

# G. Callback für Mesh Opacity - OPTIMIERT
@app.callback(
    Output('sphere-plot', 'figure', allow_duplicate=True),
    [Input('mesh-opacity-slider', 'value')],
    [State('sphere-plot', 'figure')],
    prevent_initial_call=True
)
def update_mesh_opacity(new_opacity, current_fig):
    """
    OPTIMIERT: Updated nur die Opacity des Mesh3d-Traces,
    ohne die ganze Figure neu zu bauen. Das ist deutlich schneller!
    """
    if not current_fig:
        return no_update
    
    # Schnelle Shallow Copy statt vollständiger Rebuild
    fig_copy = go.Figure(current_fig)
    
    # Finde und update nur den Mesh3d Trace
    for trace in fig_copy.data:
        if trace.type == 'mesh3d':
            trace.opacity = new_opacity
            break
    
    return fig_copy

if __name__ == '__main__':
    app.run(debug=True)