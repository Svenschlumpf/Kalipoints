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

rng = np.random.default_rng(2) # Erzeugung eines Zufallsgenerators

# ============================================================
# KONSTANTEN & KONFIGURATION
# ============================================================
VALID_SAMPLE_COUNTS = [200, 500, 1000, 2000, 4000, 8000]
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

def generate_random_soft_iron_params(seed_string, max_distortion=2.0, max_rotation=10.0):
    """
    Generiert zufällige Soft Iron Parameter (Verzerrung und Rotation) aus einem Seed.
    
    Args:
        seed_string: Seed-String zur Reproduzierbarkeit
        max_distortion: Maximale Verzerrung pro Achse (Standard: 2.0)
        max_rotation: Maximale Rotation pro Achse in Grad (Standard: 10.0)
    
    Returns:
        Tuple: (x_scale, y_scale, z_scale, x_rot, y_rot, z_rot)
    """
    # Defaultwerte, falls None
    max_distortion = max_distortion if max_distortion is not None else 2.0
    max_rotation = max_rotation if max_rotation is not None else 10.0
    
    # Seed für Zufallsgenerator
    clean_seed = clean_seed_string(seed_string)
    rng = np.random.default_rng(int(clean_seed) if clean_seed else 0)
    
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
        seed_string: Seed für Reproduzierbarkeit
        xyz0: Hard Iron Offset [x, y, z]
        xyz1: Soft Iron Skalierung [x_scale, y_scale, z_scale]
        x_rot, y_rot, z_rot: Rotationen in Grad
    
    Rückgabe: (points_array, used_offset_list)
    """
    # Seed parsen für Hard Iron Random Logic
    clean_seed = clean_seed_string(seed_string)
    rng = np.random.default_rng(int(clean_seed) if clean_seed else 0) 

    # --- Hard Iron Random Logic aus Seed lesen ---
    current_offset = list(xyz0) # Kopie der manuellen Offsets (Standard)
    
    if seed_string:
        try:
            parts = seed_string.split('_')
            if len(parts) >= 5:
                hi_val = int(parts[4]) # Das HI Segment
                
                # Wenn HI zwischen 1 und 9 ist -> Zufallsmodus aktiv
                if 1 <= hi_val <= 9:
                    max_dev = float(hi_val)
                    # Erzeuge zufällige Offsets zwischen 0 und max_dev für x, y, z
                    ox = rng.uniform(0, max_dev)
                    oy = rng.uniform(0, max_dev)
                    oz = rng.uniform(0, max_dev)
                    current_offset = [ox, oy, oz]
        except:
            pass # Fallback auf manuelles xyz0 falls Parsing scheitert
    
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

def encode_custom_float(value):
    """ Wandelt Zahlen in das Format '000-0' um. """
    if value == 0:
        return "000-0"
    if value % 1 == 0:
        mantissa = int(value)
        exponent = 0
    else:
        mantissa = int(round(value * 1000))
        exponent = 3
    return f"{mantissa:03d}-{exponent}"

def decode_custom_float(string_val):
    """ Wandelt String '125-3' zurück in float 0.125. """
    try:
        mantissa_str, exponent_str = string_val.split('-')
        mantissa = float(mantissa_str)
        exponent = int(exponent_str)
        return mantissa * (10 ** -exponent)
    except:
        return 0.0

def encode_magnetic_error(value, max_value=9.9):
    """ 
    Wandelt ein Magnetic Error (HI oder SI Distortion) in das Format '00-99' um.
    Format: Die Zahl wird mit 10 multipliziert und als Ganzzahl dargestellt.
    Beispiel: 0.1 -> 01, 4.0 -> 40, 9.9 -> 99
    Bei 0 oder Manual Mode -> '00'
    """
    if value == 0:
        return "00"
    try:
        encoded = int(round(value * 10))
        return f"{encoded:02d}"
    except:
        return "00"

def decode_magnetic_error(string_val):
    """ 
    Wandelt String '01' oder '40' zurück in float (0.1 oder 4.0).
    '00' bedeutet Manual Mode oder 0.0.
    """
    try:
        encoded = int(string_val)
        return encoded / 10.0
    except:
        return 0.0

def encode_magnetic_rotation(value):
    """ 
    Wandelt ein Rotation-Winkel in das Format '000-360' um.
    Bei 0 oder Manual Mode -> '000'
    Beispiel: 45.5 -> '046' (gekürzt als 046 wenn < 100)
    """
    if value == 0:
        return "000"
    try:
        encoded = int(round(value))
        return f"{encoded:03d}"
    except:
        return "000"

def decode_magnetic_rotation(string_val):
    """ 
    Wandelt String '046' zurück in float Rotation-Winkel (45.5 -> 45 oder 46).
    '000' bedeutet Manual Mode oder 0.0.
    """
    try:
        encoded = int(string_val)
        return float(encoded)
    except:
        return 0.0

def generate_seed_string(alpha, samples, gen_mode, noise_value, offset_mode, offset_rand, 
                        distortion_mode, distortion_rand, distortion_rand_rotation,
                        hi_x=0.0, hi_y=0.0, hi_z=0.0,
                        si_x=1.0, si_y=1.0, si_z=1.0,
                        si_rot_x=0.0, si_rot_y=0.0, si_rot_z=0.0,
                        current_time=None):
    """
    Generiert einen Seed-String basierend auf den Eingabeparametern.
    Neue Format (11 Segmente, getrennt durch _):
    alpha_samples_gen_mode_noise_hi_si_si_rotation_date_time
    
    HI/SI Parameter:
    - offset_mode/distortion_mode 'random' -> kodiert den Zufallsbereich (01-99 oder 001-360)
    - offset_mode/distortion_mode 'manual' -> kodiert die tatsächlichen Werte (00 = manual überschrieben)
    """
    if current_time is None:
        current_time = datetime.now()
    
    s_alpha = int(alpha)
    s_samples = int(samples)
    mode_map = {'optimal': 0, 'random': 1, 'path': 2}
    s_gen_mode = mode_map.get(gen_mode, 0)
    s_noise = encode_custom_float(noise_value)
    
    # --- Hard Iron Encoding ---
    # Wenn offset_mode == 'random': kodiere den Zufallsbereich (offset_rand = 1-9)
    # Wenn offset_mode == 'manual': kodiere die tatsächlichen Werte
    if offset_mode == 'random':
        try:
            val = int(offset_rand) if offset_rand is not None else 1
            s_hard_iron = max(1, min(9, val))  # Bereich 1-9
        except:
            s_hard_iron = 1
        s_hard_iron = str(s_hard_iron)
    else:
        # Manual Mode: kodiere die Durchschnittswerte (optional)
        s_hard_iron = "00"
    
    # --- Soft Iron Encoding ---
    # Wenn distortion_mode == 'random': kodiere die Zufallsbereiche
    # Wenn distortion_mode == 'manual': kodiere die tatsächlichen Werte
    if distortion_mode == 'random':
        # Kodiere den max_distortion Bereich
        max_dist = float(distortion_rand) if distortion_rand else 2.0
        s_soft_iron_dist = encode_magnetic_error(max_dist)
        
        # Kodiere den max_rotation Bereich
        max_rot = float(distortion_rand_rotation) if distortion_rand_rotation else 10.0
        s_soft_iron_rot = encode_magnetic_rotation(max_rot)
    else:
        # Manual Mode: kodiere 00 (bedeutet: Seed enthält keine SI-Info, verwende UI-Werte)
        s_soft_iron_dist = "00"
        s_soft_iron_rot = "000"
    
    s_date = current_time.strftime("%y-%m-%d")
    s_time = current_time.strftime("%H-%M-%S")
    
    # Neues Format mit 11 Segmenten
    return f"{s_alpha}_{s_samples}_{s_gen_mode}_{s_noise}_{s_hard_iron}_{s_soft_iron_dist}_{s_soft_iron_rot}_{s_date}_{s_time}"

def build_figure_with_points(data_points, center_offset, sphere_vertices, sphere_triangles, 
                             alpha, noise_value, x_distortion, y_distortion, z_distortion, 
                             mesh_opacity):
    """
    Erstellt eine Plotly Figure mit Kugel-Mesh und Datenpunkten.
    Die Achsenlängen sind ALLE gleich (keine Skalenverzerrung).
    Die Skala wird so angepasst, dass alle Punkte (Mesh + Datenpunkte) sichtbar sind.
    """
    # Bestimme die Bounding Box aller Punkte (Mesh + Datenpunkte)
    all_points = np.vstack([sphere_vertices, data_points])
    
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
    
    # Cache intensity calculation
    intensity = np.linspace(0, 1, len(sphere_vertices))
    
    data_traces = [
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
        ),
        go.Scatter3d(
            x=data_points[:, 0], 
            y=data_points[:, 1], 
            z=data_points[:, 2], 
            mode='markers', 
            marker=dict(size=2, color='blue', opacity=1.0), 
            name=f'Gesamtpunkte ({len(data_points)})', 
            hoverinfo='none'
        )
    ]
    
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
                         x_distortion, y_distortion, z_distortion, x_rot=0, y_rot=0, z_rot=0):
    """
    Erstellt die Ergebnis-Anzeige für die rechte Sidebar.
    """
    return [
        html.P(f"Anzahl Punkte: {len(data_points)}"),
        html.P(f"Eingestellter Winkel: {alpha}°"),
        html.P(f"Modus: {'Gleichmässig' if generation_mode == 'optimal' else 'Zufall' if generation_mode == 'random' else 'Pfad'}"),
        html.P(f"Rauschen: {noise_value:.2f}"),
        html.P(f"Hard Iron (X,Y,Z): ({center_offset[0]:.2f}, {center_offset[1]:.2f}, {center_offset[2]:.2f})"),
        html.P(f"Soft Iron Verzerrung (X,Y,Z): ({x_distortion:.2f}, {y_distortion:.2f}, {z_distortion:.2f})"),
        html.P(f"Soft Iron Rotation (X,Y,Z): ({x_rot:.2f}°, {y_rot:.2f}°, {z_rot:.2f}°)"),
        html.Hr()
    ]
    
def clean_seed_string(seed_string):
    """ Entfernt alle nicht-numerischen Zeichen aus einem Seed-String. """
    return re.sub(r'[^0-9]', '', seed_string)

def parse_seed_string(seed_string):
    """
    Parst einen Seed-String (neues Format mit 9 Segmenten) und gibt strukturierte Daten zurück.
    Neues Format (9 Segmente):
    alpha_samples_gen_mode_noise_hi_si_dist_si_rot_date_time
    
    Rückgabe: (success, seed_dict, error_message)
    """
    parts = seed_string.strip().split('_')
    
    # Neues Format hat 9 Segmente (statt 8)
    if len(parts) != 9:
        return False, None, f"Fehler: Ungültiges Format (Erwarte 9 Segmente, erhalten {len(parts)})"

    try:
        # 1. Winkeleinschränkung
        alpha = int(parts[0])
        if not (0 <= alpha <= 90):
            return False, None, f"Fehler: Winkel {alpha}° ungültig."
        
        # 2. Anzahl Punkte
        samples = int(parts[1])
        if samples not in VALID_SAMPLE_COUNTS:
            return False, None, f"Fehler: Punktzahl {samples} unzulässig."

        # 3. Generierungsmodus
        gen_mode_int = int(parts[2])
        if gen_mode_int not in [0, 1, 2]:
            return False, None, "Fehler: Ungültiger Modus."
        
        mode_map_rev = {0: 'optimal', 1: 'random', 2: 'path'}
        gen_mode = mode_map_rev[gen_mode_int]

        # 4. Rauschen
        noise_str = parts[3]
        if not re.match(r'^\d{3}-\d$', noise_str):
            return False, None, "Fehler: Rauschen Format ungültig."
        noise_value = decode_custom_float(noise_str)

        # 5. Hard Iron (1-9 = random, 00 = manual)
        hi_val_str = parts[4]
        hi_is_random = False
        hi_random_range = 0
        if hi_val_str == "00":
            hi_is_random = False
        else:
            try:
                hi_random_range = int(hi_val_str)
                if 1 <= hi_random_range <= 9:
                    hi_is_random = True
                else:
                    return False, None, f"Fehler: Hard Iron Wert {hi_val_str} ungültig."
            except:
                return False, None, "Fehler: Hard Iron muss numerisch sein."

        # 6. Soft Iron Distortion (00-99 Format)
        si_dist_str = parts[5]
        if not re.match(r'^\d{2}$', si_dist_str):
            return False, None, "Fehler: Soft Iron Distortion Format ungültig."
        si_dist_val = decode_magnetic_error(si_dist_str)
        si_dist_is_random = (si_dist_val > 0)

        # 7. Soft Iron Rotation (000-360 Format)
        si_rot_str = parts[6]
        if not re.match(r'^\d{3}$', si_rot_str):
            return False, None, "Fehler: Soft Iron Rotation Format ungültig."
        si_rot_val = decode_magnetic_rotation(si_rot_str)
        si_rot_is_random = (si_rot_val > 0)

        # 8. Datum
        if not re.match(r'^\d{2}-\d{2}-\d{2}$', parts[7]):
            return False, None, "Fehler: Datum Format ungültig (yy-mm-dd)."
        
        # 9. Uhrzeit
        if not re.match(r'^\d{2}-\d{2}-\d{2}$', parts[8]):
            return False, None, "Fehler: Zeit Format ungültig (hh-mm-ss)."
        
        hh, mm, ss = map(int, parts[8].split('-'))

        seed_dict = {
            'Punktegenerierung': gen_mode,
            'Fehlerabweichung_Start': noise_value,
            'Kalibrierdauer/anzahlpunkte': samples,
            'Winkeleinschränkung': alpha,
            'Hard_Iron': [0.0, 0.0, 0.0],  # Wird ggfs. aus UI oder Datei geladen
            'Soft_Iron': [1.0, 1.0, 1.0],
            'Soft_Iron_Rotation': [0.0, 0.0, 0.0],
            'Uhrzeit': [hh, mm, ss],
            'hi_is_random': hi_is_random,
            'hi_random_range': hi_random_range,
            'si_dist_is_random': si_dist_is_random,
            'si_dist_random_range': si_dist_val,
            'si_rot_is_random': si_rot_is_random,
            'si_rot_random_range': si_rot_val
        }
        
        return True, seed_dict, None

    except ValueError as e:
        return False, None, f"Fehler: Zahl erwartet, aber Text gefunden. ({str(e)})"

def load_seed_with_offsets(seed_string, hi_override=None, si_dist_override=None, si_rot_override=None):
    """
    Lädt einen Seed und versucht, Hard Iron + Soft Iron Metadaten aus der CSV-Datei zu laden.
    
    Args:
        seed_string: Der Seed-String
        hi_override: (optional) Hard Iron Offset als Liste [x, y, z] - überschreibt Seed-Werte wenn Seed auf "00" (manual) gesetzt ist
        si_dist_override: (optional) Soft Iron Distortion als Liste [x, y, z] - überschreibt Seed-Werte wenn Seed auf "00" (manual) gesetzt ist
        si_rot_override: (optional) Soft Iron Rotation als Liste [x, y, z] - überschreibt Seed-Werte wenn Seed auf "000" (manual) gesetzt ist
    
    Rückgabe: (success, seed_dict, error_message)
    """
    success, seed_dict, error = parse_seed_string(seed_string)
    
    if not success:
        return False, None, error
    
    # Wenn Hard Iron im Manual Mode (00), verwende Override-Werte falls vorhanden
    if not seed_dict['hi_is_random'] and hi_override:
        seed_dict['Hard_Iron'] = list(hi_override)
    
    # Wenn Soft Iron Distortion im Manual Mode (00), verwende Override-Werte falls vorhanden
    if not seed_dict['si_dist_is_random'] and si_dist_override:
        seed_dict['Soft_Iron'] = list(si_dist_override)
    
    # Wenn Soft Iron Rotation im Manual Mode (000), verwende Override-Werte falls vorhanden
    if not seed_dict['si_rot_is_random'] and si_rot_override:
        seed_dict['Soft_Iron_Rotation'] = list(si_rot_override)
    
    # Versuchen, Metadaten aus der Datei zu laden (optional, leise wenn nicht vorhanden)
    _, metadata, load_error = load_csv_data_by_seed(seed_string, suppress_error=True)
    if not load_error and metadata:
        # Hard Iron
        if not seed_dict['hi_is_random']:  # Nur wenn Manual Mode im Seed
            seed_dict['Hard_Iron'] = metadata.get('HI_OFFSET', hi_override or [0, 0, 0])
        
        # Soft Iron
        if not seed_dict['si_dist_is_random']:  # Nur wenn Manual Mode im Seed
            si_dist = metadata.get('SI_DISTORTION', si_dist_override or [1.0, 1.0, 1.0])
            seed_dict['Soft_Iron'] = list(si_dist)
        
        if not seed_dict['si_rot_is_random']:  # Nur wenn Manual Mode im Seed
            si_rot = metadata.get('SI_ROTATION', si_rot_override or [0.0, 0.0, 0.0])
            seed_dict['Soft_Iron_Rotation'] = list(si_rot)
    
    return True, seed_dict, None

def load_csv_data_by_seed(seed_string, suppress_error=False):
    """
    Liest Punktedaten und Metadaten (Hard Iron + Soft Iron) aus der CSV.
    Rückgabe: (numpy_array, metadaten_dict, error_message)
    metadaten_dict: {'HI_OFFSET': [x,y,z], 'SI_DISTORTION': [x,y,z], 'SI_ROTATION': [x,y,z]}
    
    Args:
        seed_string: Der Seed zum Laden
        suppress_error: Wenn True, wird Fehler "Datei nicht gefunden" nicht zurückgegeben (leise fehlschlag)
    """
    full_path = os.path.join(OUTPUT_DIR, f"{seed_string}.csv")
    
    if not os.path.exists(full_path):
        if suppress_error:
            return None, None, None  # Leise fehlschlagen wenn suppress=True
        return None, None, f"Fehler: Datei '{seed_string}.csv' nicht gefunden."
        
    try:
        # 1. Metadaten aus den ersten Zeilen lesen
        metadata = {
            'HI_OFFSET': [0.0, 0.0, 0.0],
            'SI_DISTORTION': [1.0, 1.0, 1.0],
            'SI_ROTATION': [0.0, 0.0, 0.0]
        }
        
        with open(full_path, 'r') as f:
            for i in range(5):  # Lese bis zu 5 Zeilen für Metadaten
                line = f.readline().strip()
                if not line.startswith("#"):
                    break
                    
                if line.startswith("# HI_OFFSET:"):
                    content = line.split(":")[1]
                    metadata['HI_OFFSET'] = [float(x) for x in content.split(";")]
                elif line.startswith("# SI_DISTORTION:"):
                    content = line.split(":")[1]
                    metadata['SI_DISTORTION'] = [float(x) for x in content.split(";")]
                elif line.startswith("# SI_ROTATION:"):
                    content = line.split(":")[1]
                    metadata['SI_ROTATION'] = [float(x) for x in content.split(";")]

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

VALID_SAMPLE_COUNTS = [200, 500, 1000, 2000, 4000, 8000]

app.layout = html.Div([
    dcc.Store(id='left-sidebar-state', data=True),
    dcc.Store(id='right-sidebar-state', data=False),
    dcc.Store(id='point-data-storage', data=None),
    dcc.Store(id='seed-data-storage', data=None),
    dcc.Store(id='loaded-seed-string-store', data=""),  # Speichert den ORIGINAL-Seed der geladen wurde
    dcc.Store(id='current-seed-store', data=""), 
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
                        options=[{'label': ' Gleichmässig ', 'value': 'optimal'}, {'label': 'Zufall', 'value': 'random'}, {'label': ' Pfad', 'value': 'path'}],
                        value='optimal', inline=True, style={'display': 'flex', 'gap': '10px', 'flexGrow': 1} 
                    ),
                ]),
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
            
            # 3. Kalibrierdauer / Anzahl Punkte
            html.Div(style=SECTION_STYLE, children=[
                html.Label("Kalibrierdauer / Anzahl Punkte:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='sample-duration-dropdown',
                    options=[{'label': f'{int(x/200)} min ({x} Punkte)', 'value': x} for x in VALID_SAMPLE_COUNTS], 
                    value=1000, clearable=False, style={'marginTop': '5px'}
                ),
            ]),
            
            # 4. Winkeleinschränkung 
            html.Div(style=SECTION_STYLE, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                    html.Label("Winkeleinschränkung in Grad:", style={'fontWeight': 'bold', 'flexShrink': 0}),
                    dcc.Input(id='angular-constraint-input', type='number', min=0, max=90, step=1, placeholder='90', style={'flexGrow': 1, 'minWidth': '0'}),
                ])
            ]),

            # 5. & 6. Hard/Soft Iron Fehler 
            html.Div(style=SECTION_STYLE, children=[
                html.Label("Magnetischer Fehler:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                dcc.Tabs(id="error-correction-tabs", value='hard-iron-tab', style={'height': '30px'}, children=[
                    dcc.Tab(label='Hard Iron (Versatz)', value='hard-iron-tab', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                        html.Div(style={'paddingTop': '10px'}, children=[
                            dcc.RadioItems(id='offset-mode', options=[{'label': ' Manuell ', 'value': 'manual'}, {'label': ' Zufällig ', 'value': 'random'}], value='manual', inline=True, style={'display': 'flex', 'gap': '10px', 'marginTop': '5px'}),
                            html.Div(id='offset-manual-settings', style={'marginTop': '10px'}, children=[
                                html.Label("Achsenabweichung:"),
                                 html.Div(style={'display': 'flex', 'gap': '5px'}, children=[
                                    html.Div(style={'flex': '1'}, children=[html.Div("x", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='x-offset-input', type='number', value=0, step=0.1, style={'width': '100%'})]),
                                    html.Div(style={'flex': '1'}, children=[html.Div("y", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='y-offset-input', type='number', value=0, step=0.1, style={'width': '100%'})]),
                                    html.Div(style={'flex': '1'}, children=[html.Div("z", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='z-offset-input', type='number', value=0, step=0.1, style={'width': '100%'})]),
                                ]),
                            ]),
                            html.Div(id='offset-random-settings', style={'marginTop': '10px', 'opacity': 0.4, 'pointerEvents': 'none'}, children=[
                                html.Label("Zufallsbereich:"),
                                html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '5px'}, children=[
                                    dcc.Input(id='offset-rand', type='number', max = '9', placeholder='Maximale Abweichung (zwischen 1 und 9)', style={'flexGrow': 1}),
                                ]),
                            ]),
                        ]),
                    ]),
                    dcc.Tab(label='Soft Iron (Verzerrung)', value='soft-iron-tab', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                        html.Div(style={'paddingTop': '10px'}, children=[
                            dcc.RadioItems(id='distortion-mode', options=[{'label': ' Manuell ', 'value': 'manual'}, {'label': ' Zufällig ', 'value': 'random'}], value='manual', inline=True, style={'display': 'flex', 'gap': '10px', 'marginTop': '5px'}),
                            html.Div(id='distortion-manual-settings', style={'marginTop': '10px'}, children=[
                                html.Label("Verzerrungsfaktor:"),
                                 html.Div(style={'display': 'flex', 'gap': '5px'}, children=[
                                    html.Div(style={'flex': '1'}, children=[html.Div("x", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='x-distortion-input', type='number', value=1, step=0.01, style={'width': '100%'})]),
                                    html.Div(style={'flex': '1'}, children=[html.Div("y", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='y-distortion-input', type='number', value=1, step=0.01, style={'width': '100%'})]),
                                    html.Div(style={'flex': '1', 'marginBottom': '10px'}, children=[html.Div("z", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='z-distortion-input', type='number', value=1, step=0.01, style={'width': '100%'})]),
                                ]),
                                html.Label("Rotation: (x, y, z Reihenfolge)"),
                                 html.Div(style={'display': 'flex', 'gap': '5px'}, children=[
                                    html.Div(style={'flex': '1'}, children=[html.Div("x", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='x-rotation', placeholder='0 - 360', type='number', step=0.01, style={'width': '100%'})]),
                                    html.Div(style={'flex': '1'}, children=[html.Div("y", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='y-rotation', placeholder='0 - 360', type='number', step=0.01, style={'width': '100%'})]),
                                    html.Div(style={'flex': '1'}, children=[html.Div("z", style={'textAlign': 'center', 'fontWeight': 'bold'}), dcc.Input(id='z-rotation', placeholder='0 - 360', type='number', step=0.01, style={'width': '100%'})]),
                                ]),
                            ]),
                            html.Div(id='distortion-random-settings', style={'marginTop': '10px', 'opacity': 0.4, 'pointerEvents': 'none'}, children=[
                                html.Label("Zufallsbereich:"),
                                html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '5px'}, children=[
                                    dcc.Input(id='distortion-rand', type='number', placeholder='Max Verzerrung (Std: 2)', style={'flexGrow': 1}),
                                ]),
                                html.Div(style={'display': 'flex', 'gap': '5px', 'marginTop': '5px'}, children=[
                                    dcc.Input(id='distortion-rand-rotation', type='number', placeholder='Max Rotation in ° (Std: 10)', style={'flexGrow': 1}, min=0, max=360),
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
                html.P("Seed: (W_P_V_R:000-0_H_S_D_T)", id='display-seed', style={'fontSize': '0.8em', 'color': '#555', 'marginBottom': '10px'}),
                
                html.Button('Punkte Erzeugen', id='submit-button', n_clicks=0, style={'width': '100%', 'marginBottom': '15px'}),
                html.Button('Datenset exportieren', id='export-dataset', n_clicks=0, style={'width': '100%', 'marginBottom': '15px'}),
                html.Div(id='export-status', children='', style={'color': 'green', 'textAlign': 'center', 'marginTop': '5px'}),

                # Mesh Opacity Slider
                html.Div(style=SECTION_STYLE, children=[
                    html.Label("Kugeltransparenz:", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='mesh-opacity-slider',
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        value=0.5,
                        marks={i/10: str(round(i/10, 1)) for i in range(0, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ]),
                html.Hr(),


                # Seed Eingeben / Laden
                html.Div(style={'marginTop': '15px'}, children=[
                    html.Label("Seed Laden:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center'}, children=[
                        dcc.Dropdown(
                            id='import-seed', 
                            options=get_available_seeds(), 
                            value=None, 
                            clearable=True, 
                            placeholder="Seed auswählen", 
                            style={'flexGrow': 1}
                        ),
                        html.Button('Laden', id='load-dataset-button', n_clicks=0, style=BUTTON_STYLE_INLINE)
                    ])
                ]),
                
                html.Div(style={'marginTop': '15px'}, children=[ 
                    html.Label("Seed Eingeben:", style={'display': 'block', 'marginBottom': '5px'}), 
                    html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center'}, children=[ 
                        dcc.Input(id='seed-input', type='text', placeholder='Seed eingeben', style={'flexGrow': 1}),
                        html.Button('Bestätigen', id='load-seed-button', n_clicks=0, style=BUTTON_STYLE_INLINE)
                    ]),
                ]),
                html.Div(id='seed-load-status', children='', style={'fontSize': '0.8em', 'textAlign': 'center', 'marginTop': '5px'}),
                
                # --- Optionale Felder für Manual HI/SI Override (nur wenn Seed auf "Manual" gesetzt ist) ---
                html.Div(style={'marginTop': '15px', 'paddingTop': '10px', 'borderTop': '1px dashed #ccc'}, children=[
                    html.Label("Optional - Manual Fehler (nur wenn Seed im Manual Mode):", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '10px', 'fontSize': '0.85em'}),
                    
                    # Hard Iron Override
                    html.Label("Hard Iron Versatz (optional):", style={'display': 'block', 'marginBottom': '5px', 'fontSize': '0.9em'}),
                    html.Div(style={'display': 'flex', 'gap': '5px', 'marginBottom': '10px'}, children=[
                        html.Div(style={'flex': '1'}, children=[html.Div("x", style={'textAlign': 'center', 'fontSize': '0.85em'}), dcc.Input(id='seed-hi-x', type='number', placeholder='0', step=0.1, style={'width': '100%', 'fontSize': '0.9em'})]),
                        html.Div(style={'flex': '1'}, children=[html.Div("y", style={'textAlign': 'center', 'fontSize': '0.85em'}), dcc.Input(id='seed-hi-y', type='number', placeholder='0', step=0.1, style={'width': '100%', 'fontSize': '0.9em'})]),
                        html.Div(style={'flex': '1'}, children=[html.Div("z", style={'textAlign': 'center', 'fontSize': '0.85em'}), dcc.Input(id='seed-hi-z', type='number', placeholder='0', step=0.1, style={'width': '100%', 'fontSize': '0.9em'})]),
                    ]),
                    
                    # Soft Iron Distortion Override
                    html.Label("Soft Iron Verzerrung (optional):", style={'display': 'block', 'marginBottom': '5px', 'fontSize': '0.9em'}),
                    html.Div(style={'display': 'flex', 'gap': '5px', 'marginBottom': '10px'}, children=[
                        html.Div(style={'flex': '1'}, children=[html.Div("x", style={'textAlign': 'center', 'fontSize': '0.85em'}), dcc.Input(id='seed-si-dist-x', type='number', placeholder='1', step=0.01, style={'width': '100%', 'fontSize': '0.9em'})]),
                        html.Div(style={'flex': '1'}, children=[html.Div("y", style={'textAlign': 'center', 'fontSize': '0.85em'}), dcc.Input(id='seed-si-dist-y', type='number', placeholder='1', step=0.01, style={'width': '100%', 'fontSize': '0.9em'})]),
                        html.Div(style={'flex': '1'}, children=[html.Div("z", style={'textAlign': 'center', 'fontSize': '0.85em'}), dcc.Input(id='seed-si-dist-z', type='number', placeholder='1', step=0.01, style={'width': '100%', 'fontSize': '0.9em'})]),
                    ]),
                    
                    # Soft Iron Rotation Override
                    html.Label("Soft Iron Drehung (optional):", style={'display': 'block', 'marginBottom': '5px', 'fontSize': '0.9em'}),
                    html.Div(style={'display': 'flex', 'gap': '5px'}, children=[
                        html.Div(style={'flex': '1'}, children=[html.Div("x", style={'textAlign': 'center', 'fontSize': '0.85em'}), dcc.Input(id='seed-si-rot-x', type='number', placeholder='0', step=0.1, style={'width': '100%', 'fontSize': '0.9em'})]),
                        html.Div(style={'flex': '1'}, children=[html.Div("y", style={'textAlign': 'center', 'fontSize': '0.85em'}), dcc.Input(id='seed-si-rot-y', type='number', placeholder='0', step=0.1, style={'width': '100%', 'fontSize': '0.9em'})]),
                        html.Div(style={'flex': '1'}, children=[html.Div("z", style={'textAlign': 'center', 'fontSize': '0.85em'}), dcc.Input(id='seed-si-rot-z', type='number', placeholder='0', step=0.1, style={'width': '100%', 'fontSize': '0.9em'})]),
                    ]),
                ]),
                
                html.Div(id='consistency-status-display', children='', style={'fontSize': '0.9em', 'textAlign': 'center', 'marginTop': '10px'}),
                
            ], style={'marginTop': '10px'}),
        ]),

        # --- MITTLERER BEREICH (PLOT) ---
        html.Div([
            html.Div("❮", id="btn-toggle-left", n_clicks=0, style={**BUTTON_STYLE, "left": "0", "borderRadius": "0 8px 8px 0"}),
            html.Div("❯", id="btn-toggle-right", n_clicks=0, style={**BUTTON_STYLE, "right": "0", "borderRadius": "8px 0 0 8px"}),
            dcc.Graph(id='sphere-plot', style={'width': '100%', 'height': '100vh'}, config={'responsive': True}) 
        ], style={'flex': '1', 'position': 'relative', 'height': '100vh', 'overflow': 'hidden'}),

        # --- RECHTES FENSTER (RESULTATE) ---
        html.Div(id='right-sidebar', style={**SIDEBAR_STYLE, "borderRight": "none", "borderLeft": "1px solid #ddd", "width": "250px"}, children=[
            html.H3("Resultate", style={'marginBottom': '20px', 'whiteSpace': 'nowrap'}),
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
    right_base = {**SIDEBAR_STYLE, "borderRight": "none", "borderLeft": "1px solid #ddd", "width": "250px"}
    right_style = right_base if right_is_open else COLLAPSED_STYLE.copy()
    return left_style, right_style, "❮" if left_is_open else "❯", "❯" if right_is_open else "❮", left_is_open, right_is_open

# B. Callback für die Steuerung der Fehler-Modi
@app.callback(
    [Output('offset-manual-settings', 'style'), Output('offset-random-settings', 'style'), Output('distortion-manual-settings', 'style'), Output('distortion-random-settings', 'style')],
    [Input('offset-mode', 'value'), Input('distortion-mode', 'value')]
)
def toggle_error_settings(offset_mode, distortion_mode):
    disabled = {'marginTop': '10px', 'opacity': 0.4, 'pointerEvents': 'none'}
    enabled = {'marginTop': '10px'}
    return (
        enabled if offset_mode == 'manual' else disabled,
        enabled if offset_mode == 'random' else disabled,
        enabled if distortion_mode == 'manual' else disabled,
        enabled if distortion_mode == 'random' else disabled
    )

# ============================================================
# 4. HELPER-FUNKTIONEN FÜR CALLBACKS
# ============================================================

def resolve_input_data(trigger_id, loaded_seed_data, ui_params):
    """
    Bestimmt die Eingabeparameter basierend auf der Quelle (Seed oder UI).
    Rückgabe: Alle resolvierten Parameter als Dictionary
    """
    if trigger_id == 'seed-data-storage' and loaded_seed_data is not None:
        # Aus geladenem Seed
        data = loaded_seed_data
        result = {
            'generation_mode': data['Punktegenerierung'],
            'noise_value': float(data.get('Fehlerabweichung_Start', DEFAULT_NOISE)),
            'sample_count': int(data['Kalibrierdauer/anzahlpunkte']),
            'alpha': float(data['Winkeleinschränkung']),
            'x_offset': float(data['Hard_Iron'][0]),
            'y_offset': float(data['Hard_Iron'][1]),
            'z_offset': float(data['Hard_Iron'][2]),
            'x_distortion': float(data['Soft_Iron'][0]),
            'y_distortion': float(data['Soft_Iron'][1]),
            'z_distortion': float(data['Soft_Iron'][2]),
            'from_seed': True
        }
    else:
        # Aus UI mit Default-Fallbacks
        result = {
            'generation_mode': ui_params['generation_mode'],
            'noise_value': ui_params['noise'] if ui_params['noise'] is not None else DEFAULT_NOISE,
            'sample_count': int(ui_params['sample_count'] if ui_params['sample_count'] is not None else DEFAULT_SAMPLES),
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

def check_data_consistency(seed_string, generated_points):
    """
    Überprüft die Konsistenz zwischen generierten Punkten und gespeicherten Daten.
    Rückgabe: (status_html, is_consistent)
    """
    loaded_data_array, _, load_error = load_csv_data_by_seed(seed_string)
    
    if load_error:
        return html.Span(load_error, style={'color': 'red', 'fontWeight': 'bold'}), False
    
    if loaded_data_array is None:
        return None, True
    
    if generated_points.shape != loaded_data_array.shape:
        msg = f"WARNUNG: Punktanzahl ({generated_points.shape[0]}) ≠ CSV ({loaded_data_array.shape[0]})"
        return html.Span(msg, style={'color': 'orange', 'fontWeight': 'bold'}), False
    
    if not np.allclose(generated_points, loaded_data_array, atol=1e-6):
        msg = "FEHLER: Daten-Integrität verletzt!"
        return html.Span(msg, style={'color': 'red', 'fontWeight': 'bold'}), False
    
    return html.Span("Daten-Konsistenz OK ✓", style={'color': 'green'}), True


@app.callback(
    [Output('seed-load-status', 'children'),
     Output('seed-load-status', 'style'),
     Output('seed-data-storage', 'data'),
     Output('loaded-seed-string-store', 'data')],
    [Input('load-seed-button', 'n_clicks'),      
     Input('load-dataset-button', 'n_clicks')],  
    [State('seed-input', 'value'),
     State('import-seed', 'value'),
     State('seed-hi-x', 'value'),
     State('seed-hi-y', 'value'),
     State('seed-hi-z', 'value'),
     State('seed-si-dist-x', 'value'),
     State('seed-si-dist-y', 'value'),
     State('seed-si-dist-z', 'value'),
     State('seed-si-rot-x', 'value'),
     State('seed-si-rot-y', 'value'),
     State('seed-si-rot-z', 'value')]              
)
def validate_and_load_seed(btn_manual, btn_dropdown, text_seed, dropdown_seed, 
                          hi_x, hi_y, hi_z, si_dist_x, si_dist_y, si_dist_z,
                          si_rot_x, si_rot_y, si_rot_z):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    seed_string = ""
    
    if trigger_id == 'load-dataset-button':
        if not dropdown_seed or dropdown_seed == 'none':
            return "Bitte einen Datensatz aus der Liste auswählen.", {'color': 'orange', 'textAlign': 'center', 'marginTop': '5px'}, no_update, no_update
        seed_string = dropdown_seed
    elif trigger_id == 'load-seed-button':
        if not text_seed:
            return "Bitte Seed eingeben.", {'color': 'orange', 'textAlign': 'center', 'marginTop': '5px'}, no_update, no_update
        seed_string = text_seed

    # Prepare override values (nur wenn vom Benutzer eingegeben)
    hi_override = None
    si_dist_override = None
    si_rot_override = None
    
    if hi_x is not None or hi_y is not None or hi_z is not None:
        hi_override = [hi_x or 0, hi_y or 0, hi_z or 0]
    
    if si_dist_x is not None or si_dist_y is not None or si_dist_z is not None:
        si_dist_override = [si_dist_x or 1.0, si_dist_y or 1.0, si_dist_z or 1.0]
    
    if si_rot_x is not None or si_rot_y is not None or si_rot_z is not None:
        si_rot_override = [si_rot_x or 0, si_rot_y or 0, si_rot_z or 0]

    success, seed_data, error = load_seed_with_offsets(seed_string, hi_override, si_dist_override, si_rot_override)
    
    if not success:
        return error, {'color': 'red', 'textAlign': 'center', 'marginTop': '5px'}, no_update, no_update
        
    return f"Seed '{seed_string}' geladen!", {'color': 'green', 'textAlign': 'center', 'marginTop': '5px'}, seed_data, seed_string

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
    """
    if seed_data is None:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
    # Hard Iron Offset aus Seed (oder Default)
    hi = seed_data.get('Hard_Iron', [0, 0, 0])
    x_offset = hi[0] if len(hi) > 0 else 0
    y_offset = hi[1] if len(hi) > 1 else 0
    z_offset = hi[2] if len(hi) > 2 else 0
    
    # Soft Iron Distortion aus Seed (oder Default)
    si_dist = seed_data.get('Soft_Iron', [1.0, 1.0, 1.0])
    x_distortion = si_dist[0] if len(si_dist) > 0 else 1.0
    y_distortion = si_dist[1] if len(si_dist) > 1 else 1.0
    z_distortion = si_dist[2] if len(si_dist) > 2 else 1.0
    
    # Soft Iron Rotation aus Seed (oder Default)
    si_rot = seed_data.get('Soft_Iron_Rotation', [0.0, 0.0, 0.0])
    x_rotation = si_rot[0] if len(si_rot) > 0 else 0.0
    y_rotation = si_rot[1] if len(si_rot) > 1 else 0.0
    z_rotation = si_rot[2] if len(si_rot) > 2 else 0.0
    
    # Weitere Seed-Parameter
    generation_mode = seed_data.get('Punktegenerierung', 'optimal')
    alpha = seed_data.get('Winkeleinschränkung', DEFAULT_ALPHA)
    sample_count = seed_data.get('Kalibrierdauer/anzahlpunkte', DEFAULT_SAMPLES)
    noise = seed_data.get('Fehlerabweichung_Start', DEFAULT_NOISE)
    
    return x_offset, y_offset, z_offset, x_distortion, y_distortion, z_distortion, x_rotation, y_rotation, z_rotation, generation_mode, alpha, sample_count, noise

# D. Callback für den Graphen (Berechnung)
@app.callback(
    [Output('sphere-plot', 'figure'),
     Output('results-container', 'children'),
     Output('point-data-storage', 'data'),
     Output('display-time', 'children'),
     Output('display-seed', 'children'),
     Output('seed-data-storage', 'data', allow_duplicate=True),
     Output('current-seed-store', 'data'),
     Output('consistency-status-display', 'children'),
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
     State('offset-rand', 'value'),
     State('offset-mode', 'value'),
     State('distortion-mode', 'value'),
     State('distortion-rand', 'value'),
     State('distortion-rand-rotation', 'value'),
     State('mesh-opacity-slider', 'value'),
     State('loaded-seed-string-store', 'data')], 
    prevent_initial_call=True
)
def update_graph(n_clicks, loaded_seed_data, ui_sample_count, ui_angular_constraint, ui_x_offset, ui_y_offset, 
                 ui_z_offset, ui_x_distortion, ui_y_distortion, ui_z_distortion, x_rotation, y_rotation, 
                 z_rotation, ui_generation_mode, ui_noise, offset_rand, ui_offset_mode, distortion_mode, 
                 distortion_rand, distortion_rand_rotation, ui_mesh_opacity, loaded_seed_string):
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
    
    # KRITISCH: Unterscheide zwischen zwei Fällen:
    # 1. Seed wurde von außen geladen (trigger_id == 'seed-data-storage') -> LADE PUNKTE AUS CSV!
    # 2. Benutzer klickt "Punkte Erzeugen" (trigger_id == 'submit-button') -> generiere NEUEN SEED
    
    if trigger_id == 'seed-data-storage' and loaded_seed_string:
        # Seed wurde von außen geladen -> VERWENDE GESPEICHERTE PUNKTE AUS CSV!
        seed_string = loaded_seed_string
        display_time_str = f"Uhrzeit (aus Seed): ({current_time_str})"
        
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
                xyz0=[params['x_offset'], params['y_offset'], params['z_offset']],
                xyz1=[si_x_scale, si_y_scale, si_z_scale],
                x_rot=si_x_rot,
                y_rot=si_y_rot,
                z_rot=si_z_rot
            )
        else:
            # Nutze die Metadaten aus der CSV-Datei
            actual_hi_offset = csv_metadata.get('HI_OFFSET', [params['x_offset'], params['y_offset'], params['z_offset']])
            si_x_scale = csv_metadata.get('SI_DISTORTION', [params['x_distortion'], params['y_distortion'], params['z_distortion']])[0]
            si_y_scale = csv_metadata.get('SI_DISTORTION', [params['x_distortion'], params['y_distortion'], params['z_distortion']])[1]
            si_z_scale = csv_metadata.get('SI_DISTORTION', [params['x_distortion'], params['y_distortion'], params['z_distortion']])[2]
            si_x_rot = csv_metadata.get('SI_ROTATION', [x_rotation if x_rotation else 0, y_rotation if y_rotation else 0, z_rotation if z_rotation else 0])[0]
            si_y_rot = csv_metadata.get('SI_ROTATION', [x_rotation if x_rotation else 0, y_rotation if y_rotation else 0, z_rotation if z_rotation else 0])[1]
            si_z_rot = csv_metadata.get('SI_ROTATION', [x_rotation if x_rotation else 0, y_rotation if y_rotation else 0, z_rotation if z_rotation else 0])[2]
    else:
        # Benutzer hat "Punkte Erzeugen" geklickt -> GENERIERE NEUEN SEED
        seed_string = generate_seed_string(
            params['alpha'], params['sample_count'], params['generation_mode'], 
            params['noise_value'], ui_offset_mode, offset_rand,
            distortion_mode, distortion_rand, distortion_rand_rotation,
            current_time=current_time
        )
        display_time_str = f"Uhrzeit: ({current_time_str})"
        
        # --- SOFT IRON RANDOM LOGIC ---
        # Wenn Soft Iron im Zufalls-Modus ist, generiere zufällige Parameter
        si_x_scale = params['x_distortion']
        si_y_scale = params['y_distortion']
        si_z_scale = params['z_distortion']
        si_x_rot = x_rotation if x_rotation else 0
        si_y_rot = y_rotation if y_rotation else 0
        si_z_rot = z_rotation if z_rotation else 0
        
        if distortion_mode == 'random':
            # Verwende Defaultwerte wenn nicht eingegeben
            max_dist = float(distortion_rand) if distortion_rand else 2.0
            max_rot = float(distortion_rand_rotation) if distortion_rand_rotation else 10.0
            
            # Generiere zufällige SI-Parameter
            si_x_scale, si_y_scale, si_z_scale, si_x_rot, si_y_rot, si_z_rot = \
                generate_random_soft_iron_params(seed_string, max_dist, max_rot)
        
        # Generate points mit Soft Iron Rotation
        data_points, actual_hi_offset = fibonacci_sphere(
            params['generation_mode'], params['noise_value'], params['sample_count'], 
            params['alpha'], seed_string, 
            xyz0=[params['x_offset'], params['y_offset'], params['z_offset']],
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
    
    # Build figure mit transformierten Mesh-Vertices
    fig = build_figure_with_points(
        data_points, center_offset, transformed_vertices, sphere_triangles, 
        params['alpha'], params['noise_value'], si_x_scale, 
        si_y_scale, si_z_scale, ui_mesh_opacity
    )
    
    # Build results display
    result_text = build_results_display(
        data_points, params['alpha'], params['generation_mode'], 
        params['noise_value'], center_offset, si_x_scale, 
        si_y_scale, si_z_scale, si_x_rot, si_y_rot, si_z_rot
    )
    
    # Check consistency - DEAKTIVIERT (Fehlermeldung soll nicht angezeigt werden)
    consistency_status = None
    
    # Export data
    export_points_xyz = data_points[:, [0, 1, 2]]
    points_data = pd.DataFrame(export_points_xyz, columns=['X', 'Y', 'Z']).to_json(orient='split')
    seed_display = f"Seed: {seed_string}"
    
    # Speichere die tatsächlich verwendeten magnetischen Fehlerwerte
    magnetic_errors = {
        'HI': [center_offset[0], center_offset[1], center_offset[2]],
        'SI_DISTORTION': [si_x_scale, si_y_scale, si_z_scale],
        'SI_ROTATION': [si_x_rot, si_y_rot, si_z_rot]
    }
    
    return fig, result_text, points_data, display_time_str, seed_display, None, seed_string, consistency_status, magnetic_errors

# E. Callback für den Export der Punktedaten
@app.callback(
    [Output("download-csv", "data"),
     Output('export-status', 'children'),
     Output('export-success-trigger', 'data')],
    [Input("export-dataset", "n_clicks")],
    [State('point-data-storage', 'data'),
     State('current-seed-store', 'data'),
     State('magnetic-error-values-store', 'data')]
)
def export_dataset(n_clicks, json_data, seed, magnetic_errors):
    if n_clicks > 0 and json_data is not None:
        df = pd.read_json(json_data, orient='split')
        filename = f"{seed}.csv"
        full_path = os.path.join(OUTPUT_DIR, filename)
        
        # Nutze die tatsächlich generierten Werte aus dem Store
        if magnetic_errors:
            hi = magnetic_errors.get('HI', [0, 0, 0])
            si_dist = magnetic_errors.get('SI_DISTORTION', [1.0, 1.0, 1.0])
            si_rot = magnetic_errors.get('SI_ROTATION', [0.0, 0.0, 0.0])
            
            ox, oy, oz = hi[0], hi[1], hi[2]
            x_distortion, y_distortion, z_distortion = si_dist[0], si_dist[1], si_dist[2]
            x_rotation, y_rotation, z_rotation = si_rot[0], si_rot[1], si_rot[2]
        else:
            # Fallback auf Defaults
            ox, oy, oz = 0, 0, 0
            x_distortion, y_distortion, z_distortion = 1.0, 1.0, 1.0
            x_rotation, y_rotation, z_rotation = 0.0, 0.0, 0.0

        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            with open(full_path, 'w', newline='') as f:
                f.write(f"# HI_OFFSET:{ox};{oy};{oz}\n")
                f.write(f"# SI_DISTORTION:{x_distortion};{y_distortion};{z_distortion}\n")
                f.write(f"# SI_ROTATION:{x_rotation};{y_rotation};{z_rotation}\n")
                df.to_csv(f, index=False, sep=';', float_format='%.8f')
            
            return None, f"✓ Export erfolgreich: {filename}", n_clicks

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