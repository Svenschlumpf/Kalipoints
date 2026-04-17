import math
import numpy as np
import pyvista as pv

from utils.iron_math import apply_soft_iron_transformation, rng

# ============================================================
# KONSTANTEN
# ============================================================
MESH_OFFSET_SCALE = 1.02
MESH_GRID_RESOLUTION = 100  # Reduziert von 250 für bessere Performance
CACHED_MESHES = {}  # Cache für berechnete Meshes


def _normalize_axis_constraint_mode(axis_constraint_mode):
    """Normalisiert die Achseneinschraenkung auf bekannte interne Modi."""
    normalized = str(axis_constraint_mode or '').strip().lower()
    if normalized in ('pitch_only', 'nicken_ohne_rollen', 'ohne_rollen'):
        return 'pitch_only'
    return 'pitch_roll'


def fibonacci_sphere(generation_mode, noise_value, samples=1000, alpha=90, seed_string="", 
                     xyz0=[0, 0, 0], xyz1=[0, 0, 0], x_rot=0, y_rot=0, z_rot=0,
                     radius=1.0, maintain_density=False, axis_constraint_mode='pitch_roll'):
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
        radius: Kugelradius in Nanotesla (nT). Standard 1.0 = dimensionslose Einheitskugel.
        maintain_density: Wenn True, werden Punkte über die gesamte Kugel verteilt
                          und anschließend gefiltert (Punktedichte gleich wie ohne Einschränkung)
        axis_constraint_mode: 'pitch_roll' (Standard) oder 'pitch_only'
    
    Rückgabe: (points_array, used_offset_list, noise_array, ids_array)
    """
    # Verwende globalen RNG ohne Seed

    # Hard Iron Offset wird direkt aus den UI-Eingaben verwendet
    current_offset = list(xyz0) # Kopie der manuellen Offsets
    
    # --------------------------------------------------

    alpha_rad = math.pi * alpha / 180.0
    phi = math.pi * (2-(math.sqrt(5.) - 1.))
    z_limit = math.sin(alpha_rad)
    axis_mode = _normalize_axis_constraint_mode(axis_constraint_mode)

    # Wenn maintain_density aktiv: Punkte über gesamte Kugel verteilen, dann filtern
    if maintain_density and alpha < 90:
        gen_z_limit = 1.0
        gen_z_span = 2.0
    else:
        gen_z_limit = z_limit
        gen_z_span = z_limit * 2

    points = []
    noise_vectors = []
    ids = []
    sample_divisor = float(samples - 1) if samples > 1 else 1.0

    for i in range(samples):
        fraction = i / sample_divisor
        z = gen_z_limit - (fraction * gen_z_span)
        r_xy = math.sqrt(max(0.0, 1 - z * z))
        theta = phi * i
        if generation_mode == "random": 
            theta = rng.uniform(0, math.pi*2)
        x = math.cos(theta) * r_xy
        y = math.sin(theta) * r_xy

        noise_x = 0.0
        noise_y = 0.0
        noise_z = 0.0
        if noise_value != 0.0:
            noise_x = rng.uniform(-noise_value, noise_value)
            noise_y = rng.uniform(-noise_value, noise_value)
            noise_z = rng.uniform(-noise_value, noise_value)
            x += noise_x
            y += noise_y
            z += noise_z

        points.append((x, y, z))
        noise_vectors.append((noise_x, noise_y, noise_z))
        ids.append(i + 1)

    points_array = np.array(points)
    noise_array = np.array(noise_vectors)
    ids_array = np.array(ids, dtype=int)

    # Filter: bei maintain_density nur Punkte innerhalb der Winkeleinschränkung behalten
    if maintain_density and alpha < 90:
        if axis_mode == 'pitch_only':
            # Nur Nicken: Winkel im YZ-Schnitt (x spielt keine Rolle), y-Vorzeichen wird ignoriert.
            # Damit ist die Beschneidung an der xz-Ebene gespiegelt.
            yz_angles_deg = np.degrees(np.arctan2(np.abs(points_array[:, 2]), np.abs(points_array[:, 1])))
            mask = yz_angles_deg <= float(alpha)
        else:
            mask = np.abs(points_array[:, 2]) <= z_limit
        points_array = points_array[mask]
        noise_array = noise_array[mask]
        ids_array = ids_array[mask]

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

    # --- RADIUS-SKALIERUNG (magnetische Flussdichte in nT) ---
    # Skaliert die Kugelform. Wird VOR dem Hard Iron Offset angewendet.
    if radius != 1.0:
        points_array *= radius
        noise_array *= radius

    # --- HARD IRON OFFSET (in nT) ---
    # Wird nach der Skalierung addiert, sodass der Offset unabhängig vom Radius ist.
    points_array[:, 0] += current_offset[0]
    points_array[:, 1] += current_offset[1]
    points_array[:, 2] += current_offset[2]

    return points_array, current_offset, noise_array, ids_array


def create_sphere_mesh(xyz0=[0, 0, 0]):
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
