import math
import numpy as np

# Zufallsgenerator ohne festen Seed - erzeugt echte Zufallszahlen
rng = np.random.default_rng()


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
    
    # Kombinierte Transformationsmatrix: R_z * R_y * R_x * S    
    # (Zuerst Skalierung, dann Rotation um x, dann y, dann z)
    rotation_matrix = R_z @ R_y @ R_x
    inverse_rotation_matrix = np.linalg.inv(R_z @ R_y @ R_x)

    return transform_matrix, inverse_rotation_matrix


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
    
    transform_matrix, inverse_matrix = create_soft_iron_matrix(x_scale, y_scale, z_scale,
                                              x_rot_deg, y_rot_deg, z_rot_deg)
    
    # Wende Transformation auf alle Punkte an (Nx3 @ 3x3)
    transformed_points = points @ inverse_matrix.T @ transform_matrix.T
    
    
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
    transform_matrix_3x3, _ = create_soft_iron_matrix(x_scale, y_scale, z_scale,
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
