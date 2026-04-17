import os
import numpy as np
import pandas as pd

from components.styles import OUTPUT_DIR, DEFAULT_NOISE, DEFAULT_SAMPLES, DEFAULT_ALPHA, DEFAULT_OFFSET, DEFAULT_DISTORTION

CALIBRATION_DIR = os.path.join("datasets", "3-calibration_results", "calibration_summary")
BASE_CALIB_DIR = os.path.join("datasets", "3-calibration_results")
SIMULATED_BASE_DIR = os.path.join("datasets", "1-kalipoints_exports")
REALLIFE_DIR = os.path.join("datasets", "0-realLifeData", "kalipoints_real")
DEFAULT_DATASET_SUBDIR_BY_SOURCE = {
    'exports': 'neu',
    'reallife': 'Unbeschnitten',
}


def get_dataset_source_base_dir(dataset_source):
    """Gibt den Basisordner fuer die gewaehlte Datenquelle zurueck."""
    return REALLIFE_DIR if dataset_source == 'reallife' else SIMULATED_BASE_DIR


def get_dataset_subdir_options(dataset_source):
    """Listet alle Unterordner der gewaehlten Quelle auf (ohne Hauptordner)."""
    base_dir = get_dataset_source_base_dir(dataset_source)
    options = []

    if os.path.exists(base_dir):
        rel_dirs = []
        for root, dirs, _files in os.walk(base_dir):
            rel_root = os.path.relpath(root, base_dir)
            for dir_name in sorted(dirs):
                rel_path = os.path.normpath(os.path.join(rel_root, dir_name))
                if rel_path == '.':
                    rel_path = dir_name
                rel_dirs.append(rel_path.replace('\\', '/'))

        for rel_path in sorted(set(rel_dirs)):
            options.append({'label': rel_path, 'value': rel_path})

    return options


def get_default_dataset_subdir(dataset_source):
    """Liefert den Standard-Unterordner je Quelle (falls vorhanden)."""
    options = get_dataset_subdir_options(dataset_source)
    values = [opt.get('value') for opt in options if opt.get('value')]
    preferred = DEFAULT_DATASET_SUBDIR_BY_SOURCE.get(dataset_source)

    if preferred in values:
        return preferred
    if values:
        return values[0]
    return ''


def resolve_dataset_directory(dataset_source, dataset_subdir):
    """Loest den absoluten/relativen Ordnerpfad fuer Quelle + Unterordner auf."""
    base_dir = get_dataset_source_base_dir(dataset_source)
    if not dataset_subdir:
        return base_dir
    return os.path.join(base_dir, dataset_subdir)


def load_calibration_data(seed_string, calib_dir=None):
    """
    Liest Kalibrierdaten aus der Calibration Summary CSV.
    Returns: (calib_dict, error_message)
    calib_dict ist None wenn keine Kalibrierdaten vorhanden sind (kein Fehler).
    """
    dir_to_use = calib_dir if calib_dir is not None else CALIBRATION_DIR
    full_path = os.path.join(dir_to_use, f"{seed_string}.csv")
    if not os.path.exists(full_path):
        return None, None

    try:
        df = pd.read_csv(full_path, sep=',')
        df.columns = df.columns.str.strip()
        row = df.iloc[0]

        calib_data = {
            'fit_center_x': float(row['fit_center_x']),
            'fit_center_y': float(row['fit_center_y']),
            'fit_center_z': float(row['fit_center_z']),
            'fit_radius_x': float(row['fit_radius_x']),
            'fit_radius_y': float(row['fit_radius_y']),
            'fit_radius_z': float(row['fit_radius_z']),
            'fit_rotation_roll': float(row['fit_rotation_roll']),
            'fit_rotation_pitch': float(row['fit_rotation_pitch']),
            'fit_rotation_yaw': float(row['fit_rotation_yaw']),
            'matrix': [
                [float(row['h00']), float(row['h01']), float(row['h02']), float(row['h03'])],
                [float(row['h10']), float(row['h11']), float(row['h12']), float(row['h13'])],
                [float(row['h20']), float(row['h21']), float(row['h22']), float(row['h23'])],
                [float(row['h30']), float(row['h31']), float(row['h32']), float(row['h33'])],
            ]
        }
        return calib_data, None
    except Exception as e:
        return None, f"Fehler beim Laden der Kalibrierdaten: {e}"


def load_csv_data_by_seed(seed_string, suppress_error=False, directory=None):
    """
    Liest Punktedaten und erweiterte Metadaten aus der CSV.
    Unterstützt sowohl simulierte Daten (mit # Metadaten) als auch einfache X;Y;Z CSV-Dateien.
    Rückgabe: (numpy_array, metadaten_dict, error_message)
    """
    source_dir = directory if directory is not None else OUTPUT_DIR
    full_path = os.path.join(source_dir, f"{seed_string}.csv")
    
    if not os.path.exists(full_path):
        if suppress_error:
            return None, None, None  # Leise fehlschlagen wenn suppress=True
        return None, None, f"Fehler: Datei '{seed_string}.csv' nicht gefunden."
        
    try:
        # 1. Metadaten aus den ersten Zeilen lesen (Rückwärtskompatibilität + Neue Formate)
        # WICHTIG: Mit None initialisieren, nicht mit Defaults!
        # So können fehlende Felder erkannt und als "N/A" angezeigt werden
        metadata = {
            'UNIT': None,
            'DISTRIBUTION_STYLE': None,
            'EXPORT_DATE': None,
            'EXPORT_TIME': None,
            'TRUE_MAGNETIC_FIELD_STRENGTH': None,
            'NOISE': None,
            'POINT_AMOUNT': None,
            'ANGULAR_CONSTRAINT_DEG': None,
            'KEEP_POINT_DENSITY': None,
            'AXIS_CONSTRAINT': None,
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
                
                if line.startswith("# UNIT:"):
                    unit_value = line.split(":", 1)[1].strip()
                    metadata['UNIT'] = unit_value if unit_value else None
                # Neue Metadaten-Format
                elif line.startswith("# DISTRIBUTION_STYLE:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        metadata['DISTRIBUTION_STYLE'] = value
                elif line.startswith("# EXPORT_DATE:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        metadata['EXPORT_DATE'] = value
                elif line.startswith("# EXPORT_TIME:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        metadata['EXPORT_TIME'] = value
                elif line.startswith("# TRUE MAGNETIC FIELD STRENGTH:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        try:
                            metadata['TRUE_MAGNETIC_FIELD_STRENGTH'] = float(value)
                        except (ValueError, TypeError):
                            metadata['TRUE_MAGNETIC_FIELD_STRENGTH'] = None
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
                elif line.startswith("# KEEP POINT DENSITY:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        metadata['KEEP_POINT_DENSITY'] = value.lower() == 'true'
                elif line.startswith("# AXIS_CONSTRAINT:"):
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        metadata['AXIS_CONSTRAINT'] = value
                elif line.startswith("# AXSIS_CONSTRAINT:"):
                    # Rueckwaertskompatibilitaet fuer moeglichen Schreibfehler im Key.
                    value = line.split(":", 1)[1].strip()
                    if value and value not in ('N/A', ''):
                        metadata['AXIS_CONSTRAINT'] = value
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
        data = df[['X', 'Y', 'Z']].values

        # Fuer interne Berechnung arbeitet die App in nT.
        # Wenn keine verlässliche UNIT-Metadaten vorliegen, wird Tesla über den Wertebereich erkannt.
        unit_raw = str(metadata.get('UNIT') or '').strip().lower()
        needs_tesla_to_nt = unit_raw in ('tesla', 't')

        if not unit_raw:
            max_abs = np.nanmax(np.abs(data)) if data.size else 0.0
            # Reale Messdaten liegen typischerweise im Bereich ~1e-5 T.
            # nT-Daten liegen mehrere Groessenordnungen hoeher.
            if np.isfinite(max_abs) and max_abs < 1e-2:
                needs_tesla_to_nt = True

        if needs_tesla_to_nt:
            data = data * 1e9
            for key in ('HI_OFFSET', 'HI_X_Y_Z_OFFSET'):
                if metadata.get(key) is not None:
                    metadata[key] = [v * 1e9 for v in metadata[key]]

        metadata['UNIT'] = 'nT'

        return data, metadata, None 
    except Exception as e:
        return None, None, f"Fehler beim Lesen der CSV-Datei: {e}"


def get_available_seeds():
    """Liest alle CSV-Dateien im Output-Ordner und gibt sie als Dropdown-Optionen zurück."""
    return get_seeds_from_dir(OUTPUT_DIR)


def get_calibration_dirs():
    """Listet alle verfügbaren Kalibrierordner unter datasets/3-calibration_results/ auf."""
    options = []
    if os.path.exists(BASE_CALIB_DIR):
        for name in sorted(os.listdir(BASE_CALIB_DIR)):
            full_path = os.path.join(BASE_CALIB_DIR, name)
            if os.path.isdir(full_path):
                options.append({'label': name, 'value': full_path})
    if not options:
        return [{'label': 'Keine Kalibrierordner gefunden', 'value': '', 'disabled': True}]
    return options


def get_seeds_from_dir(directory):
    """Liest alle CSV-Dateien aus einem beliebigen Ordner und gibt sie als Dropdown-Optionen zurück."""
    options = []

    if os.path.exists(directory):
        files = os.listdir(directory)
        files.sort(reverse=True)

        for f in files:
            if f.endswith(".csv"):
                seed_name = f.replace(".csv", "")
                options.append({'label': seed_name, 'value': seed_name})

    if not options:
        return [{'label': 'Keine Datensätze gefunden', 'value': 'none', 'disabled': True}]

    return options


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
