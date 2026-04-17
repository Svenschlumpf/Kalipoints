"""Aggregiert Export- und Kalibrierdaten je Unterordner in Results_<unterordner>.csv."""

import csv
import os
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')
EXPORTS_DIR = os.path.join(DATASETS_DIR, '1-kalipoints_exports')
CALIBRATION_BASE_DIR = os.path.join(DATASETS_DIR, '3-calibration_results')
RESULTS_DIR = os.path.join(DATASETS_DIR, '4-calibrated_exports_for_analysis')

HEADER = [
    'datasetname',
    'HI-x-True', 'HI-y-True', 'HI-z-True',
    'SI-x-Faktor-True', 'SI-y-Faktor-True', 'SI-z-Faktor-True',
    'SI-x-Rotation-True', 'SI-y-Rotation-True', 'SI-z-Rotation-True',
    'HI-x-Meassured', 'HI-y-Meassured', 'HI-z-Meassured',
    'SI-x-Faktor-Meassured', 'SI-y-Faktor-Meassured', 'SI-z-Faktor-Meassured',
    'SI-x-Rotation-Meassured', 'SI-y-Rotation-Meassured', 'SI-z-Rotation-Meassured',
    'RSME-xyz',
    'MAE-xyz',
    'Mean-xyz',
    'RSME-xy',
    'MAE-xy',
    'Mean-xy',
    'Azimut-Max',
    'Azimut-Mean',
    'Azimut-MAE',
    'Polar-Max',
    'Polar-Mean',
    'Polar-MAE',
    'Metadata',
]


def parse_metadata_from_csv(filepath):
    """Liest die Metadaten-Kommentarzeilen aus einer kalipoints-Export-CSV."""
    metadata = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                break
            content = line[2:] if line.startswith('# ') else line[1:]
            if ':' in content:
                key, _, value = content.partition(':')
                key = key.strip()
                value = value.strip()
                if value.startswith('['):
                    continue
                metadata[key] = value
    return metadata


def infer_export_timestamp(dataset_name, meta):
    """Liest Export-Datum/-Zeit aus Metadaten oder leitet sie aus dem Dateinamen ab."""
    export_date = meta.get('EXPORT_DATE', '')
    export_time = meta.get('EXPORT_TIME', '')
    if export_date and export_time:
        return export_date, export_time

    match = re.search(r'(\d{2}-\d{2}-\d{2})_(\d{2}-\d{2})$', dataset_name)
    if not match:
        return export_date, export_time

    inferred_date, inferred_time = match.groups()
    if not export_date:
        export_date = inferred_date
    if not export_time:
        export_time = inferred_time.replace('-', ':')
    return export_date, export_time


def parse_float_triplet(value, default):
    raw_value = value if value not in (None, '') else default
    try:
        return [float(part.strip()) for part in raw_value.split(',')]
    except (AttributeError, ValueError):
        return [float(part.strip()) for part in default.split(',')]


def build_metadata_string(meta):
    """Baut den Metadata-Zusammenfassungsstring für die letzte Spalte."""
    distribution = meta.get('DISTRIBUTION_STYLE', '')
    export_date = meta.get('EXPORT_DATE', '')
    export_time = meta.get('EXPORT_TIME', '')
    true_field = meta.get('TRUE MAGNETIC FIELD STRENGTH', '')
    noise = meta.get('NOISE', '')
    point_amount = meta.get('POINT_AMOUNT', '')
    angular = meta.get('ANGULAR_CONSTRAINT-DEG', '')
    keep_density = meta.get('KEEP POINT DENSITY', '')
    axis_constraint = meta.get('AXIS_CONSTRAINT') or meta.get('AXSIS_CONSTRAINT', '')
    field_line = meta.get('FIELD_LINE_ANGLE-DEG', '0').split('(')[0].strip()

    hi_vals = parse_float_triplet(meta.get('HI-X-Y-Z-OFFSET'), '0,0,0')
    hi_active = any(abs(v) > 0.0 for v in hi_vals)

    si_dist_vals = parse_float_triplet(meta.get('SI-X-Y-Z-DISTORTION'), '1,1,1')
    si_dist_active = any(abs(v - 1.0) > 0.0 for v in si_dist_vals)

    si_rot_vals = parse_float_triplet(meta.get('SI-X-Y-Z-ROTATION-DEG'), '0,0,0')
    si_rot_active = any(abs(v) > 0.0 for v in si_rot_vals)

    return (
        f"Metadata("
        f"DISTRIBUTION_STYLE: {distribution}, "
        f"EXPORT_DATE: {export_date}, "
        f"EXPORT_TIME: {export_time}, "
        f"TRUE MAGNETIC FIELD STRENGTH: {true_field}, "
        f"NOISE: {noise}, "
        f"POINT_AMOUNT: {point_amount}, "
        f"ANGULAR_CONSTRAINT-DEG: {angular}, "
        f"KEEP POINT DENSITY: {keep_density}, "
        f"AXIS_CONSTRAINT: {axis_constraint}, "
        f"FIELD_LINE_ANGLE-DEG: {field_line}, "
        f"HI-X-Y-Z-OFFSET: {hi_active}, "
        f"SI-X-Y-Z-DISTORTION: {si_dist_active}, "
        f"SI-X-Y-Z-ROTATION-DEG: {si_rot_active})"
    )


def list_export_subdirs():
    """Listet alle Unterordner unter 1-kalipoints_exports."""
    if not os.path.exists(EXPORTS_DIR):
        return []
    return sorted(
        [name for name in os.listdir(EXPORTS_DIR) if os.path.isdir(os.path.join(EXPORTS_DIR, name))]
    )


def results_filename_for_subdir(subdir_name):
    safe_name = str(subdir_name).strip().replace(' ', '_')
    return f'Results_{safe_name}.csv'


def list_results_files():
    """Listet alle Results_*.csv aus dem Analyse-Ausgabeordner."""
    if not os.path.exists(RESULTS_DIR):
        return []
    names = [n for n in os.listdir(RESULTS_DIR) if n.startswith('Results_') and n.endswith('.csv')]
    return sorted(names)


def _read_calibration_for_dataset(calibration_dir, dataset_name):
    """Liest Kalibrierdaten aus <calibration_dir>/<dataset_name>.csv (erste Zeile)."""
    file_path = os.path.join(calibration_dir, f'{dataset_name}.csv')
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            first_row = next(reader, None)
    except Exception:
        return {}

    if not first_row:
        return {}

    return {
        'HI-x': first_row.get('fit_center_x', ''),
        'HI-y': first_row.get('fit_center_y', ''),
        'HI-z': first_row.get('fit_center_z', ''),
        'SI-x-Faktor': first_row.get('fit_radius_x', ''),
        'SI-y-Faktor': first_row.get('fit_radius_y', ''),
        'SI-z-Faktor': first_row.get('fit_radius_z', ''),
        'SI-x-Rotation': first_row.get('fit_rotation_roll', ''),
        'SI-y-Rotation': first_row.get('fit_rotation_pitch', ''),
        'SI-z-Rotation': first_row.get('fit_rotation_yaw', ''),
    }


def build_results_csv(export_subdir):
    """Erstellt/überschreibt Results_<unterordner>.csv auf Basis der Export/Kalibrierdaten."""
    if not export_subdir:
        return {'ok': False, 'message': 'Bitte einen Daten-Unterordner auswählen.', 'rows': 0}

    if not os.path.exists(EXPORTS_DIR):
        return {'ok': False, 'message': f'Export-Verzeichnis nicht gefunden: {EXPORTS_DIR}', 'rows': 0}

    export_dir = os.path.join(EXPORTS_DIR, export_subdir)
    if not os.path.exists(export_dir):
        return {'ok': False, 'message': f'Export-Unterordner nicht gefunden: {export_dir}', 'rows': 0}

    calibration_dir = os.path.join(CALIBRATION_BASE_DIR, export_subdir)
    if not os.path.exists(calibration_dir):
        return {'ok': False, 'message': f'Kalibrier-Unterordner nicht gefunden: {calibration_dir}', 'rows': 0}

    export_files = sorted([file_name for file_name in os.listdir(export_dir) if file_name.endswith('.csv')])

    rows = []
    total_files = len(export_files)
    for idx, filename in enumerate(export_files, start=1):
        print(f"[Results fuellen] Datensatz {idx}/{total_files}: {filename}")
        filepath = os.path.join(export_dir, filename)
        meta = parse_metadata_from_csv(filepath)
        dataset_name = filename.replace('.csv', '')
        export_date, export_time = infer_export_timestamp(dataset_name, meta)
        meta['EXPORT_DATE'] = export_date
        meta['EXPORT_TIME'] = export_time

        hi_vals = [value.strip() for value in meta.get('HI-X-Y-Z-OFFSET', '0,0,0').split(',')]
        si_dist_vals = [value.strip() for value in meta.get('SI-X-Y-Z-DISTORTION', '1.0,1.0,1.0').split(',')]
        si_rot_vals = [value.strip() for value in meta.get('SI-X-Y-Z-ROTATION-DEG', '0,0,0').split(',')]

        cal = _read_calibration_for_dataset(calibration_dir, dataset_name)
        metadata_str = build_metadata_string(meta)

        row = [
            dataset_name,
            hi_vals[0], hi_vals[1], hi_vals[2],
            si_dist_vals[0], si_dist_vals[1], si_dist_vals[2],
            si_rot_vals[0], si_rot_vals[1], si_rot_vals[2],
            cal.get('HI-x', ''), cal.get('HI-y', ''), cal.get('HI-z', ''),
            cal.get('SI-x-Faktor', ''), cal.get('SI-y-Faktor', ''), cal.get('SI-z-Faktor', ''),
            cal.get('SI-x-Rotation', ''), cal.get('SI-y-Rotation', ''), cal.get('SI-z-Rotation', ''),
            '', '', '', '', '', '',
            '', '', '', '', '', '',
            metadata_str,
        ]
        rows.append(row)

    results_file = os.path.join(RESULTS_DIR, results_filename_for_subdir(export_subdir))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    return {
        'ok': True,
        'message': f'{os.path.basename(results_file)} erstellt mit {len(rows)} Datensätzen.',
        'rows': len(rows),
        'results_file': results_file,
    }


def main():
    subdirs = list_export_subdirs()
    selected = 'neu' if 'neu' in subdirs else (subdirs[0] if subdirs else None)
    result = build_results_csv(selected)
    print(result['message'])
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
