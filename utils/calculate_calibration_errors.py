import os
import math

import numpy as np
import pandas as pd

from utils.csv_io import load_calibration_data, load_csv_data_by_seed

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'datasets', '4-calibrated_exports_for_analysis')
EXPORTS_DIR = os.path.join(ROOT_DIR, 'datasets', '1-kalipoints_exports')
CALIBRATION_BASE_DIR = os.path.join(ROOT_DIR, 'datasets', '3-calibration_results')
CORRECTION_DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets', '5-correction_datasets')
EPSILON = 1e-12


def _normalize_axis_constraint_mode(value):
    normalized = str(value or '').strip().lower()
    if normalized in ('pitch_only', 'nicken_ohne_rollen', 'ohne_rollen'):
        return 'pitch_only'
    return 'pitch_roll'


def _extract_match_key(seed_name, metadata):
    point_amount = metadata.get('POINT_AMOUNT') if isinstance(metadata, dict) else None

    if point_amount is None:
        match = str(seed_name or '').split('_')
        if len(match) >= 3:
            point_amount = match[2]

    try:
        point_amount = int(float(point_amount)) if point_amount is not None else None
    except Exception:
        point_amount = None

    return point_amount


def _minimal_angle_diff_deg(angle_a_deg, angle_b_deg):
    diff = (float(angle_a_deg) - float(angle_b_deg) + 180.0) % 360.0 - 180.0
    return abs(diff)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in ('1', 'true', 'yes', 'y'):
        return True
    if normalized in ('0', 'false', 'no', 'n', ''):
        return False
    return bool(value)


def _infer_ids_from_generation_metadata(metadata, actual_count):
    if not isinstance(metadata, dict):
        return None

    try:
        samples = int(float(metadata.get('POINT_AMOUNT')))
        alpha = float(metadata.get('ANGULAR_CONSTRAINT_DEG'))
    except Exception:
        return None

    axis_mode = _normalize_axis_constraint_mode(metadata.get('AXIS_CONSTRAINT'))
    maintain_density = _parse_bool(metadata.get('KEEP_POINT_DENSITY'))

    if samples <= 0:
        return None

    # Ohne Dichte-Erhalt oder ohne Einschränkung gibt es keine Beschneidung.
    if (not maintain_density) or alpha >= 90.0:
        ids = np.arange(1, samples + 1, dtype=int)
        return ids[:actual_count] if len(ids) >= actual_count else None

    alpha_rad = math.pi * alpha / 180.0
    z_limit = math.sin(alpha_rad)
    phi = math.pi * (2 - (math.sqrt(5.0) - 1.0))
    sample_divisor = float(samples - 1) if samples > 1 else 1.0

    kept_ids = []
    for i in range(samples):
        fraction = i / sample_divisor
        z = 1.0 - (fraction * 2.0)

        keep = False
        if axis_mode == 'pitch_only':
            r_xy = math.sqrt(max(0.0, 1.0 - z * z))
            theta = phi * i
            y = math.sin(theta) * r_xy
            yz_angle_deg = math.degrees(math.atan2(abs(z), abs(y)))
            keep = yz_angle_deg <= alpha
        else:
            keep = abs(z) <= z_limit

        if keep:
            kept_ids.append(i + 1)

    if len(kept_ids) != actual_count:
        return None
    return np.array(kept_ids, dtype=int)


def _load_points_noise_ids(seed_name, directory, metadata=None):
    file_path = os.path.join(directory, f"{seed_name}.csv")
    if not os.path.exists(file_path):
        return None, None, None, False
    try:
        df = pd.read_csv(file_path, sep=';', comment='#')
    except Exception:
        return None, None, None, False

    if not {'X', 'Y', 'Z'}.issubset(df.columns):
        return None, None, None, False

    points = df[['X', 'Y', 'Z']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    id_in_file = 'ID' in df.columns
    if points.size == 0:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0,), dtype=int),
            id_in_file,
        )

    max_abs = np.nanmax(np.abs(points)) if np.isfinite(points).any() else 0.0
    if np.isfinite(max_abs) and max_abs < 1e-2:
        points = points * 1e9

    if {'X_noise', 'Y_noise', 'Z_noise'}.issubset(df.columns):
        noise = df[['X_noise', 'Y_noise', 'Z_noise']].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
        noise_max = np.nanmax(np.abs(noise)) if np.isfinite(noise).any() else 0.0
        if np.isfinite(noise_max) and noise_max < 1e-2:
            noise = noise * 1e9
    else:
        noise = np.zeros_like(points)

    if id_in_file:
        ids = pd.to_numeric(df['ID'], errors='coerce').fillna(-1).astype(int).to_numpy()
    else:
        inferred_ids = _infer_ids_from_generation_metadata(metadata, len(points))
        ids = inferred_ids if inferred_ids is not None else np.arange(1, len(points) + 1, dtype=int)

    return points, noise, ids, id_in_file


def _compute_azimuth_zenith_errors_deg_by_id(calibrated_points_nt, noise_nt, test_ids, reference_points_nt, reference_ids):
    if len(calibrated_points_nt) == 0 or len(reference_points_nt) == 0:
        return None, None, 0, 0, 0

    ref_map = {}
    for idx, point_id in enumerate(reference_ids):
        if int(point_id) > 0:
            ref_map[int(point_id)] = reference_points_nt[idx]

    azimuth_errors = []
    zenith_errors = []
    matched = 0
    missing_in_ref = 0
    invalid_test_ids = 0

    for idx, point_id in enumerate(test_ids):
        point_id_int = int(point_id)
        if point_id_int <= 0:
            invalid_test_ids += 1
            continue
        ref_point = ref_map.get(point_id_int)
        if ref_point is None:
            missing_in_ref += 1
            continue

        # Kalibrierte Punkte bleiben unverändert; das Rauschen wird auf die
        # idealen Referenzpunkte (ID-basiert) addiert.
        test_point = calibrated_points_nt[idx]
        if noise_nt is not None and idx < len(noise_nt):
            ref_point = ref_point + noise_nt[idx]

        test_xy = test_point[:2]
        ref_xy = ref_point[:2]
        norm_test = float(np.linalg.norm(test_xy))
        norm_ref = float(np.linalg.norm(ref_xy))
        if norm_test <= EPSILON or norm_ref <= EPSILON:
            invalid_test_ids += 1
            continue

        dot_uv = float(ref_xy[0] * test_xy[0] + ref_xy[1] * test_xy[1])
        cos_value = dot_uv / (norm_ref * norm_test)
        cos_value = max(-1.0, min(1.0, cos_value))
        azimuth_errors.append(float(np.degrees(np.arccos(cos_value))))

        test_zenith = np.degrees(np.arctan2(np.linalg.norm(test_point[:2]), test_point[2]))
        ref_zenith = np.degrees(np.arctan2(np.linalg.norm(ref_point[:2]), ref_point[2]))
        zenith_errors.append(_minimal_angle_diff_deg(test_zenith, ref_zenith))
        matched += 1

    if matched == 0:
        return None, None, matched, missing_in_ref, invalid_test_ids

    return np.array(azimuth_errors, dtype=float), np.array(zenith_errors, dtype=float), matched, missing_in_ref, invalid_test_ids


def _build_reference_index():
    index = {}
    if not os.path.isdir(CORRECTION_DATASETS_DIR):
        return index

    duplicate_point_amount_refs = 0

    for file_name in sorted(os.listdir(CORRECTION_DATASETS_DIR)):
        if not file_name.lower().endswith('.csv'):
            continue
        seed_name = os.path.splitext(file_name)[0]
        points_nt, metadata, load_error = load_csv_data_by_seed(
            seed_name,
            suppress_error=True,
            directory=CORRECTION_DATASETS_DIR,
        )
        if load_error is not None or points_nt is None:
            continue
        key = _extract_match_key(seed_name, metadata or {})
        if key is None:
            continue
        ref_points, _ref_noise, ref_ids, _id_in_file = _load_points_noise_ids(
            seed_name,
            CORRECTION_DATASETS_DIR,
            metadata=metadata,
        )
        if ref_points is None or ref_ids is None:
            ref_points = points_nt
            inferred_ids = _infer_ids_from_generation_metadata(metadata or {}, len(points_nt))
            ref_ids = inferred_ids if inferred_ids is not None else np.arange(1, len(points_nt) + 1, dtype=int)

        if key in index:
            duplicate_point_amount_refs += 1
            continue

        index[key] = {
            'seed': seed_name,
            'points': ref_points,
            'ids': ref_ids,
        }

    if duplicate_point_amount_refs > 0:
        print(f"[Kalibrierfehler berechnen] Hinweis: {duplicate_point_amount_refs} zusätzliche Referenzdateien mit gleicher Punktezahl wurden ignoriert.")
    return index

def calculate_axis_error(true_value, measured_value, error_mode):
    """Berechnet den Achsenfehler je nach Modus.

    error_mode:
      - absolute: measured - true
      - ratio: measured / true
      - normalized: (measured - true) / true
    """
    if pd.isna(true_value) or pd.isna(measured_value):
        return None
    true_float = float(true_value)
    measured_float = float(measured_value)
    delta = measured_float - true_float

    if error_mode == 'absolute':
        return delta
    if abs(true_float) <= EPSILON:
        return None
    if error_mode == 'ratio':
        return measured_float / true_float
    return delta / true_float


def _apply_calibration_matrix(points_nt, calibration_data):
    h_matrix = np.array(calibration_data.get('matrix', []), dtype=float)
    if h_matrix.shape != (4, 4):
        return None

    # Translationseintraege liegen teils in Tesla und teils in nT vor.
    if np.max(np.abs(h_matrix[:3, 3])) < 1.0:
        h_matrix[:3, 3] = h_matrix[:3, 3] * 1e9

    ones = np.ones((len(points_nt), 1), dtype=float)
    points_h = np.hstack([points_nt[:, :3], ones])
    calibrated = (h_matrix @ points_h.T).T
    return calibrated[:, :3]


def _to_results_filename(export_subdir):
    safe_name = str(export_subdir).strip().replace(' ', '_')
    return f'Results_{safe_name}.csv'


def _find_seed_with_same_prefix(dataset_name, directory):
    """Findet eine CSV mit gleichem Namenspraefix (ohne letzten Zeitblock)."""
    if not dataset_name or not os.path.isdir(directory):
        return None

    if '_' not in dataset_name:
        return None

    prefix = f"{dataset_name.rsplit('_', 1)[0]}_"
    candidates = []

    for file_name in os.listdir(directory):
        if not file_name.lower().endswith('.csv'):
            continue
        stem = os.path.splitext(file_name)[0]
        if stem.startswith(prefix):
            candidates.append(stem)

    if not candidates:
        return None

    # Lexikografisch letzter Treffer ist i.d.R. der neueste Zeitstempel.
    return sorted(candidates)[-1]


def calculate_and_write_radius_errors(export_subdir, magnetic_field_strength_nt):
    """Berechnet Radiusfehler (xyz und xy) je Datensatz und schreibt in Results_<subdir>.csv."""
    if not export_subdir:
        return {'ok': False, 'message': 'Bitte einen Daten-Unterordner auswählen.', 'rows': 0}

    try:
        field_strength_nt = float(magnetic_field_strength_nt)
    except (TypeError, ValueError):
        return {'ok': False, 'message': 'Magnetische Feldstärke ist ungültig.', 'rows': 0}

    if field_strength_nt <= 0.0:
        return {'ok': False, 'message': 'Magnetische Feldstärke muss > 0 nT sein.', 'rows': 0}

    results_file = os.path.join(RESULTS_DIR, _to_results_filename(export_subdir))
    if not os.path.exists(results_file):
        return {'ok': False, 'message': f'Results-Datei nicht gefunden: {results_file}', 'rows': 0}

    export_dir = os.path.join(EXPORTS_DIR, export_subdir)
    calibration_dir = os.path.join(CALIBRATION_BASE_DIR, export_subdir)
    if not os.path.exists(export_dir):
        return {'ok': False, 'message': f'Export-Unterordner nicht gefunden: {export_dir}', 'rows': 0}
    if not os.path.exists(calibration_dir):
        return {'ok': False, 'message': f'Kalibrier-Unterordner nicht gefunden: {calibration_dir}', 'rows': 0}

    df = pd.read_csv(results_file)
    if 'datasetname' not in df.columns:
        return {'ok': False, 'message': 'Results-Datei enthält keine Spalte datasetname.', 'rows': 0}

    # Rueckwaertskompatibel: alte Spaltennamen uebernehmen.
    if 'RSME-xyz' not in df.columns:
        if 'RSME' in df.columns:
            df['RSME-xyz'] = pd.to_numeric(df['RSME'], errors='coerce')
        else:
            df['RSME-xyz'] = np.nan
    if 'MAE-xyz' not in df.columns:
        if 'MAE' in df.columns:
            df['MAE-xyz'] = pd.to_numeric(df['MAE'], errors='coerce')
        else:
            df['MAE-xyz'] = np.nan
    if 'Mean-xyz' not in df.columns:
        if 'Mean' in df.columns:
            df['Mean-xyz'] = pd.to_numeric(df['Mean'], errors='coerce')
        else:
            df['Mean-xyz'] = np.nan
    if 'RSME-xy' not in df.columns:
        df['RSME-xy'] = np.nan
    if 'MAE-xy' not in df.columns:
        df['MAE-xy'] = np.nan
    if 'Mean-xy' not in df.columns:
        df['Mean-xy'] = np.nan
    if 'Azimut-Max' not in df.columns:
        df['Azimut-Max'] = np.nan
    if 'Azimut-Mean' not in df.columns:
        df['Azimut-Mean'] = np.nan
    if 'Azimut-MAE' not in df.columns:
        df['Azimut-MAE'] = np.nan
    if 'Polar-Max' not in df.columns:
        df['Polar-Max'] = np.nan
    if 'Polar-Mean' not in df.columns:
        df['Polar-Mean'] = np.nan
    if 'Polar-MAE' not in df.columns:
        df['Polar-MAE'] = np.nan

    reference_index = _build_reference_index()

    field_strength_t = field_strength_nt / 1e9
    calculated_rows = 0
    skipped_missing_points = 0
    skipped_missing_calibration = 0
    skipped_invalid_matrix = 0
    skipped_reference_not_found = 0
    skipped_reference_mismatch = 0
    fallback_calibration_matches = 0

    total_rows = len(df.index)
    skipped_missing_id_matches = 0
    skipped_invalid_id_matches = 0

    for index, row in df.iterrows():
        dataset_name = str(row.get('datasetname') or '').strip()
        if not dataset_name:
            continue
        print(f"[Kalibrierfehler berechnen] Datensatz {index + 1}/{total_rows}: {dataset_name}")

        points_nt, metadata, load_error = load_csv_data_by_seed(
            dataset_name,
            suppress_error=True,
            directory=export_dir,
        )
        _points_with_id, noise_nt, test_ids, _id_in_file = _load_points_noise_ids(
            dataset_name,
            export_dir,
            metadata=metadata,
        )
        calibration_data, _ = load_calibration_data(dataset_name, calib_dir=calibration_dir)

        if calibration_data is None:
            fallback_seed = _find_seed_with_same_prefix(dataset_name, calibration_dir)
            if fallback_seed is not None and fallback_seed != dataset_name:
                calibration_data, _ = load_calibration_data(fallback_seed, calib_dir=calibration_dir)
                if calibration_data is not None:
                    fallback_calibration_matches += 1

        if load_error is not None or points_nt is None or noise_nt is None or test_ids is None:
            skipped_missing_points += 1
            df.at[index, 'RSME-xyz'] = np.nan
            df.at[index, 'MAE-xyz'] = np.nan
            df.at[index, 'Mean-xyz'] = np.nan
            df.at[index, 'RSME-xy'] = np.nan
            df.at[index, 'MAE-xy'] = np.nan
            df.at[index, 'Mean-xy'] = np.nan
            df.at[index, 'Azimut-Max'] = np.nan
            df.at[index, 'Azimut-Mean'] = np.nan
            df.at[index, 'Azimut-MAE'] = np.nan
            df.at[index, 'Polar-Max'] = np.nan
            df.at[index, 'Polar-Mean'] = np.nan
            df.at[index, 'Polar-MAE'] = np.nan
            continue

        if calibration_data is None:
            skipped_missing_calibration += 1
            df.at[index, 'RSME-xyz'] = np.nan
            df.at[index, 'MAE-xyz'] = np.nan
            df.at[index, 'Mean-xyz'] = np.nan
            df.at[index, 'RSME-xy'] = np.nan
            df.at[index, 'MAE-xy'] = np.nan
            df.at[index, 'Mean-xy'] = np.nan
            df.at[index, 'Azimut-Max'] = np.nan
            df.at[index, 'Azimut-Mean'] = np.nan
            df.at[index, 'Azimut-MAE'] = np.nan
            df.at[index, 'Polar-Max'] = np.nan
            df.at[index, 'Polar-Mean'] = np.nan
            df.at[index, 'Polar-MAE'] = np.nan
            continue

        calibrated_points_nt = _apply_calibration_matrix(points_nt, calibration_data)
        if calibrated_points_nt is None or len(calibrated_points_nt) == 0:
            skipped_invalid_matrix += 1
            df.at[index, 'RSME-xyz'] = np.nan
            df.at[index, 'MAE-xyz'] = np.nan
            df.at[index, 'Mean-xyz'] = np.nan
            df.at[index, 'RSME-xy'] = np.nan
            df.at[index, 'MAE-xy'] = np.nan
            df.at[index, 'Mean-xy'] = np.nan
            df.at[index, 'Azimut-Max'] = np.nan
            df.at[index, 'Azimut-Mean'] = np.nan
            df.at[index, 'Azimut-MAE'] = np.nan
            df.at[index, 'Polar-Max'] = np.nan
            df.at[index, 'Polar-Mean'] = np.nan
            df.at[index, 'Polar-MAE'] = np.nan
            continue

        # 3D-Radius: sqrt(x^2 + y^2 + z^2)
        radii_xyz_t = np.linalg.norm(calibrated_points_nt, axis=1) / 1e9
        deltas_xyz = radii_xyz_t - field_strength_t
        normalized_xyz = deltas_xyz / field_strength_t

        # 2D-Radius (z ignoriert): sqrt(x^2 + y^2)
        radii_xy_t = np.linalg.norm(calibrated_points_nt[:, :2], axis=1) / 1e9
        deltas_xy = radii_xy_t - field_strength_t
        normalized_xy = deltas_xy / field_strength_t

        rmse_xyz = float(np.sqrt(np.mean(np.square(deltas_xyz))))
        mae_xyz = float(np.mean(np.abs(deltas_xyz)))
        mean_normalized_xyz = float(np.mean(normalized_xyz))
        rmse_xy = float(np.sqrt(np.mean(np.square(deltas_xy))))
        mae_xy = float(np.mean(np.abs(deltas_xy)))
        mean_normalized_xy = float(np.mean(normalized_xy))

        df.at[index, 'RSME-xyz'] = rmse_xyz
        df.at[index, 'MAE-xyz'] = mae_xyz
        df.at[index, 'Mean-xyz'] = mean_normalized_xyz
        df.at[index, 'RSME-xy'] = rmse_xy
        df.at[index, 'MAE-xy'] = mae_xy
        df.at[index, 'Mean-xy'] = mean_normalized_xy

        match_key = _extract_match_key(dataset_name, metadata or {})
        if match_key is None or match_key not in reference_index:
            skipped_reference_not_found += 1
            df.at[index, 'Azimut-Max'] = np.nan
            df.at[index, 'Azimut-Mean'] = np.nan
            df.at[index, 'Azimut-MAE'] = np.nan
            df.at[index, 'Polar-Max'] = np.nan
            df.at[index, 'Polar-Mean'] = np.nan
            df.at[index, 'Polar-MAE'] = np.nan
            continue

        reference_points_nt = reference_index[match_key]['points']
        reference_ids = reference_index[match_key].get('ids')
        if reference_ids is None:
            reference_ids = np.arange(1, len(reference_points_nt) + 1, dtype=int)

        azimuth_errors, zenith_errors, matched_count, missing_in_ref, invalid_test_ids = _compute_azimuth_zenith_errors_deg_by_id(
            calibrated_points_nt,
            noise_nt,
            test_ids,
            reference_points_nt,
            reference_ids,
        )
        if azimuth_errors is None or zenith_errors is None:
            skipped_reference_mismatch += 1
            df.at[index, 'Azimut-Max'] = np.nan
            df.at[index, 'Azimut-Mean'] = np.nan
            df.at[index, 'Azimut-MAE'] = np.nan
            df.at[index, 'Polar-Max'] = np.nan
            df.at[index, 'Polar-Mean'] = np.nan
            df.at[index, 'Polar-MAE'] = np.nan
            continue

        skipped_missing_id_matches += missing_in_ref
        skipped_invalid_id_matches += invalid_test_ids

        df.at[index, 'Azimut-Max'] = float(np.max(azimuth_errors))
        df.at[index, 'Azimut-Mean'] = float(np.mean(azimuth_errors))
        df.at[index, 'Azimut-MAE'] = float(np.mean(np.abs(azimuth_errors)))
        df.at[index, 'Polar-Max'] = float(np.max(zenith_errors))
        df.at[index, 'Polar-Mean'] = float(np.mean(zenith_errors))
        df.at[index, 'Polar-MAE'] = float(np.mean(np.abs(zenith_errors)))
        calculated_rows += 1

    df.to_csv(results_file, index=False)
    return {
        'ok': True,
        'message': (
            f'Radiusfehler berechnet: {calculated_rows} Datensaetze aktualisiert. '
            f'Skipped fehlende Punkte: {skipped_missing_points}, '
            f'fehlende Kalibrierung: {skipped_missing_calibration}, '
            f'ungueltige Matrix/leere Punkte: {skipped_invalid_matrix}, '
            f'keine Referenz gefunden: {skipped_reference_not_found}, '
            f'keine gueltigen ID-Vergleiche: {skipped_reference_mismatch}, '
            f'ID-Matches fehlen: {skipped_missing_id_matches}, '
            f'ungueltige Test-IDs: {skipped_invalid_id_matches}, '
            f'Kalibrier-Fallback genutzt: {fallback_calibration_matches}.'
        ),
        'rows': calculated_rows,
        'results_file': results_file,
    }


def main():
    result = calculate_and_write_radius_errors(export_subdir='neu', magnetic_field_strength_nt=50000)
    print(result['message'])
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
