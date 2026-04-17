import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc, html, Input, Output, State, callback_context, no_update, callback
from datetime import datetime

from components.styles import (
    SIDEBAR_STYLE, COLLAPSED_STYLE, OUTPUT_DIR,
    DEFAULT_NOISE, DEFAULT_SAMPLES, DEFAULT_ALPHA
)
from components.layout import build_left_sidebar, build_center_area, build_right_sidebar
from utils.iron_math import (
    apply_soft_iron_transformation, create_iron_error_matrix, rng
)
from utils.sphere import fibonacci_sphere, create_sphere_mesh
from utils.plotting import build_figure_with_points, build_results_display, apply_axes_legend_scale
from utils.csv_io import (
    load_csv_data_by_seed, get_available_seeds, resolve_input_data,
    load_calibration_data, get_seeds_from_dir, CALIBRATION_DIR,
    get_dataset_subdir_options, resolve_dataset_directory, get_default_dataset_subdir
)


RESULTS_DIR = os.path.join('datasets', '4-calibrated_exports_for_analysis')
CORRECTION_DATASETS_DIR = os.path.join('datasets', '5-correction_datasets')
DEFAULT_FLUX_DENSITY_NT = 49750.0
PLOT_EXPORT_DIR = os.path.join(OUTPUT_DIR, 'plot_html_exports')


def _camera_for_view(projection_mode, plane_mode):
    projection_type = 'orthographic' if projection_mode == 'isometric' else 'perspective'
    center = {'x': 0, 'y': 0, 'z': 0}

    if plane_mode == 'iso':
        return {
            'up': {'x': 0, 'y': 0, 'z': 1},
            'center': {'x': 0, 'y': 0, 'z': -0.1},
            'eye': {'x': -1.3, 'y': -1.3, 'z': 1.3},
            'projection': {'type': projection_type},
        }

    if plane_mode == 'xz':
        return {
            'up': {'x': 0, 'y': 0, 'z': 1},
            'center': center,
            'eye': {'x': 0.0, 'y': -2.0, 'z': 0.0},
            'projection': {'type': projection_type},
        }
    if plane_mode == 'yz':
        return {
            'up': {'x': 0, 'y': 0, 'z': 1},
            'center': center,
            'eye': {'x': -2.0, 'y': 0, 'z': 0},
            'projection': {'type': projection_type},
        }
    if plane_mode == 'xy':
        return {
            'up': {'x': 0, 'y': 1, 'z': 0},
            'center': center,
            'eye': {'x': 0, 'y': 0, 'z': 2.0},
            'projection': {'type': projection_type},
        }

    return {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': center,
        'eye': {'x': 1.25, 'y': 1.25, 'z': 1.25},
        'projection': {'type': projection_type},
    }


def _apply_view_camera(fig, projection_mode, plane_mode):
    camera = _camera_for_view(projection_mode or 'perspective', plane_mode or 'iso')
    fig.update_layout(scene_camera=camera)
    return fig


def _build_plot_html_filename(current_seed, export_filename):
    base_name = current_seed or export_filename or 'kalipoints_plot'
    base_name = os.path.splitext(str(base_name))[0]
    base_name = re.sub(r'[<>:"/\\|?*]+', '_', base_name).strip(' ._-')
    if not base_name:
        base_name = 'kalipoints_plot'
    return f'{base_name}.html'


def _build_standalone_plot_html(fig):
    export_fig = go.Figure(fig)
    export_fig.update_layout(
        autosize=True,
        width=None,
        height=None,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    if export_fig.layout.showlegend is not False:
        export_fig.update_layout(
            legend=dict(
                orientation='h',
                x=0.5,
                xanchor='center',
                y=1,
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.75)',
            )
        )
    plot_div = pio.to_html(
        export_fig,
        include_plotlyjs='inline',
        full_html=False,
        default_width='100vw',
        default_height='100vh',
        config={
            'responsive': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'kalipoints_plot',
                'scale': 1,
            },
        },
    )
    return f"""<!DOCTYPE html>
<html lang=\"de\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>Kalipoints Plot Export</title>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background: #ffffff;
            overflow: hidden;
            font-family: Arial, sans-serif;
        }}
        .plot-shell {{
            width: 100vw;
            height: 100vh;
        }}
        .plot-shell .plotly-graph-div {{
            width: 100% !important;
            height: 100% !important;
        }}
    </style>
</head>
<body>
    <div class=\"plot-shell\">{plot_div}</div>
    <script>
        (function() {{
            const resizePlot = () => {{
                const plot = document.querySelector('.plotly-graph-div');
                if (!plot || !window.Plotly) {{
                    return;
                }}

                const width = window.innerWidth;
                const height = window.innerHeight;
                const update = {{
                    autosize: true,
                    width: width,
                    height: height,
                    margin: {{ l: 0, r: 0, b: 0, t: 0 }},
                    legend: {{
                        orientation: 'h',
                        x: 0.5,
                        xanchor: 'center',
                        y: 1,
                        yanchor: 'top'
                    }}
                }};

                if (plot.layout && plot.layout.scene) {{
                    update['scene.domain.x'] = [0, 1];
                    update['scene.domain.y'] = [0, 1];
                    update['scene.aspectmode'] = plot.layout.scene.aspectmode || 'cube';
                }}

                window.Plotly.relayout(plot, update).then(function() {{
                    window.Plotly.Plots.resize(plot);
                }}).catch(function() {{
                    window.Plotly.Plots.resize(plot);
                }});
            }};

            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', function() {{
                    window.setTimeout(resizePlot, 0);
                    window.setTimeout(resizePlot, 150);
                }});
            }} else {{
                window.setTimeout(resizePlot, 0);
                window.setTimeout(resizePlot, 150);
            }}

            window.addEventListener('resize', resizePlot);
        }})();
    </script>
</body>
</html>
"""


def _metadata_value(metadata_text, key):
    if not isinstance(metadata_text, str):
        return None
    match = re.search(rf"{re.escape(key)}:\s*([^,\)]+)", metadata_text)
    return match.group(1).strip() if match else None


def _metadata_bool(metadata_text, key):
    value = _metadata_value(metadata_text, key)
    if value is None:
        return None
    return value.lower() == 'true'


def _normalize_axis_constraint_mode(axis_constraint_mode):
    normalized = str(axis_constraint_mode or '').strip().lower()
    if normalized in ('pitch_only', 'nicken_ohne_rollen', 'ohne_rollen'):
        return 'pitch_only'
    return 'pitch_roll'


def _iron_error_category(hi_active, si_dist_active, si_rot_active):
    if hi_active and si_dist_active and si_rot_active:
        return 'all'
    if hi_active and si_dist_active and not si_rot_active:
        return 'hi_si_dist'
    if hi_active and not si_dist_active and si_rot_active:
        return 'hi_si_rot'
    if not hi_active and si_dist_active and si_rot_active:
        return 'si_dist_si_rot'
    if hi_active and not si_dist_active and not si_rot_active:
        return 'hi_only'
    if not hi_active and si_dist_active and not si_rot_active:
        return 'si_dist_only'
    if not hi_active and not si_dist_active and si_rot_active:
        return 'si_rot_only'
    return 'none'


def _results_filename_for_subdir(dataset_subdir):
    safe_name = str(dataset_subdir or '').strip().replace(' ', '_')
    return f'Results_{safe_name}.csv'


def _normalize_dataset_name(value):
    name = str(value or '').strip()
    if name.lower().endswith('.csv'):
        name = name[:-4]
    return name.lower()


def _normalize_results_stem(value):
    return re.sub(r'[^a-z0-9]+', '', str(value or '').lower())


def _read_csv_points_and_ids(csv_path):
    """Liest X/Y/Z (+ optional ID) aus CSV und gibt Punkte in nT zurück."""
    if not csv_path or not os.path.exists(csv_path):
        return None, None

    try:
        df = pd.read_csv(csv_path, sep=';', comment='#')
    except Exception:
        return None, None

    required_cols = {'X', 'Y', 'Z'}
    if not required_cols.issubset(df.columns):
        return None, None

    xyz_df = df[['X', 'Y', 'Z']].apply(pd.to_numeric, errors='coerce')
    if 'ID' in df.columns:
        ids_series = pd.to_numeric(df['ID'], errors='coerce')
    else:
        ids_series = pd.Series(np.arange(1, len(df) + 1, dtype=int), index=df.index, dtype='float64')

    valid_mask = xyz_df.notna().all(axis=1) & ids_series.notna()
    if not valid_mask.any():
        return None, None

    xyz_df = xyz_df.loc[valid_mask]
    ids_series = ids_series.loc[valid_mask].astype(int)

    points = xyz_df.to_numpy(dtype=float)
    if points.size:
        max_abs = np.nanmax(np.abs(points))
        # Tesla-Dateien in nT umrechnen.
        if np.isfinite(max_abs) and max_abs < 1e-2:
            points = points * 1e9

    return points, ids_series.to_numpy(dtype=int)


def _load_point_ids_for_seed(seed_string, dataset_directory, fallback_count):
    csv_path = os.path.join(dataset_directory, f"{seed_string}.csv")
    _points, ids = _read_csv_points_and_ids(csv_path)
    if ids is None or len(ids) == 0:
        return np.arange(1, int(fallback_count) + 1, dtype=int)
    return ids


def _load_noise_by_id_for_seed(seed_string, dataset_directory):
    """Lädt Rauschwerte (X_noise/Y_noise/Z_noise) je ID in nT aus einem Datensatz."""
    csv_path = os.path.join(dataset_directory, f"{seed_string}.csv")
    if not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path, sep=';', comment='#')
    except Exception:
        return {}

    if not {'X_noise', 'Y_noise', 'Z_noise'}.issubset(df.columns):
        return {}

    noise_df = df[['X_noise', 'Y_noise', 'Z_noise']].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    noise = noise_df.to_numpy(dtype=float)
    if noise.size:
        noise_max = np.nanmax(np.abs(noise))
        if np.isfinite(noise_max) and noise_max < 1e-2:
            noise = noise * 1e9

    if 'ID' in df.columns:
        ids = pd.to_numeric(df['ID'], errors='coerce').fillna(-1).astype(int).to_numpy()
    else:
        ids = np.arange(1, len(df) + 1, dtype=int)

    noise_by_id = {}
    for idx, point_id in enumerate(ids):
        point_id_int = int(point_id)
        if point_id_int <= 0:
            continue
        noise_by_id[point_id_int] = noise[idx]
    return noise_by_id


def _find_correction_dataset_for_point_amount(point_amount):
    if point_amount is None or not os.path.exists(CORRECTION_DATASETS_DIR):
        return None
    try:
        target = int(point_amount)
    except Exception:
        return None

    pattern = re.compile(rf"_e_{target}_", re.IGNORECASE)
    candidates = []
    for filename in os.listdir(CORRECTION_DATASETS_DIR):
        if not filename.lower().endswith('.csv'):
            continue
        if pattern.search(filename):
            candidates.append(filename)

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return os.path.join(CORRECTION_DATASETS_DIR, candidates[0])


def _load_optimal_points_for_ids(point_amount, allowed_ids):
    """Lädt optimale Punkte passend zur Punktezahl und filtert sie über die übergebenen IDs."""
    if not allowed_ids:
        return None, None

    correction_path = _find_correction_dataset_for_point_amount(point_amount)
    if not correction_path:
        return None, None

    points, ids = _read_csv_points_and_ids(correction_path)
    if points is None or ids is None:
        return None, None

    ordered_allowed_ids = []
    seen = set()
    for raw_id in allowed_ids:
        try:
            norm_id = int(raw_id)
        except Exception:
            continue
        if norm_id in seen:
            continue
        seen.add(norm_id)
        ordered_allowed_ids.append(norm_id)

    if not ordered_allowed_ids:
        return None, None

    id_to_point = {}
    for index, point_id in enumerate(ids):
        if int(point_id) not in id_to_point:
            id_to_point[int(point_id)] = points[index]

    filtered_points = [id_to_point[point_id] for point_id in ordered_allowed_ids if point_id in id_to_point]
    if not filtered_points:
        return None, None

    filtered_ids = [point_id for point_id in ordered_allowed_ids if point_id in id_to_point]

    return np.array(filtered_points, dtype=float), np.array(filtered_ids, dtype=int)


def _results_path_candidates(dataset_subdir):
    if not dataset_subdir:
        return [os.path.join(RESULTS_DIR, 'Results.csv')]

    raw = str(dataset_subdir).strip().replace('\\', '/').strip('/')
    if not raw:
        return [os.path.join(RESULTS_DIR, 'Results.csv')]

    leaf = raw.split('/')[-1]
    variants = [
        raw,
        raw.replace(' ', '_'),
        raw.replace('-', '_'),
        raw.replace('_', '-'),
        leaf,
        leaf.replace(' ', '_'),
        leaf.replace('-', '_'),
        leaf.replace('_', '-'),
    ]

    unique_variants = []
    seen = set()
    for variant in variants:
        key = variant.lower()
        if key and key not in seen:
            seen.add(key)
            unique_variants.append(variant)

    return [os.path.join(RESULTS_DIR, f'Results_{variant}.csv') for variant in unique_variants]


def _load_results_df(dataset_subdir=None):
    for path in _results_path_candidates(dataset_subdir):
        if not os.path.exists(path):
            continue
        try:
            return pd.read_csv(path)
        except Exception:
            continue

    # Last fallback: tolerant match against any Results_*.csv by stem.
    if dataset_subdir and os.path.isdir(RESULTS_DIR):
        target_stem = _normalize_results_stem(dataset_subdir)
        for filename in os.listdir(RESULTS_DIR):
            if not filename.lower().startswith('results_') or not filename.lower().endswith('.csv'):
                continue
            stem = filename[len('Results_'):-4]
            if _normalize_results_stem(stem) != target_stem:
                continue
            path = os.path.join(RESULTS_DIR, filename)
            try:
                return pd.read_csv(path)
            except Exception:
                continue

    return None


def _get_calibration_errors_for_seed(seed_name, dataset_subdir=None):
    results_df = _load_results_df(dataset_subdir)
    if results_df is None or 'datasetname' not in results_df.columns:
        return None

    seed_key = _normalize_dataset_name(seed_name)
    dataset_keys = results_df['datasetname'].astype(str).map(_normalize_dataset_name)
    row_match = results_df[dataset_keys == seed_key]
    if row_match.empty:
        return None

    row = row_match.iloc[0]

    def _to_float(value):
        try:
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    return {
        'angle': {
            'azimuth_mean_deg': _to_float(row.get('Azimut-Mean')),
            'zenith_mean_deg': _to_float(row.get('Polar-Mean')),
            'azimuth_max_deg': _to_float(row.get('Azimut-Max')),
            'zenith_max_deg': _to_float(row.get('Polar-Max')),
        },
        'radius': {
            # In der Results-CSV liegen diese Werte in Tesla vor.
            'mean_t': _to_float(row.get('Mean-xyz')),
            'mae_t': _to_float(row.get('MAE-xyz')),
            'rmse_t': _to_float(row.get('RSME-xyz')),
        },
    }


def _filter_simulated_seed_options(iron_error_filter, point_amount_filter, keep_density_filter, axis_constraint_filter,
                                   angle_range_filter, dataset_subdir=None, base_options=None):
    if base_options is None:
        base_options = get_available_seeds()
    valid_base = [
        option for option in base_options
        if not option.get('disabled') and option.get('value') not in (None, '', 'none')
    ]
    if not valid_base:
        return base_options

    results_df = _load_results_df(dataset_subdir)
    if results_df is None or 'datasetname' not in results_df.columns:
        return base_options

    results_by_name = {}
    for _, row in results_df.iterrows():
        key = _normalize_dataset_name(row.get('datasetname'))
        if key and key not in results_by_name:
            results_by_name[key] = row
    has_restrictive_filter = bool(iron_error_filter or point_amount_filter or keep_density_filter or axis_constraint_filter)

    min_angle, max_angle = 5, 90
    if isinstance(angle_range_filter, (list, tuple)) and len(angle_range_filter) == 2:
        min_angle, max_angle = angle_range_filter
    if min_angle != 5 or max_angle != 90:
        has_restrictive_filter = True

    filtered = []
    for option in valid_base:
        seed_name = option['value']
        row = results_by_name.get(_normalize_dataset_name(seed_name))

        if row is None:
            if not has_restrictive_filter:
                filtered.append(option)
            continue

        metadata_text = row.get('Metadata', '')
        hi_active = _metadata_bool(metadata_text, 'HI-X-Y-Z-OFFSET')
        si_dist_active = _metadata_bool(metadata_text, 'SI-X-Y-Z-DISTORTION')
        si_rot_active = _metadata_bool(metadata_text, 'SI-X-Y-Z-ROTATION-DEG')
        point_amount = _metadata_value(metadata_text, 'POINT_AMOUNT')
        keep_density = _metadata_bool(metadata_text, 'KEEP POINT DENSITY')
        axis_constraint = _normalize_axis_constraint_mode(
            _metadata_value(metadata_text, 'AXIS_CONSTRAINT') or _metadata_value(metadata_text, 'AXSIS_CONSTRAINT')
        )
        angular_constraint = _metadata_value(metadata_text, 'ANGULAR_CONSTRAINT-DEG')

        if hi_active is None or si_dist_active is None or si_rot_active is None:
            if not has_restrictive_filter:
                filtered.append(option)
            continue

        category = _iron_error_category(hi_active, si_dist_active, si_rot_active)

        if iron_error_filter and category != iron_error_filter:
            continue

        if point_amount_filter is not None:
            try:
                if int(float(point_amount)) != int(point_amount_filter):
                    continue
            except Exception:
                continue

        if keep_density_filter in ('on', 'off'):
            keep_density_target = keep_density_filter == 'on'
            if keep_density is None or keep_density != keep_density_target:
                continue

        if axis_constraint_filter in ('pitch_roll', 'pitch_only'):
            if axis_constraint != axis_constraint_filter:
                continue

        try:
            angular_value = float(angular_constraint)
        except Exception:
            angular_value = None

        if angular_value is None:
            if min_angle != 5 or max_angle != 90:
                continue
        elif angular_value < float(min_angle) or angular_value > float(max_angle):
            continue

        filtered.append(option)

    if not filtered:
        return [{'label': 'Keine Datensätze gefunden', 'value': 'none', 'disabled': True}]

    return filtered


def create_layout():
    """Erstellt das komplette Layout der Kalipoints-Seite."""
    return html.Div([
        dcc.Store(id='left-sidebar-state', data=True),
        dcc.Store(id='right-sidebar-state', data=True),
        dcc.Store(id='point-data-storage', data=None),
        dcc.Store(id='seed-data-storage', data=None),
        dcc.Store(id='loaded-seed-string-store', data=""),
        dcc.Store(id='current-seed-store', data=""),
        dcc.Store(id='export-filename-store', data=""),
        dcc.Store(id='export-success-trigger', data=0),
        dcc.Store(id='magnetic-error-values-store', data=None),
        dcc.Store(id='calibration-data-store', data=None),
        dcc.Store(id='show-uncalibrated-store', data=True),
        dcc.Store(id='show-calibrated-store', data=True),
        dcc.Store(id='show-optimal-store', data=False),
        dcc.Store(id='dataset-source-store', data='exports'),
        dcc.Store(id='dataset-subdir-store', data='neu'),
        dcc.Store(id='calibration-dir-store', data=CALIBRATION_DIR),
        dcc.Store(id='raw-points-store', data=None),
        dcc.Store(id='view-projection-store', data='perspective'),
        dcc.Store(id='view-plane-store', data='iso'),
        dcc.Store(id='show-origin-store', data=True),
        dcc.Store(id='scale-axes-legend-store', data=False),
        dcc.Store(id='sim-filter-applied-store', data={
            'iron': None,
            'point': None,
            'density': None,
            'axis': None,
            'angle': [5, 90],
        }),
        html.Div(id='download-output', style={'display': 'none'}),
        dcc.Download(id="download-csv"),
        dcc.Download(id='download-plot-html'),

        html.Div([
            build_left_sidebar(),
            build_center_area(),
            build_right_sidebar(),
        ], style={'display': 'flex', 'flexDirection': 'row', 'height': '100vh', 'width': '100%', 'overflow': 'hidden'})
    ])


def _generate_hi_params(hard_iron_mode, hard_iron_random_type,
                        manual_x, manual_y, manual_z,
                        rand_min_coll, rand_max_coll,
                        rand_min_x, rand_max_x,
                        rand_min_y, rand_max_y,
                        rand_min_z, rand_max_z):
    """Bestimmt Hard Iron Offset basierend auf Modus (Manuell/Zufällig)."""
    if hard_iron_mode == 'hard-iron-manual':
        return float(manual_x or 0), float(manual_y or 0), float(manual_z or 0)
    elif hard_iron_mode == 'hard-iron-random':
        if hard_iron_random_type == 'hi-random-collective':
            lo = float(rand_min_coll) if rand_min_coll is not None else 0.0
            hi = float(rand_max_coll) if rand_max_coll is not None else 0.0
            return rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)
        else:
            return (rng.uniform(float(rand_min_x or 0), float(rand_max_x or 0)),
                    rng.uniform(float(rand_min_y or 0), float(rand_max_y or 0)),
                    rng.uniform(float(rand_min_z or 0), float(rand_max_z or 0)))
    return 0.0, 0.0, 0.0


def _generate_si_params(soft_iron_mode, soft_iron_random_type,
                        manual_dist_x, manual_dist_y, manual_dist_z,
                        manual_rot_x, manual_rot_y, manual_rot_z,
                        dist_min_coll, dist_max_coll, rot_min_coll, rot_max_coll,
                        dist_min_x, dist_max_x, dist_min_y, dist_max_y, dist_min_z, dist_max_z,
                        rot_min_x, rot_max_x, rot_min_y, rot_max_y, rot_min_z, rot_max_z):
    """Bestimmt Soft Iron Parameter basierend auf Modus (Manuell/Zufällig)."""
    if soft_iron_mode == 'soft-iron-manual':
        return (float(manual_dist_x or 1.0), float(manual_dist_y or 1.0), float(manual_dist_z or 1.0),
                float(manual_rot_x or 0), float(manual_rot_y or 0), float(manual_rot_z or 0))
    elif soft_iron_mode == 'soft-iron-random':
        if soft_iron_random_type == 'si-random-collective':
            d_lo = float(dist_min_coll) if dist_min_coll is not None else 1.0
            d_hi = float(dist_max_coll) if dist_max_coll is not None else 1.0
            r_lo = float(rot_min_coll) if rot_min_coll is not None else 0.0
            r_hi = float(rot_max_coll) if rot_max_coll is not None else 0.0
            return (rng.uniform(d_lo, d_hi), rng.uniform(d_lo, d_hi), rng.uniform(d_lo, d_hi),
                    rng.uniform(r_lo, r_hi), rng.uniform(r_lo, r_hi), rng.uniform(r_lo, r_hi))
        else:
            return (rng.uniform(float(dist_min_x or 1.0), float(dist_max_x or 1.0)),
                    rng.uniform(float(dist_min_y or 1.0), float(dist_max_y or 1.0)),
                    rng.uniform(float(dist_min_z or 1.0), float(dist_max_z or 1.0)),
                    rng.uniform(float(rot_min_x or 0.0), float(rot_max_x or 0.0)),
                    rng.uniform(float(rot_min_y or 0.0), float(rot_max_y or 0.0)),
                    rng.uniform(float(rot_min_z or 0.0), float(rot_max_z or 0.0)))
    return 1.0, 1.0, 1.0, 0.0, 0.0, 0.0


def _write_single_csv(filepath, df, distribution_style, noise, point_amount,
                      angular_constraint_deg, hi, si_dist, si_rot, iron_error_matrix,
                      true_magnetic_field_strength=DEFAULT_FLUX_DENSITY_NT, keep_point_density=False,
                      axis_constraint='pitch_roll', export_timestamp=None):
    """Schreibt eine einzelne CSV-Datei mit Metadaten-Header. Werte in Tesla."""
    with open(filepath, 'w', newline='') as f:
        f.write(f"# UNIT: Tesla\n")
        f.write(f"# DISTRIBUTION_STYLE: {distribution_style}\n")
        if export_timestamp is not None:
            f.write(f"# EXPORT_DATE: {export_timestamp.strftime('%d-%m-%y')}\n")
            f.write(f"# EXPORT_TIME: {export_timestamp.strftime('%H:%M:%S')}\n")
        f.write(f"# TRUE MAGNETIC FIELD STRENGTH: {true_magnetic_field_strength}\n")
        f.write(f"# NOISE: {noise}\n")
        f.write(f"# POINT_AMOUNT: {point_amount}\n")
        f.write(f"# ANGULAR_CONSTRAINT-DEG: {angular_constraint_deg}\n")
        f.write(f"# KEEP POINT DENSITY: {keep_point_density}\n")
        f.write(f"# AXIS_CONSTRAINT: {_normalize_axis_constraint_mode(axis_constraint)}\n")
        f.write(f"# FIELD_LINE_ANGLE-DEG: 0 (-> nicht implementiert)\n")
        # HI in Tesla (Eingabe war nT, /1e9 fuer Tesla)
        f.write(f"# HI-X-Y-Z-OFFSET: {hi[0]/1e9},{hi[1]/1e9},{hi[2]/1e9}\n")
        f.write(f"# SI-X-Y-Z-DISTORTION: {si_dist[0]},{si_dist[1]},{si_dist[2]}\n")
        f.write(f"# SI-X-Y-Z-ROTATION-DEG: {si_rot[0]},{si_rot[1]},{si_rot[2]}\n")
        if iron_error_matrix is not None:
            f.write("# IRON_ERROR_MATRIX_FORMAT-RAD:\n")
            for row in iron_error_matrix:
                row_vals = row if isinstance(row, list) else row.tolist()
                f.write(f"# [{', '.join([f'{val:.8f}' for val in row_vals])}]\n")
        # Export-Format sicherstellen: X;Y;Z;X_noise;Y_noise;Z_noise;ID
        df_tesla = df.copy()
        row_count = len(df_tesla.index)
        if 'X_noise' not in df_tesla.columns:
            df_tesla['X_noise'] = 0.0
        if 'Y_noise' not in df_tesla.columns:
            df_tesla['Y_noise'] = 0.0
        if 'Z_noise' not in df_tesla.columns:
            df_tesla['Z_noise'] = 0.0
        if 'ID' not in df_tesla.columns:
            df_tesla['ID'] = np.arange(1, row_count + 1, dtype=int)

        # Messwerte von nT in Tesla umrechnen (inkl. Noise-Komponenten)
        df_tesla[['X', 'Y', 'Z']] = df_tesla[['X', 'Y', 'Z']] / 1e9
        df_tesla[['X_noise', 'Y_noise', 'Z_noise']] = df_tesla[['X_noise', 'Y_noise', 'Z_noise']] / 1e9
        df_tesla = df_tesla[['X', 'Y', 'Z', 'X_noise', 'Y_noise', 'Z_noise', 'ID']]
        df_tesla.to_csv(f, index=False, sep=';', float_format='%.12e')


def _build_filename(alpha, gen_mode, sample_count, custom_text, current_num, total_sets, current_time,
                    axis_constraint_mode='pitch_roll'):
    """Baut den Dateinamen nach vorgegebenem Schema."""
    mode_char = 'e' if gen_mode in ('optimal', 'evenly') else 'r' if gen_mode in ('random', 'randomly') else 'p'
    axis_marker = 'R'
    if _normalize_axis_constraint_mode(axis_constraint_mode) == 'pitch_only':
        axis_marker = 'N'
    date_str = current_time.strftime('%d-%m-%y')
    time_str = current_time.strftime('%H-%M')
    return f"{int(alpha)}_{mode_char}_{int(sample_count)}_0_{custom_text}_{current_num}-{total_sets}_{axis_marker}_{date_str}_{time_str}.csv"


def register_callbacks(app):
    """Registriert alle Callbacks für die Kalipoints-Seite."""

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

    # B. Callback: Batch-Eingabefeld und Winkeleinschränkung aktivieren/deaktivieren
    @app.callback(
        [Output('batch-step-input', 'disabled'),
         Output('batch-step-input', 'style'),
         Output('angular-constraint-input', 'disabled'),
         Output('angular-constraint-input', 'style')],
        [Input('batch-mode-toggle', 'value')]
    )
    def toggle_batch_input(batch_values):
        disabled_style = {'flexGrow': 1, 'minWidth': '0', 'backgroundColor': '#e9ecef', 'color': '#6c757d'}
        enabled_style = {'flexGrow': 1, 'minWidth': '0'}
        is_batch = 'batch' in (batch_values or [])
        if is_batch:
            return False, enabled_style, True, disabled_style
        else:
            return True, disabled_style, False, enabled_style

    @app.callback(
        [Output('axis-constraint-mode', 'options'),
         Output('axis-constraint-mode', 'style')],
        Input('density-mode-toggle', 'value')
    )
    def toggle_axis_constraint_control(density_values):
        density_on = 'density' in (density_values or [])
        options = [
            {'label': ' Nicken und Rollen', 'value': 'pitch_roll', 'disabled': not density_on},
            {'label': ' Nicken ohne Rollen', 'value': 'pitch_only', 'disabled': not density_on},
        ]
        style = {'fontSize': '0.8em', 'opacity': 1.0 if density_on else 0.55}
        return options, style

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
         Output('noise-input', 'value'),
         Output('density-mode-toggle', 'value'),
         Output('axis-constraint-mode', 'value')],
        [Input('seed-data-storage', 'data')],
        prevent_initial_call=True
    )
    def populate_ui_from_seed(seed_data):
        """
        Füllt die UI-Eingabefelder mit Werten aus dem geladenen Seed.
        Unterstützt sowohl alte als auch neue Metadaten-Formate.
        """
        if seed_data is None or not isinstance(seed_data, dict):
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
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
        keep_density = seed_data.get('KEEP_POINT_DENSITY')
        density_toggle_value = no_update if keep_density is None else (['density'] if bool(keep_density) else [])
        axis_constraint_mode = _normalize_axis_constraint_mode(seed_data.get('AXIS_CONSTRAINT'))
        
        return x_offset, y_offset, z_offset, x_distortion, y_distortion, z_distortion, x_rotation, y_rotation, z_rotation, generation_mode, alpha, sample_count, noise, density_toggle_value, axis_constraint_mode

    # D. Callback für den Graphen (Berechnung)
    @app.callback(
        [Output('sphere-plot', 'figure'),
         Output('results-container', 'children'),
         Output('point-data-storage', 'data'),
         Output('display-time', 'children'),
         Output('seed-data-storage', 'data', allow_duplicate=True),
         Output('current-seed-store', 'data'),
         Output('export-filename-store', 'data'),
         Output('magnetic-error-values-store', 'data'),
         Output('raw-points-store', 'data')],
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
         State('loaded-seed-string-store', 'data'),
         State('calibration-data-store', 'data'),
         State('dataset-source-store', 'data'),
         State('dataset-subdir-store', 'data'),
         State('flux-density-input', 'value'),
         State('density-mode-toggle', 'value'),
         State('axis-constraint-mode', 'value'),
         State('view-projection-store', 'data'),
         State('view-plane-store', 'data'),
         State('show-origin-store', 'data'),
         State('scale-axes-legend-store', 'data')], 
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
                     ui_mesh_opacity, loaded_seed_string, calibration_store_data, dataset_source,
                     dataset_subdir, flux_density_nT, density_mode_values, axis_constraint_mode, view_projection_mode, view_plane_mode,
                     show_origin, scale_axes_legend):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        current_time = datetime.now()
        current_time_str = current_time.strftime("%H:%M:%S")

        # Neue Parameter: Flussdichte / Punktedichte
        flux_radius = float(flux_density_nT) if flux_density_nT else DEFAULT_FLUX_DENSITY_NT
        maintain_density = 'density' in (density_mode_values or [])
        axis_constraint_mode = _normalize_axis_constraint_mode(axis_constraint_mode)

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
        if hard_iron_mode == 'hard-iron-manual':
            hi_x_offset = params['x_offset']
            hi_y_offset = params['y_offset']
            hi_z_offset = params['z_offset']
        elif hard_iron_mode == 'hard-iron-random':
            if hard_iron_random_type_mode == 'hi-random-collective':
                min_val = float(hi_rand_min_collective) if hi_rand_min_collective is not None else 0.0
                max_val = float(hi_rand_max_collective) if hi_rand_max_collective is not None else 0.0
                hi_x_offset = rng.uniform(min_val, max_val)
                hi_y_offset = rng.uniform(min_val, max_val)
                hi_z_offset = rng.uniform(min_val, max_val)
            else:  # hi-random-specific
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
            hi_x_offset = 0.0
            hi_y_offset = 0.0
            hi_z_offset = 0.0
        
        # KRITISCH: Unterscheide zwischen zwei Fällen:
        # 1. Seed wurde von außen geladen (trigger_id == 'seed-data-storage') -> LADE PUNKTE AUS CSV!
        # 2. Benutzer klickt "Punkte Erzeugen" (trigger_id == 'submit-button') -> generiere NEUEN SEED
        
        csv_metadata = None
        point_ids = None
        noise_by_id = {}
        data_origin = 'generated'
        source_directory = resolve_dataset_directory(dataset_source, dataset_subdir)
        
        if trigger_id == 'seed-data-storage' and loaded_seed_string:
            # Seed wurde von außen geladen -> VERWENDE GESPEICHERTE PUNKTE AUS CSV!
            seed_string = loaded_seed_string
            display_time_str = f"Uhrzeit (aus Datensatz): ({current_time_str})"
            
            data_points, csv_metadata, load_error = load_csv_data_by_seed(
                seed_string,
                directory=source_directory
            )
            
            if load_error or data_points is None:
                # Fallback: Generiere Punkte wenn CSV nicht existiert
                si_x_scale = params['x_distortion']
                si_y_scale = params['y_distortion']
                si_z_scale = params['z_distortion']
                si_x_rot = x_rotation if x_rotation else 0
                si_y_rot = y_rotation if y_rotation else 0
                si_z_rot = z_rotation if z_rotation else 0
                
                data_points, actual_hi_offset, noise_points, point_ids = fibonacci_sphere(
                    params['generation_mode'], params['noise_value'], params['sample_count'], 
                    params['alpha'], seed_string, 
                    xyz0=[hi_x_offset, hi_y_offset, hi_z_offset],
                    xyz1=[si_x_scale, si_y_scale, si_z_scale],
                    x_rot=si_x_rot, y_rot=si_y_rot, z_rot=si_z_rot,
                    radius=flux_radius, maintain_density=maintain_density,
                    axis_constraint_mode=axis_constraint_mode
                )
                noise_by_id = {int(pid): noise_points[idx] for idx, pid in enumerate(point_ids)}
                csv_metadata = None
                data_origin = 'generated'
            else:
                actual_hi_offset = csv_metadata.get('HI_OFFSET') or csv_metadata.get('HI_X_Y_Z_OFFSET') or [params['x_offset'], params['y_offset'], params['z_offset']]
                axis_constraint_mode = _normalize_axis_constraint_mode(csv_metadata.get('AXIS_CONSTRAINT') or axis_constraint_mode)
                si_dist = csv_metadata.get('SI_DISTORTION') or csv_metadata.get('SI_X_Y_Z_DISTORTION') or [params['x_distortion'], params['y_distortion'], params['z_distortion']]
                si_x_scale = si_dist[0] if si_dist and len(si_dist) > 0 else params['x_distortion']
                si_y_scale = si_dist[1] if si_dist and len(si_dist) > 1 else params['y_distortion']
                si_z_scale = si_dist[2] if si_dist and len(si_dist) > 2 else params['z_distortion']
                si_rot = csv_metadata.get('SI_ROTATION_DEG') or csv_metadata.get('SI_ROTATION', [0.0, 0.0, 0.0])
                si_x_rot = si_rot[0] if si_rot and len(si_rot) > 0 else 0.0
                si_y_rot = si_rot[1] if si_rot and len(si_rot) > 1 else 0.0
                si_z_rot = si_rot[2] if si_rot and len(si_rot) > 2 else 0.0
                point_ids = _load_point_ids_for_seed(seed_string, source_directory, len(data_points))
                noise_by_id = _load_noise_by_id_for_seed(seed_string, source_directory)
                data_origin = 'loaded'
        else:
            # Benutzer hat "Punkte Erzeugen" geklickt -> GENERIERE NEUEN SEED
            seed_string = f"{int(params['alpha'])}_{params['generation_mode']}_{current_time.strftime('%y-%m-%d_%H-%M-%S')}"
            display_time_str = f"Uhrzeit: ({current_time_str})"
            
            # --- SOFT IRON RANDOM LOGIC ---
            if soft_iron_mode == 'soft-iron-manual':
                si_x_scale = params['x_distortion']
                si_y_scale = params['y_distortion']
                si_z_scale = params['z_distortion']
                si_x_rot = x_rotation if x_rotation else 0
                si_y_rot = y_rotation if y_rotation else 0
                si_z_rot = z_rotation if z_rotation else 0
            elif soft_iron_mode == 'soft-iron-random':
                if soft_iron_random_type_mode == 'si-random-collective':
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
                si_x_scale = 1.0
                si_y_scale = 1.0
                si_z_scale = 1.0
                si_x_rot = 0.0
                si_y_rot = 0.0
                si_z_rot = 0.0
            
            # Generate points mit Soft Iron Rotation
            data_points, actual_hi_offset, noise_points, point_ids = fibonacci_sphere(
                params['generation_mode'], params['noise_value'], params['sample_count'], 
                params['alpha'], seed_string, 
                xyz0=[hi_x_offset, hi_y_offset, hi_z_offset],
                xyz1=[si_x_scale, si_y_scale, si_z_scale],
                x_rot=si_x_rot, y_rot=si_y_rot, z_rot=si_z_rot,
                radius=flux_radius, maintain_density=maintain_density,
                axis_constraint_mode=axis_constraint_mode
            )
            noise_by_id = {int(pid): noise_points[idx] for idx, pid in enumerate(point_ids)}
            data_origin = 'generated'

        if point_ids is None:
            point_ids = np.arange(1, len(data_points) + 1, dtype=int)

        # Create mesh centered at origin (0,0,0) first
        center_offset = actual_hi_offset
        sphere_vertices, sphere_triangles, _ = create_sphere_mesh(xyz0=[0, 0, 0])
        
        # --- APPLY SOFT IRON TRANSFORMATION TO MESH VERTICES ---
        transformed_vertices = sphere_vertices.copy()
        if si_x_scale != 1.0 or si_y_scale != 1.0 or si_z_scale != 1.0 or \
           si_x_rot or si_y_rot or si_z_rot:
            transformed_vertices = apply_soft_iron_transformation(
                sphere_vertices, 
                si_x_scale, si_y_scale, si_z_scale,
                si_x_rot, si_y_rot, si_z_rot
            )
        
        # Skaliere Mesh mit Flussdichte-Radius (konsistent mit Punkten, vor HI-Offset)
        if flux_radius != 1.0:
            transformed_vertices = transformed_vertices * flux_radius

        # Addiere den Hard Iron Offset (in nT) zum transformierten Mesh
        transformed_vertices[:, 0] += center_offset[0]
        transformed_vertices[:, 1] += center_offset[1]
        transformed_vertices[:, 2] += center_offset[2]
        
        # Bestimme, ob das Mesh angezeigt werden soll und welche Farbe die Punkte haben
        show_mesh = False
        point_color = 'darkred'
        
        if trigger_id == 'submit-button' or (trigger_id == 'seed-data-storage' and not loaded_seed_string):
            show_mesh = True
            point_color = 'blue'

        point_amount_for_optimal = None
        if csv_metadata and csv_metadata.get('POINT_AMOUNT') is not None:
            try:
                point_amount_for_optimal = int(csv_metadata.get('POINT_AMOUNT'))
            except Exception:
                point_amount_for_optimal = None
        if point_amount_for_optimal is None:
            try:
                point_amount_for_optimal = int(params.get('sample_count') or 0)
            except Exception:
                point_amount_for_optimal = None

        optimal_points, optimal_ids = _load_optimal_points_for_ids(point_amount_for_optimal, point_ids.tolist())
        if optimal_points is not None and optimal_ids is not None and len(optimal_points) == len(optimal_ids):
            noise_offsets = np.array([noise_by_id.get(int(pid), np.zeros(3, dtype=float)) for pid in optimal_ids], dtype=float)
            optimal_points = optimal_points + noise_offsets

        uncalibrated_label = 'Unkalibrierte Punkte' if data_origin == 'loaded' else 'Erzeugte Punkte'
        calibrated_label = 'Kalibrierte Punkte'
        optimal_label = 'Optimale Punkte'
        
        # Build figure mit transformierten Mesh-Vertices
        # Berechne kalibrierte Punkte, wenn Kalibrierdaten vorhanden
        calibrated_points = None
        if trigger_id == 'seed-data-storage' and calibration_store_data is not None:
            try:
                import numpy as np_local
                H = np_local.array(calibration_store_data['matrix'], dtype=float)
                # Translationseintraege sind je nach Datenquelle in Tesla oder nT gespeichert.
                # Nur bei Tesla-Groessenordnungen (< 1) in nT umrechnen.
                if np_local.max(np_local.abs(H[:3, 3])) < 1.0:
                    H[:3, 3] = H[:3, 3] * 1e9
                ones = np_local.ones((len(data_points), 1))
                pts_h = np_local.hstack([data_points[:, :3], ones])
                cal = (H @ pts_h.T).T
                calibrated_points = cal[:, :3]
            except Exception:
                calibrated_points = None

        fig = build_figure_with_points(
            data_points, center_offset, transformed_vertices, sphere_triangles, 
            params['alpha'], params['noise_value'], si_x_scale, 
            si_y_scale, si_z_scale, ui_mesh_opacity, 
            show_mesh=show_mesh, point_color=point_color,
            calibrated_points=calibrated_points,
            show_origin=bool(show_origin if show_origin is not None else True),
            optimal_points=optimal_points,
            uncalibrated_ids=point_ids,
            calibrated_ids=point_ids if calibrated_points is not None else None,
            optimal_ids=optimal_ids,
            show_uncalibrated=True,
            show_calibrated=True,
            show_optimal=False,
            uncalibrated_label=uncalibrated_label,
            calibrated_label=calibrated_label,
            optimal_label=optimal_label,
        )
        fig = apply_axes_legend_scale(fig, bool(scale_axes_legend))
        fig = _apply_view_camera(fig, view_projection_mode, view_plane_mode)
        
        # Berechne die 4x4 Iron Error Matrix
        iron_error_matrix = create_iron_error_matrix(
            si_x_scale, si_y_scale, si_z_scale,
            si_x_rot, si_y_rot, si_z_rot,
            center_offset[0], center_offset[1], center_offset[2]
        )
        
        calibration_errors = None
        if dataset_source == 'exports':
            calibration_errors = _get_calibration_errors_for_seed(seed_string, dataset_subdir)

        # Build results display mit vollständigen Metadaten
        result_text = build_results_display(
            data_points, params['alpha'], params['generation_mode'], 
            params['noise_value'], center_offset, si_x_scale, 
            si_y_scale, si_z_scale, si_x_rot, si_y_rot, si_z_rot,
            iron_error_matrix=iron_error_matrix, sample_count=params['sample_count'],
            raw_metadata=csv_metadata,
            calibration_data=calibration_store_data if trigger_id == 'seed-data-storage' else None,
            calibration_errors=calibration_errors,
            keep_point_density=maintain_density,
            axis_constraint_mode=axis_constraint_mode
        )
        
        # Export data
        export_points_xyz = data_points[:, [0, 1, 2]]
        points_data = pd.DataFrame(export_points_xyz, columns=['X', 'Y', 'Z']).to_json(orient='split')
        seed_display = f"Seed: {seed_string}"
        
        # Speichere die tatsächlich verwendeten magnetischen Fehlerwerte + erweiterte Metadaten
        iron_error_matrix_list = iron_error_matrix.tolist()
        
        magnetic_errors = {
            'DISTRIBUTION_STYLE': params['generation_mode'],
            'NOISE': params['noise_value'],
            'POINT_AMOUNT': params['sample_count'],
            'ANGULAR_CONSTRAINT_DEG': params['alpha'],
            'KEEP_POINT_DENSITY': maintain_density,
            'AXIS_CONSTRAINT': axis_constraint_mode,
            'HI': [center_offset[0], center_offset[1], center_offset[2]],
            'HI_X_Y_Z_OFFSET': [center_offset[0], center_offset[1], center_offset[2]],
            'SI_DISTORTION': [si_x_scale, si_y_scale, si_z_scale],
            'SI_X_Y_Z_DISTORTION': [si_x_scale, si_y_scale, si_z_scale],
            'SI_ROTATION': [si_x_rot, si_y_rot, si_z_rot],
            'SI_ROTATION_DEG': [si_x_rot, si_y_rot, si_z_rot],
            'SI_X_Y_Z_ROTATION_DEG': [si_x_rot, si_y_rot, si_z_rot],
            'IRON_ERROR_MATRIX_RAD': iron_error_matrix_list
        }
        
        # Generiere einfachen Dateinamen
        export_filename = f"{int(params['alpha'])}_{current_time.strftime('%d-%m-%y')}_{current_time.strftime('%H-%M')}.csv"
        
        raw_points_json = {
            'uncalibrated': data_points.tolist(),
            'calibrated': calibrated_points.tolist() if calibrated_points is not None else None,
            'optimal': optimal_points.tolist() if optimal_points is not None else None,
            'uncalibrated_ids': point_ids.tolist() if point_ids is not None else None,
            'optimal_ids': optimal_ids.tolist() if optimal_ids is not None else None,
            'origin': data_origin,
        }
        return fig, result_text, points_data, display_time_str, None, None, export_filename, magnetic_errors, raw_points_json

    # E. Callback für den Export der Punktedaten
    @app.callback(
        [Output("download-csv", "data"),
         Output('export-status', 'children'),
         Output('export-success-trigger', 'data')],
        [Input("export-dataset", "n_clicks")],
        [State('point-data-storage', 'data'),
         State('export-filename-store', 'data'),
         State('magnetic-error-values-store', 'data'),
         State('batch-mode-toggle', 'value'),
         State('batch-step-input', 'value'),
         State('dataset-count-input', 'value'),
         State('custom-filename-input', 'value'),
         State('sample-duration-dropdown', 'value'),
         State('angular-constraint-input', 'value'),
         State('generation-mode', 'value'),
         State('noise-input', 'value'),
         State('x-offset-input', 'value'), State('y-offset-input', 'value'), State('z-offset-input', 'value'),
         State('x-distortion-input', 'value'), State('y-distortion-input', 'value'), State('z-distortion-input', 'value'),
         State('x-rotation', 'value'), State('y-rotation', 'value'), State('z-rotation', 'value'),
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
         State('flux-density-input', 'value'),
            State('density-mode-toggle', 'value'),
            State('axis-constraint-mode', 'value')]
    )
    def export_dataset(n_clicks, json_data, export_filename, magnetic_errors,
                       batch_values, batch_step, dataset_count, custom_filename,
                       ui_sample_count, ui_angular_constraint,
                       ui_generation_mode, ui_noise,
                       ui_x_offset, ui_y_offset, ui_z_offset,
                       ui_x_distortion, ui_y_distortion, ui_z_distortion,
                       ui_x_rotation, ui_y_rotation, ui_z_rotation,
                       hard_iron_mode, hard_iron_random_type,
                       hi_rand_min_coll, hi_rand_max_coll,
                       hi_rand_min_x, hi_rand_max_x, hi_rand_min_y, hi_rand_max_y,
                       hi_rand_min_z, hi_rand_max_z,
                       soft_iron_mode, soft_iron_random_type,
                       si_dist_min_coll, si_dist_max_coll,
                       si_rot_min_coll, si_rot_max_coll,
                       si_dist_min_x, si_dist_max_x, si_dist_min_y, si_dist_max_y,
                       si_dist_min_z, si_dist_max_z,
                       si_rot_min_x, si_rot_max_x, si_rot_min_y, si_rot_max_y,
                       si_rot_min_z, si_rot_max_z,
                       flux_density_nT, density_mode_values, axis_constraint_mode):
        if n_clicks == 0:
            return None, "", no_update

        is_batch = 'batch' in (batch_values or [])
        n_sets = max(1, int(dataset_count)) if dataset_count else 1
        generation_mode = ui_generation_mode or 'optimal'
        noise_value = float(ui_noise) if ui_noise is not None else DEFAULT_NOISE
        sample_count = int(ui_sample_count) if ui_sample_count is not None else DEFAULT_SAMPLES
        current_time = datetime.now()
        flux_radius = float(flux_density_nT) if flux_density_nT else DEFAULT_FLUX_DENSITY_NT
        maintain_density = 'density' in (density_mode_values or [])
        axis_constraint_mode = _normalize_axis_constraint_mode(axis_constraint_mode)

        # Nur beim echten Single-Export vorhandene Plot-Einstellungen aus dem aktuellen Zustand übernehmen.
        use_plot_state = bool(magnetic_errors) and (not is_batch) and (n_sets == 1)
        if use_plot_state:
            maintain_density = bool(magnetic_errors.get('KEEP_POINT_DENSITY', maintain_density))
            axis_constraint_mode = _normalize_axis_constraint_mode(magnetic_errors.get('AXIS_CONSTRAINT', axis_constraint_mode))

        custom_text= str(custom_filename).strip() if custom_filename else "-"
        if not custom_text:
            custom_text = "-"

        # Bestimme die Liste der Winkel
        if is_batch and batch_step and int(batch_step) > 0:
            step = int(batch_step)
            angles = list(range(step, 91, step))
            if not angles:
                return None, html.Span("✗ Ungültiger Gradschritt", style={'color': 'red'}), no_update
        else:
            # Einzelner Winkel aus UI oder magnetic_errors
            if magnetic_errors:
                angles = [magnetic_errors.get('ANGULAR_CONSTRAINT_DEG', DEFAULT_ALPHA)]
            else:
                angles = [int(ui_angular_constraint) if ui_angular_constraint else DEFAULT_ALPHA]

        # Generiere n_sets Datensätze pro Winkel
        if n_sets > 1 or is_batch:
            exported_files = []
            try:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                for alpha in angles:
                    for i in range(n_sets):
                        hi_x, hi_y, hi_z = _generate_hi_params(
                            hard_iron_mode, hard_iron_random_type,
                            ui_x_offset, ui_y_offset, ui_z_offset,
                            hi_rand_min_coll, hi_rand_max_coll,
                            hi_rand_min_x, hi_rand_max_x, hi_rand_min_y, hi_rand_max_y,
                            hi_rand_min_z, hi_rand_max_z)

                        si_x_s, si_y_s, si_z_s, si_x_r, si_y_r, si_z_r = _generate_si_params(
                            soft_iron_mode, soft_iron_random_type,
                            ui_x_distortion, ui_y_distortion, ui_z_distortion,
                            ui_x_rotation, ui_y_rotation, ui_z_rotation,
                            si_dist_min_coll, si_dist_max_coll, si_rot_min_coll, si_rot_max_coll,
                            si_dist_min_x, si_dist_max_x, si_dist_min_y, si_dist_max_y, si_dist_min_z, si_dist_max_z,
                            si_rot_min_x, si_rot_max_x, si_rot_min_y, si_rot_max_y, si_rot_min_z, si_rot_max_z)

                        data_points, _offset, noise_points, point_ids = fibonacci_sphere(
                            generation_mode, noise_value, sample_count, alpha, "",
                            xyz0=[hi_x, hi_y, hi_z],
                            xyz1=[si_x_s, si_y_s, si_z_s],
                            x_rot=si_x_r, y_rot=si_y_r, z_rot=si_z_r,
                            radius=flux_radius, maintain_density=maintain_density,
                            axis_constraint_mode=axis_constraint_mode)

                        iem = create_iron_error_matrix(
                            si_x_s, si_y_s, si_z_s, si_x_r, si_y_r, si_z_r,
                            hi_x, hi_y, hi_z)

                        filename = _build_filename(
                            alpha, generation_mode, sample_count, custom_text, i + 1, n_sets, current_time,
                            axis_constraint_mode=axis_constraint_mode
                        )
                        df = pd.DataFrame(data_points, columns=['X', 'Y', 'Z'])
                        df['X_noise'] = noise_points[:, 0]
                        df['Y_noise'] = noise_points[:, 1]
                        df['Z_noise'] = noise_points[:, 2]
                        df['ID'] = point_ids

                        _write_single_csv(
                            os.path.join(OUTPUT_DIR, filename), df,
                            distribution_style=generation_mode,
                            noise=noise_value, point_amount=sample_count,
                            angular_constraint_deg=alpha,
                            hi=[hi_x, hi_y, hi_z],
                            si_dist=[si_x_s, si_y_s, si_z_s],
                            si_rot=[si_x_r, si_y_r, si_z_r],
                            iron_error_matrix=iem.tolist(),
                            true_magnetic_field_strength=flux_radius,
                            keep_point_density=maintain_density,
                            axis_constraint=axis_constraint_mode,
                            export_timestamp=current_time)

                        exported_files.append(filename)

                total = len(exported_files)
                if is_batch:
                    angles_str = ', '.join([f"{a}°" for a in angles])
                    return None, f"✓ Export: {total} Dateien ({n_sets}x bei {angles_str})", n_clicks
                else:
                    return None, f"✓ Export: {total} Dateien erstellt", n_clicks

            except Exception as e:
                return None, html.Span(f"✗ Fehler: {e}", style={'color': 'red'}), no_update

        else:
            # --- SINGLE EXPORT (bestehende Logik: 1 Datensatz, kein Batch) ---
            if json_data is not None:
                df = pd.read_json(json_data, orient='split')

                if magnetic_errors:
                    distribution_style = magnetic_errors.get('DISTRIBUTION_STYLE', 'evenly')
                    noise = magnetic_errors.get('NOISE', 0.0)
                    point_amount = magnetic_errors.get('POINT_AMOUNT', 1000)
                    angular_constraint_deg = magnetic_errors.get('ANGULAR_CONSTRAINT_DEG', 90.0)
                    hi = magnetic_errors.get('HI', [0, 0, 0])
                    si_dist = magnetic_errors.get('SI_DISTORTION', [1.0, 1.0, 1.0])
                    si_rot = magnetic_errors.get('SI_ROTATION', [0.0, 0.0, 0.0])
                    iron_error_matrix = magnetic_errors.get('IRON_ERROR_MATRIX_RAD', None)
                else:
                    distribution_style = 'evenly'
                    noise = 0.0
                    point_amount = 1000
                    angular_constraint_deg = 90.0
                    hi = [0, 0, 0]
                    si_dist = [1.0, 1.0, 1.0]
                    si_rot = [0.0, 0.0, 0.0]
                    iron_error_matrix = None

                filename = _build_filename(
                    angular_constraint_deg, distribution_style, point_amount, custom_text, 1, n_sets, current_time,
                    axis_constraint_mode=axis_constraint_mode
                )
                full_path = os.path.join(OUTPUT_DIR, filename)

                try:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    _write_single_csv(full_path, df, distribution_style, noise, point_amount,
                                      angular_constraint_deg, hi, si_dist, si_rot, iron_error_matrix,
                                      true_magnetic_field_strength=flux_radius,
                                      keep_point_density=maintain_density,
                                      axis_constraint=axis_constraint_mode,
                                      export_timestamp=current_time)
                    return None, f"✓ Export erfolgreich: {filename}", n_clicks

                except Exception as e:
                    return None, html.Span(f"✗ Fehler: {e}", style={'color': 'red'}), no_update

        return None, "", no_update

    @app.callback(
        Output('download-plot-html', 'data'),
        Input('export-plot-html', 'n_clicks'),
        [State('sphere-plot', 'figure'),
         State('current-seed-store', 'data'),
         State('export-filename-store', 'data')],
        prevent_initial_call=True
    )
    def export_plot_html(n_clicks, current_fig, current_seed, export_filename):
        if not n_clicks or not current_fig:
            return no_update

        fig = go.Figure(current_fig)
        html_string = _build_standalone_plot_html(fig)
        filename = _build_plot_html_filename(current_seed, export_filename)
        os.makedirs(PLOT_EXPORT_DIR, exist_ok=True)
        export_path = os.path.join(PLOT_EXPORT_DIR, filename)

        with open(export_path, 'w', encoding='utf-8') as export_file:
            export_file.write(html_string)

        return dcc.send_file(export_path)

    # F. Callback: Simuliert-Filter nur bei Simuliert anzeigen
    @app.callback(
        Output('simulated-seed-filter-details', 'style'),
        Input('dataset-source-store', 'data')
    )
    def toggle_simulated_filter_visibility(dataset_source):
        if dataset_source == 'exports':
            return {'marginBottom': '8px', 'display': 'block'}
        return {'marginBottom': '8px', 'display': 'none'}

    @app.callback(
        [Output('sim-filter-iron-error', 'value'),
         Output('sim-filter-point-amount', 'value'),
         Output('sim-filter-keep-density', 'value'),
         Output('sim-filter-axis-constraint', 'value'),
         Output('sim-filter-angle-range', 'value')],
        Input('sim-filter-reset-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def reset_simulated_filters(n_clicks):
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update
        return None, None, None, None, [5, 90]

    @app.callback(
        Output('sim-filter-applied-store', 'data'),
        [Input('sim-filter-apply-button', 'n_clicks'),
         Input('sim-filter-reset-button', 'n_clicks')],
        [State('sim-filter-iron-error', 'value'),
         State('sim-filter-point-amount', 'value'),
         State('sim-filter-keep-density', 'value'),
         State('sim-filter-axis-constraint', 'value'),
         State('sim-filter-angle-range', 'value')],
        prevent_initial_call=True
    )
    def apply_simulated_filters(_apply_clicks, _reset_clicks, iron_filter, point_filter,
                                keep_density_filter, axis_constraint_filter, angle_range):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'sim-filter-reset-button':
            return {
                'iron': None,
                'point': None,
                'density': None,
                'axis': None,
                'angle': [5, 90],
            }

        return {
            'iron': iron_filter,
            'point': point_filter,
            'density': keep_density_filter,
            'axis': _normalize_axis_constraint_mode(axis_constraint_filter) if axis_constraint_filter else None,
            'angle': angle_range if isinstance(angle_range, (list, tuple)) and len(angle_range) == 2 else [5, 90],
        }

    # F1. Callback: Datensatzliste je nach Quelle + Simuliert-Filter aktualisieren
    @app.callback(
        [Output('import-seed', 'options'),
         Output('import-seed', 'value')],
        [Input('dataset-source-store', 'data'),
         Input('dataset-subdir-store', 'data'),
         Input('export-success-trigger', 'data'),
         Input('sim-filter-applied-store', 'data')],
        [State('import-seed', 'value')],
        prevent_initial_call=True
    )
    def update_seed_dropdown(dataset_source, dataset_subdir, _trigger_value, applied_filters, current_seed):
        applied = applied_filters or {
            'iron': None,
            'point': None,
            'density': None,
            'axis': None,
            'angle': [5, 90],
        }
        source_dir = resolve_dataset_directory(dataset_source, dataset_subdir)
        if dataset_source == 'reallife':
            options = get_seeds_from_dir(source_dir)
        else:
            base_options = get_seeds_from_dir(source_dir)
            options = _filter_simulated_seed_options(
                applied.get('iron'),
                applied.get('point'),
                applied.get('density'),
                applied.get('axis'),
                applied.get('angle'),
                dataset_subdir=dataset_subdir,
                base_options=base_options
            )

        valid_values = {
            opt.get('value')
            for opt in options
            if not opt.get('disabled') and opt.get('value') not in (None, '', 'none')
        }
        selected_value = current_seed if current_seed in valid_values else None
        return options, selected_value

    # F2. Callback: Quelle wechseln (Simuliert <-> Echtdaten)
    STYLE_SOURCE_ACTIVE = {
        'flex': '1', 'backgroundColor': '#007bff', 'color': 'white',
        'border': '2px solid #007bff', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }
    STYLE_SOURCE_INACTIVE = {
        'flex': '1', 'backgroundColor': '#f8f9fa', 'color': '#6c757d',
        'border': '2px solid #adb5bd', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }

    @app.callback(
        [Output('dataset-source-store', 'data'),
         Output('btn-source-simulated', 'style'),
         Output('btn-source-reallife', 'style')],
        [Input('btn-source-simulated', 'n_clicks'),
         Input('btn-source-reallife', 'n_clicks')],
        prevent_initial_call=True
    )
    def switch_dataset_source(n_sim, n_real):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        if trigger_id == 'btn-source-simulated':
            source = 'exports'
        else:
            source = 'reallife'
        sim_style = STYLE_SOURCE_ACTIVE if source == 'exports' else STYLE_SOURCE_INACTIVE
        real_style = STYLE_SOURCE_ACTIVE if source == 'reallife' else STYLE_SOURCE_INACTIVE
        return source, sim_style, real_style

    @app.callback(
        [Output('dataset-folder-dropdown', 'options'),
         Output('dataset-folder-dropdown', 'value')],
        Input('dataset-source-store', 'data'),
        State('dataset-folder-dropdown', 'value')
    )
    def update_dataset_folder_dropdown(dataset_source, current_subdir):
        options = get_dataset_subdir_options(dataset_source)
        valid_values = {opt.get('value') for opt in options}
        selected_value = current_subdir if current_subdir in valid_values else get_default_dataset_subdir(dataset_source)
        return options, selected_value

    @app.callback(
        Output('dataset-subdir-store', 'data'),
        Input('dataset-folder-dropdown', 'value')
    )
    def update_dataset_subdir(selected_subdir):
        return selected_subdir or ''

    @app.callback(
        Output('calibration-dir-dropdown', 'value'),
        Input('dataset-folder-dropdown', 'value'),
        [State('calibration-dir-dropdown', 'options'),
         State('calibration-dir-dropdown', 'value')],
        prevent_initial_call=True
    )
    def auto_select_calibration_dir(selected_subdir, calibration_options, current_calibration_dir):
        if not selected_subdir or not calibration_options:
            return no_update

        target_name = os.path.basename(str(selected_subdir).replace('\\', '/').rstrip('/')).strip().lower()
        if not target_name:
            return no_update

        for option in calibration_options:
            option_value = option.get('value')
            option_label = str(option.get('label', '')).strip().lower()
            option_leaf = os.path.basename(str(option_value).replace('\\', '/').rstrip('/')).strip().lower()
            if target_name in (option_label, option_leaf):
                if option_value == current_calibration_dir:
                    return no_update
                return option_value

        return no_update

    # F2b. Callback: Kalibrierordner-Store aktualisieren
    @app.callback(
        Output('calibration-dir-store', 'data'),
        Input('calibration-dir-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_calibration_dir(selected_dir):
        return selected_dir if selected_dir else CALIBRATION_DIR

    STYLE_BTN_ACTIVE_RED_F2 = {
        'flex': '1', 'backgroundColor': '#dc3545', 'color': 'white',
        'border': '2px solid #dc3545', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }
    STYLE_BTN_ACTIVE_GREEN_F2 = {
        'flex': '1', 'backgroundColor': '#28a745', 'color': 'white',
        'border': '2px solid #28a745', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }
    STYLE_BTN_INACTIVE_BLUE_F2 = {
        'flex': '1', 'backgroundColor': '#dbeafe', 'color': '#0d6efd',
        'border': '2px solid #0d6efd', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }

    # F2. Callback: Datensatz laden
    @app.callback(
        [Output('seed-data-storage', 'data'),
         Output('loaded-seed-string-store', 'data'),
         Output('seed-load-status', 'children'),
         Output('calibration-data-store', 'data'),
         Output('show-uncalibrated-store', 'data'),
         Output('show-calibrated-store', 'data'),
         Output('show-optimal-store', 'data'),
         Output('btn-toggle-uncalibrated', 'style'),
         Output('btn-toggle-calibrated', 'style'),
         Output('btn-toggle-optimal', 'style')],
        [Input('load-dataset-button', 'n_clicks')],
        [State('import-seed', 'value'),
         State('dataset-source-store', 'data'),
         State('dataset-subdir-store', 'data'),
         State('calibration-dir-store', 'data')],
        prevent_initial_call=True
    )
    def load_dataset(n_clicks, selected_seed, dataset_source, dataset_subdir, calibration_dir):
        """
        Lädt ein Datensatz aus der CSV-Datei und speichert die Metadaten im Store.
        Dies triggeriert dann automatisch den update_graph Callback.
        """
        if n_clicks == 0 or selected_seed is None:
            return no_update, no_update, "", no_update, no_update, no_update, no_update, no_update, no_update, no_update

        source_dir = resolve_dataset_directory(dataset_source, dataset_subdir)
        data_points, metadata, load_error = load_csv_data_by_seed(selected_seed, directory=source_dir)
        
        if load_error:
            return no_update, no_update, html.Span(f"✗ {load_error}", style={'color': 'red', 'fontSize': '0.8em'}), no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        if data_points is None or metadata is None:
            return no_update, no_update, html.Span(f"✗ Fehler beim Laden: Datei konnte nicht korrekt gelesen werden", style={'color': 'red', 'fontSize': '0.8em'}), no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        # Kalibrierdaten laden (nur für simulierte Daten relevant, kein Fehler wenn nicht vorhanden)
        calib_data, calib_error = load_calibration_data(selected_seed, calib_dir=calibration_dir)
        
        status_msg = html.Span(f"✓ Datensatz geladen: {selected_seed}", style={'color': 'green', 'fontSize': '0.8em'})
        return metadata, selected_seed, status_msg, calib_data, True, True, False, STYLE_BTN_ACTIVE_RED_F2, STYLE_BTN_ACTIVE_GREEN_F2, STYLE_BTN_INACTIVE_BLUE_F2

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
        ohne die ganze Figure neu zu bauen.
        """
        if not current_fig:
            return no_update
        
        fig_copy = go.Figure(current_fig)
        
        for trace in fig_copy.data:
            if trace.type == 'mesh3d':
                trace.opacity = new_opacity
                break
        
        return fig_copy

    # G1. Callback: Kameraansicht (Perspektive/Isometrisch + xz/yz/xy)
    @app.callback(
        [Output('view-projection-store', 'data'),
         Output('view-plane-store', 'data'),
         Output('btn-view-projection-toggle', 'children'),
         Output('btn-view-projection-toggle', 'style'),
         Output('btn-view-plane-iso', 'style'),
         Output('btn-view-plane-xz', 'style'),
         Output('btn-view-plane-yz', 'style'),
         Output('btn-view-plane-xy', 'style'),
         Output('sphere-plot', 'figure', allow_duplicate=True)],
        [Input('btn-view-projection-toggle', 'n_clicks'),
         Input('btn-view-plane-iso', 'n_clicks'),
         Input('btn-view-plane-xz', 'n_clicks'),
         Input('btn-view-plane-yz', 'n_clicks'),
         Input('btn-view-plane-xy', 'n_clicks')],
        [State('view-projection-store', 'data'),
         State('view-plane-store', 'data'),
         State('sphere-plot', 'figure')],
        prevent_initial_call=True
    )
    def update_view_controls(_n_proj, _n_iso, _n_xz, _n_yz, _n_xy, projection_mode, plane_mode, current_fig):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        projection_mode = projection_mode or 'perspective'
        plane_mode = plane_mode or 'iso'

        if trigger_id == 'btn-view-projection-toggle':
            projection_mode = 'isometric' if projection_mode == 'perspective' else 'perspective'
        elif trigger_id == 'btn-view-plane-iso':
            plane_mode = 'iso'
        elif trigger_id == 'btn-view-plane-xz':
            plane_mode = 'xz'
        elif trigger_id == 'btn-view-plane-yz':
            plane_mode = 'yz'
        elif trigger_id == 'btn-view-plane-xy':
            plane_mode = 'xy'

        projection_label = 'I' if projection_mode == 'isometric' else 'P'
        projection_style = {
            'width': '36px',
            'height': '36px',
            'borderRadius': '50%',
            'border': '1px solid #198754' if projection_mode == 'isometric' else '1px solid #007bff',
            'backgroundColor': '#198754' if projection_mode == 'isometric' else '#007bff',
            'color': 'white',
            'fontWeight': 'bold',
            'fontSize': '0.85em',
            'cursor': 'pointer',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.18)',
        }

        plane_active = {
            'padding': '4px 10px',
            'border': '1px solid #343a40',
            'borderRadius': '4px',
            'fontSize': '0.78em',
            'color': 'white',
            'backgroundColor': '#343a40',
            'cursor': 'pointer',
        }
        plane_inactive = {
            'padding': '4px 10px',
            'border': '1px solid #ced4da',
            'borderRadius': '4px',
            'fontSize': '0.78em',
            'color': '#495057',
            'backgroundColor': '#f8f9fa',
            'cursor': 'pointer',
        }

        style_iso = plane_active if plane_mode == 'iso' else plane_inactive
        style_xz = plane_active if plane_mode == 'xz' else plane_inactive
        style_yz = plane_active if plane_mode == 'yz' else plane_inactive
        style_xy = plane_active if plane_mode == 'xy' else plane_inactive

        if not current_fig:
            return projection_mode, plane_mode, projection_label, projection_style, style_iso, style_xz, style_yz, style_xy, no_update

        fig_copy = go.Figure(current_fig)
        fig_copy = _apply_view_camera(fig_copy, projection_mode, plane_mode)
        return projection_mode, plane_mode, projection_label, projection_style, style_iso, style_xz, style_yz, style_xy, fig_copy

    @app.callback(
        [Output('show-origin-store', 'data'),
         Output('btn-view-origin-toggle', 'style'),
         Output('sphere-plot', 'figure', allow_duplicate=True)],
        [Input('btn-view-origin-toggle', 'n_clicks')],
        [State('show-origin-store', 'data'),
         State('sphere-plot', 'figure')],
        prevent_initial_call=True
    )
    def toggle_origin_point(_n_clicks, show_origin, current_fig):
        show_origin = not bool(show_origin if show_origin is not None else True)

        btn_active = {
            'width': '30px',
            'height': '30px',
            'borderRadius': '50%',
            'border': '1px solid #343a40',
            'fontSize': '0.78em',
            'color': 'white',
            'backgroundColor': '#343a40',
            'fontWeight': 'bold',
            'cursor': 'pointer',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.18)',
        }
        btn_inactive = {
            'width': '30px',
            'height': '30px',
            'borderRadius': '50%',
            'border': '1px solid #6c757d',
            'fontSize': '0.78em',
            'color': 'white',
            'backgroundColor': '#6c757d',
            'fontWeight': 'bold',
            'cursor': 'pointer',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.18)',
        }

        if not current_fig:
            return show_origin, (btn_active if show_origin else btn_inactive), no_update

        fig_copy = go.Figure(current_fig)
        for trace in fig_copy.data:
            trace_meta = getattr(trace, 'meta', None)
            trace_name = getattr(trace, 'name', '')
            if trace_meta == 'origin-point' or trace_name == 'Ursprung (0,0,0)':
                trace.visible = True if show_origin else False
                trace.showlegend = False
            elif trace_meta == 'origin-legend':
                trace.visible = True
                trace.showlegend = True if show_origin else False

        return show_origin, (btn_active if show_origin else btn_inactive), fig_copy

    @app.callback(
        [Output('scale-axes-legend-store', 'data'),
         Output('btn-view-scale-toggle', 'style'),
         Output('sphere-plot', 'figure', allow_duplicate=True)],
        [Input('btn-view-scale-toggle', 'n_clicks')],
        [State('scale-axes-legend-store', 'data'),
         State('sphere-plot', 'figure')],
        prevent_initial_call=True
    )
    def toggle_axes_legend_scale(_n_clicks, scale_enabled, current_fig):
        scale_enabled = not bool(scale_enabled)

        btn_active = {
            'width': '30px',
            'height': '30px',
            'borderRadius': '50%',
            'border': '1px solid #343a40',
            'fontSize': '0.78em',
            'color': 'white',
            'backgroundColor': '#343a40',
            'fontWeight': 'bold',
            'cursor': 'pointer',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.18)',
        }
        btn_inactive = {
            'width': '30px',
            'height': '30px',
            'borderRadius': '50%',
            'border': '1px solid #6c757d',
            'fontSize': '0.78em',
            'color': 'white',
            'backgroundColor': '#6c757d',
            'fontWeight': 'bold',
            'cursor': 'pointer',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.18)',
        }

        if not current_fig:
            return scale_enabled, (btn_active if scale_enabled else btn_inactive), no_update

        fig_copy = go.Figure(current_fig)
        fig_copy = apply_axes_legend_scale(fig_copy, scale_enabled)
        return scale_enabled, (btn_active if scale_enabled else btn_inactive), fig_copy

    # H. Callback: Toggle-Buttons für Datenpunkte (unkalibriert / kalibriert)
    STYLE_BTN_ACTIVE_RED = {
        'flex': '1', 'backgroundColor': '#dc3545', 'color': 'white',
        'border': '2px solid #dc3545', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }
    STYLE_BTN_INACTIVE_RED = {
        'flex': '1', 'backgroundColor': '#f8d7da', 'color': '#dc3545',
        'border': '2px solid #dc3545', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }
    STYLE_BTN_ACTIVE_GREEN = {
        'flex': '1', 'backgroundColor': '#28a745', 'color': 'white',
        'border': '2px solid #28a745', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }
    STYLE_BTN_INACTIVE_GREEN = {
        'flex': '1', 'backgroundColor': '#d4edda', 'color': '#28a745',
        'border': '2px solid #28a745', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }
    STYLE_BTN_ACTIVE_BLUE = {
        'flex': '1', 'backgroundColor': '#0d6efd', 'color': 'white',
        'border': '2px solid #0d6efd', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }
    STYLE_BTN_INACTIVE_BLUE = {
        'flex': '1', 'backgroundColor': '#dbeafe', 'color': '#0d6efd',
        'border': '2px solid #0d6efd', 'borderRadius': '4px',
        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
    }

    @app.callback(
        [Output('sphere-plot', 'figure', allow_duplicate=True),
         Output('show-uncalibrated-store', 'data', allow_duplicate=True),
         Output('show-calibrated-store', 'data', allow_duplicate=True),
         Output('show-optimal-store', 'data', allow_duplicate=True),
         Output('btn-toggle-uncalibrated', 'style', allow_duplicate=True),
         Output('btn-toggle-calibrated', 'style', allow_duplicate=True),
         Output('btn-toggle-optimal', 'style', allow_duplicate=True)],
        [Input('btn-toggle-uncalibrated', 'n_clicks'),
         Input('btn-toggle-calibrated', 'n_clicks'),
         Input('btn-toggle-optimal', 'n_clicks')],
        [State('show-uncalibrated-store', 'data'),
         State('show-calibrated-store', 'data'),
         State('show-optimal-store', 'data'),
         State('sphere-plot', 'figure'),
         State('raw-points-store', 'data')],
        prevent_initial_call=True
    )
    def toggle_point_visibility(n_uncal, n_cal, n_opt, show_uncal, show_cal, show_optimal, current_fig, raw_points):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        if trigger_id == 'btn-toggle-uncalibrated':
            show_uncal = not show_uncal
        elif trigger_id == 'btn-toggle-calibrated':
            show_cal = not show_cal
        elif trigger_id == 'btn-toggle-optimal':
            show_optimal = not bool(show_optimal)

        # Button-Stile aktualisieren
        uncal_style = STYLE_BTN_ACTIVE_RED if show_uncal else STYLE_BTN_INACTIVE_RED
        cal_style = STYLE_BTN_ACTIVE_GREEN if show_cal else STYLE_BTN_INACTIVE_GREEN
        opt_style = STYLE_BTN_ACTIVE_BLUE if show_optimal else STYLE_BTN_INACTIVE_BLUE

        if not current_fig:
            return no_update, show_uncal, show_cal, show_optimal, uncal_style, cal_style, opt_style

        # Trace-Sichtbarkeit direkt im Dict setzen (zuverlässiger als go.Figure)
        for trace in current_fig['data']:
            meta = trace.get('meta', '') or ''
            name = trace.get('name', '') or ''

            if meta in ('uncal-legend', 'uncal-points') or name in ('Erzeugte Punkte', 'Unkalibrierte Punkte') or 'Unkalibriert' in name:
                trace['visible'] = show_uncal
            elif meta in ('cal-legend', 'cal-points') or name == 'Kalibrierte Punkte' or 'Kalibriert' in name:
                trace['visible'] = show_cal
            elif meta in ('optimal-legend', 'optimal-points') or name == 'Optimale Punkte':
                trace['visible'] = show_optimal

        # Achsenbereiche aus gespeicherten Rohdaten berechnen (nicht aus Traces extrahieren)
        visible_arrays = []
        if raw_points:
            if show_uncal and raw_points.get('uncalibrated'):
                visible_arrays.append(np.array(raw_points['uncalibrated'], dtype=float))
            if show_cal and raw_points.get('calibrated'):
                visible_arrays.append(np.array(raw_points['calibrated'], dtype=float))
            if show_optimal and raw_points.get('optimal'):
                visible_arrays.append(np.array(raw_points['optimal'], dtype=float))

        if visible_arrays:
            try:
                all_vis = np.vstack(visible_arrays)
                mn_x, mx_x = np.min(all_vis[:, 0]), np.max(all_vis[:, 0])
                mn_y, mx_y = np.min(all_vis[:, 1]), np.max(all_vis[:, 1])
                mn_z, mx_z = np.min(all_vis[:, 2]), np.max(all_vis[:, 2])
                half = max(mx_x - mn_x, mx_y - mn_y, mx_z - mn_z) * 0.525
                cx, cy, cz = (mn_x + mx_x) / 2, (mn_y + mx_y) / 2, (mn_z + mx_z) / 2
                scene = current_fig['layout'].get('scene', {})
                scene.setdefault('xaxis', {})['range'] = [cx - half, cx + half]
                scene.setdefault('yaxis', {})['range'] = [cy - half, cy + half]
                scene.setdefault('zaxis', {})['range'] = [cz - half, cz + half]
                current_fig['layout']['scene'] = scene
                # uirevision ändern, damit Plotly die neuen Achsenbereiche übernimmt
                current_fig['layout']['uirevision'] = datetime.now().isoformat()
            except Exception:
                pass

        return current_fig, show_uncal, show_cal, show_optimal, uncal_style, cal_style, opt_style

