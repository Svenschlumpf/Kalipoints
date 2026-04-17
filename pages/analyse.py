import os
import re

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html

from components.styles import BUTTON_STYLE_INLINE, SECTION_STYLE, SIDEBAR_STYLE, TAB_STYLE, TAB_SELECTED_STYLE
from utils.build_results import (
    build_results_csv,
    list_export_subdirs,
    list_results_files,
    results_filename_for_subdir,
)
from utils.calculate_calibration_errors import calculate_and_write_radius_errors, calculate_axis_error


RESULTS_DIR = os.path.join("datasets", "4-calibrated_exports_for_analysis")
ANGLE_ORDER = [f"± {angle}°" for angle in range(90, 0, -5)]
DEFAULT_TITLES = {
    "x": "x-Achsenwerte",
    "y": "y-Achsenwerte",
    "z": "z-Achsenwerte",
    "radius": "Radiuswerte",
    "angle_azimuth": "Azimutfehler (xy-Ebene)",
    "angle_zenith": "Zenitfehler (z-Richtung)",
}

IRON_ERROR_CONFIG = {
    "no_error": {"label": "Fehlerfrei", "fill": "#000000"},
    "hi_only": {"label": "HI", "fill": "#2e7d32"},
    "si_dist_only": {"label": "SIV", "fill": "#1d4ed8"},
    "hi_si_dist": {"label": "HI + SIV", "fill": "#14b8a6"},
    "hi_si_rot": {"label": "HI + SI Rotation", "fill": "#c99700"},
    "si_dist_rot": {"label": "SIV + SIR", "fill": "#facc15"},
    "hi_si_dist_rot": {"label": "HI + SIV + SIR", "fill": "#6c757d"},
}

POINT_AMOUNT_CONFIG = {
    100: {"label": "100", "width": 1.5},
    1000: {"label": "1000", "width": 2.5},
    10000: {"label": "10000", "width": 4.0},
}

DISPLAY_MODE_CONFIG = {
    "hard_iron": {
        "label": "Hard Iron Fehler",
        "line_color": "#d62828",
        "columns": {
            "x": ("HI-x-True", "HI-x-Meassured"),
            "y": ("HI-y-True", "HI-y-Meassured"),
            "z": ("HI-z-True", "HI-z-Meassured"),
        },
    },
    "soft_iron_distortion": {
        "label": "Soft Iron Verzerrung",
        "line_color": "#1d4ed8",
        "columns": {
            "x": ("SI-x-Faktor-True", "SI-x-Faktor-Meassured"),
            "y": ("SI-y-Faktor-True", "SI-y-Faktor-Meassured"),
            "z": ("SI-z-Faktor-True", "SI-z-Faktor-Meassured"),
        },
    },
}

IRON_FILTER_ORDER = [
    "no_error",
    "hi_only",
    "si_dist_only",
    "hi_si_dist",
    "si_dist_rot",
    "hi_si_dist_rot",
]
POINT_FILTER_ORDER = [100, 1000, 10000]
DENSITY_FILTER_ORDER = [True, False]
AXIS_CONSTRAINT_FILTER_ORDER = ["pitch_roll", "pitch_only"]
INDICATOR_LABELS = {
    "boxplot": "Box Plot",
    "points": "Punkte",
    "boxpoints": "Box Plot + Punkte",
}

BOX_FILL_OPACITY = 0.4
BOX_FILL_OPACITY_OVERRIDES = {
    # 0.4 Transparenz => 0.6 Opazitaet
    "no_error": 0.6,
    # 0.8 Transparenz => 0.2 Opazitaet
    "hi_si_dist_rot": 0.2,
}
BOX_BORDER_WIDTH = 2.3
LEGEND_BORDER_WIDTH = 12
ANALYSIS_TITLE_FONT_SIZE = 34
ANALYSIS_AXIS_TITLE_FONT_SIZE = 28
ANALYSIS_TICK_FONT_SIZE = 24
ANALYSIS_LEGEND_FONT_SIZE = 26
TRACE_BORDER_COLORS = [
    "#722F37",  # Weinrot
    "#FF8C00",  # Orange
    "#D00000",  # Rot
    "#FF4FA3",  # Pink
    "#8A2BE2",  # Violett
    "#0EA5E9",  # Blau
    "#16A34A",  # Gruen
]


def _base_toggle_button_style(active=False, accent="#495057"):
    if active:
        return {
            "backgroundColor": accent,
            "color": "white",
            "border": f"1px solid {accent}",
            "borderRadius": "6px",
            "padding": "6px 10px",
            "fontSize": "0.82em",
            "cursor": "pointer",
            "margin": "3px",
        }
    return {
        "backgroundColor": "#f8f9fa",
        "color": "#495057",
        "border": "1px solid #adb5bd",
        "borderRadius": "6px",
        "padding": "6px 10px",
        "fontSize": "0.82em",
        "cursor": "pointer",
        "margin": "3px",
    }


def _mode_toggle_style(active=False):
    base = dict(TAB_STYLE)
    selected = dict(TAB_SELECTED_STYLE)
    style = selected if active else base
    style.update({
        "display": "inline-block",
        "borderRadius": "4px",
        "padding": "4px 8px",
        "marginRight": "2px",
        "cursor": "pointer",
    })
    return style


def _build_nav(active_page):
    base_style = {
        "textDecoration": "none",
        "padding": "3px 10px",
        "borderRadius": "4px",
        "fontSize": "0.8em",
        "color": "#495057",
    }
    active_style = {
        **base_style,
        "backgroundColor": "#343a40",
        "color": "white",
    }
    return html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "2px",
            "backgroundColor": "rgba(248,249,250,0.96)",
            "border": "1px solid #ced4da",
            "borderRadius": "6px",
            "padding": "3px",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.12)",
            "width": "fit-content",
        },
        children=[
            dcc.Link("Kalipoints", href="/", style=active_style if active_page == "kalipoints" else base_style),
            html.Span("|", style={"color": "#ced4da", "fontSize": "0.75em", "margin": "0 2px"}),
            dcc.Link("Analyse", href="/analyse", style=active_style if active_page == "analyse" else base_style),
        ],
    )


def _empty_figure(title, annotation_text):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 55, "b": 130},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": annotation_text,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5,
            "showarrow": False,
            "font": {"size": 14, "color": "#6c757d"},
        }],
    )
    return fig


def _parse_metadata_string(value):
    if pd.isna(value):
        return {}

    text = str(value).strip()
    if not text:
        return {}
    if text.startswith("Metadata(") and text.endswith(")"):
        text = text[len("Metadata("):-1]

    result = {}
    for part in text.split(", "):
        key, separator, raw_value = part.partition(": ")
        if separator:
            result[key.strip()] = raw_value.strip()
    return result


def _flatten_column_name(column):
    if isinstance(column, tuple):
        return "_".join(str(part) for part in column if part is not None and str(part) != "")
    return str(column)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    return None


def _normalize_axis_constraint_mode(value):
    normalized = str(value or "").strip().lower()
    if normalized in ("pitch_only", "nicken_ohne_rollen", "ohne_rollen"):
        return "pitch_only"
    return "pitch_roll"


def _axis_constraint_label(value):
    return "Nicken ohne Rollen" if value == "pitch_only" else "Nicken und Rollen"


def _infer_export_timestamp(dataset_name):
    match = re.search(r"(\d{2}-\d{2}-\d{2})_(\d{2}-\d{2})$", str(dataset_name))
    if not match:
        return None, None
    export_date, export_time = match.groups()
    return export_date, export_time.replace("-", ":")


def _infer_angle(dataset_name):
    match = re.match(r"^(\d+)_", str(dataset_name))
    return int(match.group(1)) if match else None


def _infer_point_amount(dataset_name):
    match = re.match(r"^\d+_[A-Za-z]+_(\d+)_", str(dataset_name))
    return int(match.group(1)) if match else None


def _classify_error_category(row):
    hi_active = any(abs(row[column]) > 1e-12 for column in ("HI-x-True", "HI-y-True", "HI-z-True"))
    si_dist_active = any(abs(row[column] - 1.0) > 1e-12 for column in ("SI-x-Faktor-True", "SI-y-Faktor-True", "SI-z-Faktor-True"))
    si_rot_active = any(abs(row[column]) > 1e-12 for column in ("SI-x-Rotation-True", "SI-y-Rotation-True", "SI-z-Rotation-True"))

    lookup = {
        (False, False, False): "no_error",
        (True, False, False): "hi_only",
        (False, True, False): "si_dist_only",
        (False, False, True): "si_rot_only",
        (True, True, False): "hi_si_dist",
        (True, False, True): "hi_si_rot",
        (False, True, True): "si_dist_rot",
        (True, True, True): "hi_si_dist_rot",
    }
    return lookup[(hi_active, si_dist_active, si_rot_active)]


def _angle_label(angle_value):
    if pd.isna(angle_value):
        return None
    return f"± {int(float(angle_value))}°"


def _build_trace_label(display_label, category_key, point_amount, keep_density, axis_constraint_mode, complete_count):
    category_label = IRON_ERROR_CONFIG[category_key]["label"]
    axis_label = _axis_constraint_label(axis_constraint_mode)
    return f"Einstellungen: {category_label} | {point_amount} Punkte | {axis_label}" + (" " * 5)


def _hex_to_rgba(hex_color, alpha):
    value = str(hex_color).strip().lstrip("#")
    if len(value) != 6:
        return f"rgba(0,0,0,{alpha})"
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def _fill_opacity_for_category(category_key):
    return float(BOX_FILL_OPACITY_OVERRIDES.get(category_key, BOX_FILL_OPACITY))


def _box_stats_by_angle(angle_value_map):
    categories = []
    q1_values = []
    median_values = []
    q3_values = []
    lower_fence = []
    upper_fence = []

    for angle_label in ANGLE_ORDER:
        values_for_angle = angle_value_map.get(angle_label)
        if not values_for_angle:
            continue
        series = pd.Series(values_for_angle, dtype="float64")
        categories.append(angle_label)
        q1_values.append(float(series.quantile(0.25)))
        median_values.append(float(series.quantile(0.5)))
        q3_values.append(float(series.quantile(0.75)))
        lower_fence.append(float(series.min()))
        upper_fence.append(float(series.max()))

    return categories, q1_values, median_values, q3_values, lower_fence, upper_fence


def _group_sort_key(group_key):
    category_key, point_amount, keep_density, axis_constraint_mode = group_key
    category_index = IRON_FILTER_ORDER.index(category_key) if category_key in IRON_FILTER_ORDER else len(IRON_FILTER_ORDER)
    point_index = POINT_FILTER_ORDER.index(int(point_amount)) if int(point_amount) in POINT_FILTER_ORDER else len(POINT_FILTER_ORDER)
    density_index = DENSITY_FILTER_ORDER.index(bool(keep_density)) if bool(keep_density) in DENSITY_FILTER_ORDER else len(DENSITY_FILTER_ORDER)
    axis_index = AXIS_CONSTRAINT_FILTER_ORDER.index(axis_constraint_mode) if axis_constraint_mode in AXIS_CONSTRAINT_FILTER_ORDER else len(AXIS_CONSTRAINT_FILTER_ORDER)
    return category_index, point_index, density_index, axis_index


def _iter_grouped_entries(df):
    grouped = df.groupby(["iron_error_category", "point_amount_setting", "keep_point_density_setting", "axis_constraint_setting"], dropna=False)
    entries = []
    for key, group in grouped:
        category_key, point_amount, _, _ = key
        if pd.isna(point_amount) or category_key not in IRON_ERROR_CONFIG:
            continue
        entries.append((key, group))
    entries.sort(key=lambda item: _group_sort_key(item[0]))
    return entries


def _add_legend_square(fig, trace_label, fill_color, border_color):
    # Legend-only Dummy-Trace: nutzt eine gueltige Kategorie, beeinflusst den Plot aber nicht sichtbar.
    fig.add_trace(go.Scatter(
        x=[ANGLE_ORDER[0]],
        y=[float("nan")],
        mode="markers",
        name=trace_label,
        legendgroup=trace_label,
        showlegend=True,
        hoverinfo="skip",
        marker={
            "symbol": "square",
            "size": 40,
            "color": fill_color,
            "line": {"width": LEGEND_BORDER_WIDTH, "color": border_color},
            "opacity": 1.0,
        },
    ))


def _angle_title_with_prefix(base_title, angle_error_mode):
    prefix_map = {
        "mean": "Mittlerer",
        "max": "Maximaler",
    }
    prefix = prefix_map.get(angle_error_mode)
    if not prefix:
        return base_title

    clean = str(base_title or "").strip()
    for existing in ("Mittlerer ", "Maximaler "):
        if clean.startswith(existing):
            clean = clean[len(existing):]
            break
    return f"{prefix} {clean}"


def _count_complete_axis_datasets(group, display_mode, error_mode):
    columns = DISPLAY_MODE_CONFIG[display_mode]["columns"]
    required = [columns[axis] for axis in ("x", "y", "z")]
    count = 0
    for _, row in group.iterrows():
        complete = True
        for true_col, measured_col in required:
            true_value = row.get(true_col)
            measured_value = row.get(measured_col)
            if pd.isna(true_value) or pd.isna(measured_value):
                complete = False
                break
            if error_mode in ("ratio", "normalized") and abs(float(true_value)) <= 1e-12:
                complete = False
                break
        if complete:
            count += 1
    return count


def _load_results_dataframe(results_filename):
    if not results_filename:
        return None, "Bitte zuerst eine Results-Datei auswählen."

    results_path = os.path.join(RESULTS_DIR, results_filename)
    if not os.path.exists(results_path):
        return None, f"Results-Datei nicht gefunden: {results_path}"

    try:
        df = pd.read_csv(results_path)
    except Exception as exc:
        return None, f"Results-Datei konnte nicht gelesen werden: {exc}"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [_flatten_column_name(column).strip() for column in df.columns.to_flat_index()]
    else:
        df.columns = [_flatten_column_name(column).strip() for column in df.columns]
    if "Metadata" not in df.columns:
        return None, "Ausgewählte Results-Datei enthält keine Metadata-Spalte."

    numeric_columns = [
        "HI-x-True", "HI-y-True", "HI-z-True",
        "SI-x-Faktor-True", "SI-y-Faktor-True", "SI-z-Faktor-True",
        "SI-x-Rotation-True", "SI-y-Rotation-True", "SI-z-Rotation-True",
        "HI-x-Meassured", "HI-y-Meassured", "HI-z-Meassured",
        "SI-x-Faktor-Meassured", "SI-y-Faktor-Meassured", "SI-z-Faktor-Meassured",
        "SI-x-Rotation-Meassured", "SI-y-Rotation-Meassured", "SI-z-Rotation-Meassured",
        "RSME", "Mean",
        "MAE",
        "RSME-xyz", "MAE-xyz", "Mean-xyz", "RSME-xy", "MAE-xy", "Mean-xy",
        "Azimut-Max", "Azimut-Mean", "Azimut-MAE", "Polar-Max", "Polar-Mean", "Polar-MAE",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "RSME-xyz" not in df.columns and "RSME" in df.columns:
        df["RSME-xyz"] = df["RSME"]
    if "MAE-xyz" not in df.columns and "MAE" in df.columns:
        df["MAE-xyz"] = df["MAE"]
    if "Mean-xyz" not in df.columns and "Mean" in df.columns:
        df["Mean-xyz"] = df["Mean"]
    if "RSME-xy" not in df.columns:
        df["RSME-xy"] = pd.Series([None] * len(df), dtype="Float64")
    if "MAE-xy" not in df.columns:
        df["MAE-xy"] = pd.Series([None] * len(df), dtype="Float64")
    if "Mean-xy" not in df.columns:
        df["Mean-xy"] = pd.Series([None] * len(df), dtype="Float64")
    if "Azimut-Max" not in df.columns:
        df["Azimut-Max"] = pd.Series([None] * len(df), dtype="Float64")
    if "Azimut-Mean" not in df.columns:
        df["Azimut-Mean"] = pd.Series([None] * len(df), dtype="Float64")
    if "Azimut-MAE" not in df.columns:
        df["Azimut-MAE"] = pd.Series([None] * len(df), dtype="Float64")
    if "Polar-Max" not in df.columns:
        df["Polar-Max"] = pd.Series([None] * len(df), dtype="Float64")
    if "Polar-Mean" not in df.columns:
        df["Polar-Mean"] = pd.Series([None] * len(df), dtype="Float64")
    if "Polar-MAE" not in df.columns:
        df["Polar-MAE"] = pd.Series([None] * len(df), dtype="Float64")

    metadata_df = pd.DataFrame(df["Metadata"].apply(_parse_metadata_string).tolist())
    if not metadata_df.empty:
        if isinstance(metadata_df.columns, pd.MultiIndex):
            metadata_df.columns = [_flatten_column_name(column).strip() for column in metadata_df.columns.to_flat_index()]
        else:
            metadata_df.columns = [_flatten_column_name(column).strip() for column in metadata_df.columns]
        metadata_df = metadata_df.add_prefix("meta_").reset_index(drop=True)
        df = df.reset_index(drop=True)
        for column in metadata_df.columns:
            df[column] = metadata_df[column]

    inferred_timestamps = df["datasetname"].apply(_infer_export_timestamp)
    df["export_date"] = df.get("meta_EXPORT_DATE", pd.Series([None] * len(df))).fillna(inferred_timestamps.str[0])
    df["export_time"] = df.get("meta_EXPORT_TIME", pd.Series([None] * len(df))).fillna(inferred_timestamps.str[1])

    df["point_amount_setting"] = pd.to_numeric(
        df.get("meta_POINT_AMOUNT", pd.Series([None] * len(df))),
        errors="coerce",
    ).fillna(df["datasetname"].apply(_infer_point_amount)).astype("Int64")
    df["angular_constraint_deg"] = pd.to_numeric(
        df.get("meta_ANGULAR_CONSTRAINT-DEG", pd.Series([None] * len(df))),
        errors="coerce",
    ).fillna(df["datasetname"].apply(_infer_angle)).astype("Float64")
    df["angle_label"] = df["angular_constraint_deg"].apply(_angle_label)

    density_series = df.get("meta_KEEP POINT DENSITY", pd.Series([None] * len(df))).apply(_parse_bool)
    df["keep_point_density_setting"] = density_series.apply(lambda value: False if value is None else bool(value))
    axis_constraint_series = df.get("meta_AXIS_CONSTRAINT", df.get("meta_AXSIS_CONSTRAINT", pd.Series([None] * len(df))))
    df["axis_constraint_setting"] = axis_constraint_series.apply(_normalize_axis_constraint_mode)
    df["batch_key"] = df["export_date"].fillna("") + " " + df["export_time"].fillna("")
    df["batch_key"] = df["batch_key"].str.strip().replace("", "unbekannt")
    df["iron_error_category"] = df.apply(_classify_error_category, axis=1)

    return df, None


def _collect_axis_errors(group, display_mode, axis_key, error_mode):
    config = DISPLAY_MODE_CONFIG[display_mode]["columns"]
    if axis_key == "total":
        values = []
        source_names = []
        skipped_div_zero = 0
        required_columns = [config[axis] for axis in ("x", "y", "z")]

        for _, row in group.iterrows():
            true_components = []
            measured_components = []
            missing_value = False
            for true_col, measured_col in required_columns:
                true_value = row.get(true_col)
                measured_value = row.get(measured_col)
                if pd.isna(true_value) or pd.isna(measured_value):
                    missing_value = True
                    break
                true_components.append(float(true_value))
                measured_components.append(float(measured_value))

            if missing_value:
                continue

            true_magnitude = float((sum(component * component for component in true_components)) ** 0.5)
            measured_magnitude = float((sum(component * component for component in measured_components)) ** 0.5)
            value = calculate_axis_error(true_magnitude, measured_magnitude, error_mode)
            if value is None:
                if error_mode in ("ratio", "normalized") and abs(true_magnitude) <= 1e-12:
                    skipped_div_zero += 1
                continue
            if display_mode == "hard_iron" and error_mode == "absolute":
                # Fuer die absolute HI-Anzeige stets in nT darstellen.
                value = value * 1e9
            values.append(value)
            source_name = str(row.get("datasetname") or "unbekannt")
            source_names.append(source_name)

        return values, source_names, skipped_div_zero

    axes = ("x", "y", "z") if axis_key == "xyz" else (axis_key,)
    values = []
    source_names = []
    skipped_div_zero = 0

    for axis_name in axes:
        true_col, measured_col = config[axis_name]
        if true_col not in group.columns or measured_col not in group.columns:
            continue

        for _, row in group.iterrows():
            true_value = row.get(true_col)
            measured_value = row.get(measured_col)
            value = calculate_axis_error(true_value, measured_value, error_mode)
            if value is None:
                if error_mode in ("ratio", "normalized") and pd.notna(true_value) and pd.notna(measured_value) and abs(float(true_value)) <= 1e-12:
                    skipped_div_zero += 1
                continue
            if display_mode == "hard_iron" and error_mode == "absolute":
                # Fuer die absolute HI-Anzeige stets in nT darstellen.
                value = value * 1e9
            values.append(value)
            source_name = str(row.get("datasetname") or "unbekannt")
            source_names.append(source_name)

    return values, source_names, skipped_div_zero


def _axis_y_label(display_mode, error_mode):
    if error_mode == "absolute":
        if display_mode == "hard_iron":
            return "Absolute Abweichung (nT)"
        return "Absoluter Fehler"
    if error_mode == "ratio":
        return "Verhältnis"
    return "Normalisierte Abweichung"


def _axis_title_prefix(error_mode):
    return {
        "absolute": "Absolute",
        "ratio": "Verhältnismässige",
        "normalized": "Normalisierte",
    }.get(error_mode, "Normalisierte")


def _axis_title(display_mode, axis_name, error_mode):
    suffix = "Achsenverschiebung" if display_mode == "hard_iron" else "Achsenverzerrung"
    prefix = _axis_title_prefix(error_mode)
    if axis_name == "total":
        return f"{prefix} Abweichung der geschätzten Gesamt-{suffix}"
    return f"{prefix} Abweichung der geschätzten {axis_name}-{suffix}"


def _build_axis_analysis_figure(df, display_mode, axis_key, title, indicator_mode, error_mode):
    if df is None or df.empty:
        return _empty_figure(title, "Keine Datensätze für diese Auswahl gefunden."), 0, 0

    mode = indicator_mode or "boxplot"
    show_boxplot = mode in ("boxplot", "boxpoints")
    show_points = mode in ("points", "boxpoints")

    display_config = DISPLAY_MODE_CONFIG[display_mode]
    figure = go.Figure()
    grouped_entries = _iter_grouped_entries(df)
    line_colors = {
        key: TRACE_BORDER_COLORS[index % len(TRACE_BORDER_COLORS)]
        for index, (key, _) in enumerate(grouped_entries)
    }

    skipped_total = 0
    rendered_values = 0

    for (category_key, point_amount, keep_density, axis_constraint_mode), group in grouped_entries:

        x_values = []
        y_values = []
        point_sources = []
        angle_value_map = {}
        for angle_label in ANGLE_ORDER:
            angle_group = group[group["angle_label"] == angle_label]
            if angle_group.empty:
                continue
            errors, source_names, skipped = _collect_axis_errors(angle_group, display_mode, axis_key, error_mode)
            skipped_total += skipped
            if not errors:
                continue
            angle_value_map[angle_label] = list(errors)
            x_values.extend([angle_label] * len(errors))
            y_values.extend(errors)
            point_sources.extend(source_names)

        if not y_values:
            continue

        rendered_values += len(y_values)
        complete_count = _count_complete_axis_datasets(group, display_mode, error_mode)
        trace_label = _build_trace_label(
            display_config["label"],
            category_key,
            int(point_amount),
            bool(keep_density),
            axis_constraint_mode,
            complete_count,
        )
        fill_color = _hex_to_rgba(IRON_ERROR_CONFIG[category_key]["fill"], _fill_opacity_for_category(category_key))
        border_color = line_colors[(category_key, point_amount, keep_density, axis_constraint_mode)]

        _add_legend_square(figure, trace_label, fill_color, border_color)

        if show_boxplot or show_points:
            if mode == "boxpoints":
                stat_x, stat_q1, stat_median, stat_q3, stat_lower, stat_upper = _box_stats_by_angle(angle_value_map)
                if stat_x:
                    figure.add_trace(go.Box(
                        x=stat_x,
                        q1=stat_q1,
                        median=stat_median,
                        q3=stat_q3,
                        lowerfence=stat_lower,
                        upperfence=stat_upper,
                        name=trace_label,
                        legendgroup=trace_label,
                        offsetgroup=trace_label,
                        showlegend=False,
                        fillcolor=fill_color,
                        line={"color": border_color, "width": BOX_BORDER_WIDTH},
                        marker={"color": border_color, "size": 5},
                        opacity=1.0,
                        notched=False,
                        boxpoints=False,
                        hovertemplate="Kombination: %{fullData.name}<br>Winkel: %{x}<br>Q1: %{q1:.6f}<br>Median: %{median:.6f}<br>Q3: %{q3:.6f}<extra></extra>",
                    ))

                figure.add_trace(go.Box(
                    x=x_values,
                    y=y_values,
                    customdata=point_sources,
                    name=trace_label,
                    legendgroup=trace_label,
                    offsetgroup=trace_label,
                    showlegend=False,
                    fillcolor="rgba(0,0,0,0)",
                    line={"color": "rgba(0,0,0,0)", "width": 0},
                    marker={
                        "color": IRON_ERROR_CONFIG[category_key]["fill"],
                        "size": 7,
                        "line": {"width": 1.5, "color": border_color},
                    },
                    opacity=1.0,
                    notched=False,
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0,
                    hovertemplate="Kombination: %{fullData.name}<br>Datensatz: %{customdata}<br>Winkel: %{x}<br>Wert: %{y:.6f}<extra></extra>",
                ))
                continue

            show_box_elements = mode in ("boxplot", "boxpoints")
            show_point_elements = mode in ("points", "boxpoints")
            hover_template = (
                "Kombination: %{fullData.name}<br>Datensatz: %{customdata}<br>Winkel: %{x}<br>Wert: %{y:.6f}<extra></extra>"
                if show_point_elements
                else "Kombination: %{fullData.name}<br>Winkel: %{x}<br>Wert: %{y:.6f}<extra></extra>"
            )
            figure.add_trace(go.Box(
                x=x_values,
                y=y_values,
                customdata=point_sources,
                name=trace_label,
                legendgroup=trace_label,
                offsetgroup=trace_label,
                showlegend=False,
                fillcolor=fill_color if show_box_elements else "rgba(0,0,0,0)",
                line={"color": border_color, "width": BOX_BORDER_WIDTH} if show_box_elements else {"color": "rgba(0,0,0,0)", "width": 0},
                marker={
                    "color": border_color if show_box_elements else IRON_ERROR_CONFIG[category_key]["fill"],
                    "size": 7 if show_point_elements else 5,
                    "line": {"width": 1.5, "color": border_color} if show_point_elements else {"width": 0, "color": border_color},
                },
                opacity=1.0,
                notched=False,
                boxpoints="all" if show_point_elements else False,
                jitter=0.35 if show_point_elements else 0,
                pointpos=0,
                hovertemplate=hover_template,
            ))

    if not figure.data:
        return _empty_figure(title, "Keine Datensätze für diese Auswahl gefunden."), skipped_total, rendered_values

    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": ANALYSIS_TITLE_FONT_SIZE}},
        template="plotly_white",
        showlegend=True,
        boxmode="group",
        scattermode="group",
        margin={"l": 70, "r": 20, "t": 70, "b": 170},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.25,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#dee2e6",
            "borderwidth": 1,
            "font": {"size": ANALYSIS_LEGEND_FONT_SIZE},
            "itemsizing": "trace",
        },
        uirevision=f"{display_mode}-{axis_key}-{error_mode}",
    )
    figure.update_xaxes(
        title="Zulässiger Bewegungswinkelbereich",
        title_font={"size": ANALYSIS_AXIS_TITLE_FONT_SIZE},
        tickfont={"size": ANALYSIS_TICK_FONT_SIZE},
        type="category",
        categoryorder="array",
        categoryarray=ANGLE_ORDER,
        tickangle=0,
        gridcolor="#f1f3f5",
        zeroline=False,
    )
    figure.update_yaxes(
        title=_axis_y_label(display_mode, error_mode),
        title_font={"size": ANALYSIS_AXIS_TITLE_FONT_SIZE},
        tickfont={"size": ANALYSIS_TICK_FONT_SIZE},
        gridcolor="#e9ecef",
        zeroline=True,
        zerolinecolor="#adb5bd",
        exponentformat="power",
        showexponent="all",
    )
    return figure, skipped_total, rendered_values


def _build_radius_analysis_figure(df, title, indicator_mode, radius_error_mode, ignore_z_values):
    if df is None or df.empty:
        return _empty_figure(title, "Keine Datensätze für diese Auswahl gefunden."), 0

    mode = indicator_mode or "boxplot"
    show_boxplot = mode in ("boxplot", "boxpoints")
    show_points = mode in ("points", "boxpoints")
    use_xy_dimension = bool(ignore_z_values)
    dimension_suffix = "xy" if use_xy_dimension else "xyz"
    metric_prefix = {
        "rmse": "RSME",
        "mae": "MAE",
        "mean": "Mean",
    }.get(radius_error_mode, "RSME")
    source_column = f"{metric_prefix}-{dimension_suffix}"
    if source_column not in df.columns:
        return _empty_figure(title, "Results-Datei enthält keine Radiusfehler-Spalten."), 0

    figure = go.Figure()
    grouped_entries = _iter_grouped_entries(df)
    line_colors = {
        key: TRACE_BORDER_COLORS[index % len(TRACE_BORDER_COLORS)]
        for index, (key, _) in enumerate(grouped_entries)
    }
    rendered_values = 0

    for (category_key, point_amount, keep_density, axis_constraint_mode), group in grouped_entries:

        x_values = []
        y_values = []
        point_sources = []
        angle_value_map = {}
        for angle_label in ANGLE_ORDER:
            angle_group = group[group["angle_label"] == angle_label]
            if angle_group.empty:
                continue
            values_series = pd.to_numeric(angle_group[source_column], errors="coerce")
            values_for_angle = []
            for row_idx, raw_value in values_series.items():
                if pd.isna(raw_value):
                    continue
                value = float(raw_value)
                if radius_error_mode in ("rmse", "mae"):
                    # RSME/MAE werden in der CSV in Tesla gespeichert, fuer die Analyse in nT darstellen.
                    value = value * 1e9
                values_for_angle.append(value)
                x_values.append(angle_label)
                y_values.append(value)
                point_sources.append(str(angle_group.at[row_idx, "datasetname"]) if "datasetname" in angle_group.columns else "unbekannt")
            if values_for_angle:
                angle_value_map[angle_label] = values_for_angle

        if not y_values:
            continue

        rendered_values += len(y_values)
        complete_count = int(pd.to_numeric(group[source_column], errors="coerce").notna().sum())
        trace_label = _build_trace_label(
            "Radiusfehler",
            category_key,
            int(point_amount),
            bool(keep_density),
            axis_constraint_mode,
            complete_count,
        )
        fill_color = _hex_to_rgba(IRON_ERROR_CONFIG[category_key]["fill"], _fill_opacity_for_category(category_key))
        border_color = line_colors[(category_key, point_amount, keep_density, axis_constraint_mode)]

        _add_legend_square(figure, trace_label, fill_color, border_color)

        if show_boxplot or show_points:
            if mode == "boxpoints":
                stat_x, stat_q1, stat_median, stat_q3, stat_lower, stat_upper = _box_stats_by_angle(angle_value_map)
                if stat_x:
                    figure.add_trace(go.Box(
                        x=stat_x,
                        q1=stat_q1,
                        median=stat_median,
                        q3=stat_q3,
                        lowerfence=stat_lower,
                        upperfence=stat_upper,
                        name=trace_label,
                        legendgroup=trace_label,
                        offsetgroup=trace_label,
                        showlegend=False,
                        fillcolor=fill_color,
                        line={"color": border_color, "width": BOX_BORDER_WIDTH},
                        marker={"color": border_color, "size": 5},
                        opacity=1.0,
                        notched=False,
                        boxpoints=False,
                        hovertemplate="Kombination: %{fullData.name}<br>Winkel: %{x}<br>Q1: %{q1:.6f}<br>Median: %{median:.6f}<br>Q3: %{q3:.6f}<extra></extra>",
                    ))

                figure.add_trace(go.Box(
                    x=x_values,
                    y=y_values,
                    customdata=point_sources,
                    name=trace_label,
                    legendgroup=trace_label,
                    offsetgroup=trace_label,
                    showlegend=False,
                    fillcolor="rgba(0,0,0,0)",
                    line={"color": "rgba(0,0,0,0)", "width": 0},
                    marker={
                        "color": IRON_ERROR_CONFIG[category_key]["fill"],
                        "size": 7,
                        "line": {"width": 1.5, "color": border_color},
                    },
                    opacity=1.0,
                    notched=False,
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0,
                    hovertemplate="Kombination: %{fullData.name}<br>Datensatz: %{customdata}<br>Winkel: %{x}<br>Wert: %{y:.6f}<extra></extra>",
                ))
                continue

            show_box_elements = mode in ("boxplot", "boxpoints")
            show_point_elements = mode in ("points", "boxpoints")
            hover_template = (
                "Kombination: %{fullData.name}<br>Datensatz: %{customdata}<br>Winkel: %{x}<br>Wert: %{y:.6f}<extra></extra>"
                if show_point_elements
                else "Kombination: %{fullData.name}<br>Winkel: %{x}<br>Wert: %{y:.6f}<extra></extra>"
            )
            figure.add_trace(go.Box(
                x=x_values,
                y=y_values,
                customdata=point_sources,
                name=trace_label,
                legendgroup=trace_label,
                offsetgroup=trace_label,
                showlegend=False,
                fillcolor=fill_color if show_box_elements else "rgba(0,0,0,0)",
                line={"color": border_color, "width": BOX_BORDER_WIDTH} if show_box_elements else {"color": "rgba(0,0,0,0)", "width": 0},
                marker={
                    "color": border_color if show_box_elements else IRON_ERROR_CONFIG[category_key]["fill"],
                    "size": 7 if show_point_elements else 5,
                    "line": {"width": 1.5, "color": border_color} if show_point_elements else {"width": 0, "color": border_color},
                },
                opacity=1.0,
                notched=False,
                boxpoints="all" if show_point_elements else False,
                jitter=0.35 if show_point_elements else 0,
                pointpos=0,
                hovertemplate=hover_template,
            ))

    if not figure.data:
        return _empty_figure(title, "Keine Radiusfehlerwerte vorhanden. Bitte zuerst berechnen."), rendered_values

    if radius_error_mode == "rmse":
        y_title = "RMSE (nT)"
    elif radius_error_mode == "mae":
        y_title = "MAE (nT)"
    else:
        y_title = "Mittelwert normalisierter Fehler"
    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": ANALYSIS_TITLE_FONT_SIZE}},
        template="plotly_white",
        showlegend=True,
        boxmode="group",
        scattermode="group",
        margin={"l": 70, "r": 20, "t": 70, "b": 170},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.25,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#dee2e6",
            "borderwidth": 1,
            "font": {"size": ANALYSIS_LEGEND_FONT_SIZE},
            "itemsizing": "trace",
        },
        uirevision=f"radius-{radius_error_mode}-{dimension_suffix}",
    )
    figure.update_xaxes(
        title="Zulässiger Bewegungswinkelbereich",
        title_font={"size": ANALYSIS_AXIS_TITLE_FONT_SIZE},
        tickfont={"size": ANALYSIS_TICK_FONT_SIZE},
        type="category",
        categoryorder="array",
        categoryarray=ANGLE_ORDER,
        tickangle=0,
        gridcolor="#f1f3f5",
        zeroline=False,
    )
    figure.update_yaxes(
        title=y_title,
        title_font={"size": ANALYSIS_AXIS_TITLE_FONT_SIZE},
        tickfont={"size": ANALYSIS_TICK_FONT_SIZE},
        gridcolor="#e9ecef",
        zeroline=True,
        zerolinecolor="#adb5bd",
        exponentformat="power",
        showexponent="all",
    )
    return figure, rendered_values


def _build_angle_analysis_figure(df, title, indicator_mode, angle_error_mode, column_name, label_name):
    if df is None or df.empty:
        return _empty_figure(title, "Keine Datensätze für diese Auswahl gefunden."), 0

    mode = indicator_mode or "boxplot"
    show_boxplot = mode in ("boxplot", "boxpoints")
    show_points = mode in ("points", "boxpoints")

    if column_name not in df.columns:
        return _empty_figure(title, "Results-Datei enthält keine Winkelfehler-Spalten."), 0

    figure = go.Figure()
    grouped_entries = _iter_grouped_entries(df)
    line_colors = {
        key: TRACE_BORDER_COLORS[index % len(TRACE_BORDER_COLORS)]
        for index, (key, _) in enumerate(grouped_entries)
    }
    rendered_values = 0

    for (category_key, point_amount, keep_density, axis_constraint_mode), group in grouped_entries:

        x_values = []
        y_values = []
        point_sources = []
        angle_value_map = {}
        for angle_label in ANGLE_ORDER:
            angle_group = group[group["angle_label"] == angle_label]
            if angle_group.empty:
                continue
            values_series = pd.to_numeric(angle_group[column_name], errors="coerce")
            values_for_angle = []
            for row_idx, raw_value in values_series.items():
                if pd.isna(raw_value):
                    continue
                value = float(raw_value)
                values_for_angle.append(value)
                x_values.append(angle_label)
                y_values.append(value)
                point_sources.append(str(angle_group.at[row_idx, "datasetname"]) if "datasetname" in angle_group.columns else "unbekannt")
            if values_for_angle:
                angle_value_map[angle_label] = values_for_angle

        if not y_values:
            continue

        rendered_values += len(y_values)
        complete_count = int(pd.to_numeric(group[column_name], errors="coerce").notna().sum())
        trace_label = _build_trace_label(
            label_name,
            category_key,
            int(point_amount),
            bool(keep_density),
            axis_constraint_mode,
            complete_count,
        )
        fill_color = _hex_to_rgba(IRON_ERROR_CONFIG[category_key]["fill"], _fill_opacity_for_category(category_key))
        border_color = line_colors[(category_key, point_amount, keep_density, axis_constraint_mode)]

        _add_legend_square(figure, trace_label, fill_color, border_color)

        if show_boxplot or show_points:
            if mode == "boxpoints":
                stat_x, stat_q1, stat_median, stat_q3, stat_lower, stat_upper = _box_stats_by_angle(angle_value_map)
                if stat_x:
                    figure.add_trace(go.Box(
                        x=stat_x,
                        q1=stat_q1,
                        median=stat_median,
                        q3=stat_q3,
                        lowerfence=stat_lower,
                        upperfence=stat_upper,
                        name=trace_label,
                        legendgroup=trace_label,
                        offsetgroup=trace_label,
                        showlegend=False,
                        fillcolor=fill_color,
                        line={"color": border_color, "width": BOX_BORDER_WIDTH},
                        marker={"color": border_color, "size": 5},
                        opacity=1.0,
                        notched=False,
                        boxpoints=False,
                        hovertemplate="Kombination: %{fullData.name}<br>Winkel: %{x}<br>Q1: %{q1:.6f}°<br>Median: %{median:.6f}°<br>Q3: %{q3:.6f}°<extra></extra>",
                    ))

                figure.add_trace(go.Box(
                    x=x_values,
                    y=y_values,
                    customdata=point_sources,
                    name=trace_label,
                    legendgroup=trace_label,
                    offsetgroup=trace_label,
                    showlegend=False,
                    fillcolor="rgba(0,0,0,0)",
                    line={"color": "rgba(0,0,0,0)", "width": 0},
                    marker={
                        "color": IRON_ERROR_CONFIG[category_key]["fill"],
                        "size": 7,
                        "line": {"width": 1.5, "color": border_color},
                    },
                    opacity=1.0,
                    notched=False,
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0,
                    hovertemplate="Kombination: %{fullData.name}<br>Datensatz: %{customdata}<br>Winkel: %{x}<br>Wert: %{y:.6f}°<extra></extra>",
                ))
                continue

            show_box_elements = mode in ("boxplot", "boxpoints")
            show_point_elements = mode in ("points", "boxpoints")
            hover_template = (
                "Kombination: %{fullData.name}<br>Datensatz: %{customdata}<br>Winkel: %{x}<br>Wert: %{y:.6f}°<extra></extra>"
                if show_point_elements
                else "Kombination: %{fullData.name}<br>Winkel: %{x}<br>Wert: %{y:.6f}°<extra></extra>"
            )
            figure.add_trace(go.Box(
                x=x_values,
                y=y_values,
                customdata=point_sources,
                name=trace_label,
                legendgroup=trace_label,
                offsetgroup=trace_label,
                showlegend=False,
                fillcolor=fill_color if show_box_elements else "rgba(0,0,0,0)",
                line={"color": border_color, "width": BOX_BORDER_WIDTH} if show_box_elements else {"color": "rgba(0,0,0,0)", "width": 0},
                marker={
                    "color": border_color if show_box_elements else IRON_ERROR_CONFIG[category_key]["fill"],
                    "size": 7 if show_point_elements else 5,
                    "line": {"width": 1.5, "color": border_color} if show_point_elements else {"width": 0, "color": border_color},
                },
                opacity=1.0,
                notched=False,
                boxpoints="all" if show_point_elements else False,
                jitter=0.35 if show_point_elements else 0,
                pointpos=0,
                hovertemplate=hover_template,
            ))

    if not figure.data:
        return _empty_figure(title, "Keine Winkelfehlerwerte vorhanden. Bitte zuerst berechnen."), rendered_values

    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": ANALYSIS_TITLE_FONT_SIZE}},
        template="plotly_white",
        showlegend=True,
        boxmode="group",
        scattermode="group",
        margin={"l": 70, "r": 20, "t": 70, "b": 170},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.25,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#dee2e6",
            "borderwidth": 1,
            "font": {"size": ANALYSIS_LEGEND_FONT_SIZE},
            "itemsizing": "trace",
        },
        uirevision=f"angle-{column_name}-{mode}",
    )
    figure.update_xaxes(
        title="Zulässiger Bewegungswinkelbereich",
        title_font={"size": ANALYSIS_AXIS_TITLE_FONT_SIZE},
        tickfont={"size": ANALYSIS_TICK_FONT_SIZE},
        type="category",
        categoryorder="array",
        categoryarray=ANGLE_ORDER,
        tickangle=0,
        gridcolor="#f1f3f5",
        zeroline=False,
    )
    figure.update_yaxes(
        title="Winkelfehler",
        title_font={"size": ANALYSIS_AXIS_TITLE_FONT_SIZE},
        tickfont={"size": ANALYSIS_TICK_FONT_SIZE},
        ticksuffix="°",
        gridcolor="#e9ecef",
        zeroline=True,
        zerolinecolor="#adb5bd",
    )
    return figure, rendered_values


def _build_graph_card(graph_id, card_id, hidden=False):
    style = {
        "backgroundColor": "white",
        "border": "1px solid #dee2e6",
        "borderRadius": "10px",
        "padding": "10px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.04)",
    }
    if hidden:
        style["display"] = "none"

    return html.Div(
        id=card_id,
        style=style,
        children=[
            dcc.Graph(
                id=graph_id,
                style={"height": "640px"},
                config={
                    "responsive": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "analyse_plot",
                        "scale": 4,
                    },
                },
            )
        ],
    )


def _build_export_subdir_options():
    return [{"label": name, "value": name} for name in list_export_subdirs()]


def _default_export_subdir(options):
    values = [opt.get("value") for opt in options if opt.get("value")]
    if "neu" in values:
        return "neu"
    return values[0] if values else None


def _build_results_file_options():
    return [{"label": name, "value": name} for name in list_results_files()]


def _default_results_file(options, preferred_subdir=None):
    values = [opt.get("value") for opt in options if opt.get("value")]
    preferred = results_filename_for_subdir(preferred_subdir) if preferred_subdir else None
    if preferred and preferred in values:
        return preferred
    return values[0] if values else None


def create_layout():
    export_subdir_options = _build_export_subdir_options()
    default_export_subdir = _default_export_subdir(export_subdir_options)
    results_file_options = _build_results_file_options()
    default_results_file = _default_results_file(results_file_options, default_export_subdir)

    return html.Div(
        style={"height": "100vh", "display": "flex", "overflow": "hidden", "backgroundColor": "#f8f9fa"},
        children=[
            dcc.Store(id="analysis-title-store", data=DEFAULT_TITLES),
            dcc.Store(id="analysis-settings-open-store", data=True),
            dcc.Store(id="analysis-iron-filter-store", data=[]),
            dcc.Store(id="analysis-point-filter-store", data=[]),
            dcc.Store(id="analysis-density-filter-store", data=[True]),
            dcc.Store(id="analysis-axis-constraint-filter-store", data=[]),
            dcc.Store(id="analysis-export-subdir-store", data=default_export_subdir),
            dcc.Store(id="analysis-results-file-store", data=default_results_file),
            dcc.Store(id="analysis-view-mode-store", data="angle"),
            dcc.Store(id="analysis-radius-ignore-z-store", data=False),
            dcc.Store(id="analysis-refresh-trigger", data=0),
            html.Div(
                id="analyse-settings-sidebar",
                style={**SIDEBAR_STYLE, "width": "360px"},
                children=[
                    html.H2("Einstellungen", style={"marginBottom": "20px"}),
                    html.Div(style=SECTION_STYLE, children=[
                        html.Label("Datenquelle", style={"fontWeight": "bold", "display": "block", "marginBottom": "8px"}),
                        html.Div("Daten-Unterordner für Results füllen:", style={"fontSize": "0.9em", "marginBottom": "4px", "color": "#495057"}),
                        dcc.Dropdown(
                            id="analyse-export-subdir-dropdown",
                            options=export_subdir_options,
                            value=default_export_subdir,
                            clearable=False,
                            style={"marginBottom": "10px"},
                        ),
                        html.Button("Results füllen", id="analyse-fill-results-button", n_clicks=0, style={**BUTTON_STYLE_INLINE, "marginBottom": "8px"}),
                        html.Div("magnetische Feldstärke (in nT)", style={"fontSize": "0.9em", "marginBottom": "4px", "color": "#495057"}),
                        dcc.Input(
                            id="analyse-magnetic-field-input",
                            type="number",
                            value=49750,
                            min=0,
                            step=1,
                            style={"width": "100%", "marginBottom": "8px"},
                        ),
                        html.Button("Kalibrierfehler berechnen", id="analyse-calc-errors-button", n_clicks=0, style={**BUTTON_STYLE_INLINE, "marginBottom": "6px"}),
                        html.Button("Results neu laden", id="analyse-reload-button", n_clicks=0, style=BUTTON_STYLE_INLINE),
                        html.Div(id="analyse-action-status", style={"marginTop": "10px", "fontSize": "0.85em", "color": "#495057"}),
                        html.Div(id="analyse-status", style={"marginTop": "8px", "fontSize": "0.85em", "color": "#495057"}),
                    ]),
                    html.Div(style=SECTION_STYLE, children=[
                        html.Label("Analyse-Results-Datei", style={"fontWeight": "bold", "display": "block", "marginBottom": "8px"}),
                        dcc.Dropdown(
                            id="analyse-results-file-dropdown",
                            options=results_file_options,
                            value=default_results_file,
                            clearable=False,
                            placeholder="Results_*.csv auswählen",
                        ),
                    ]),
                    html.Div(style=SECTION_STYLE, children=[
                        html.Label("Filter", style={"fontWeight": "bold", "display": "block", "marginBottom": "8px"}),
                        html.Label("Iron Fehler", style={"fontSize": "0.9em", "display": "block", "marginBottom": "4px"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                            html.Button(IRON_ERROR_CONFIG[key]["label"], id=f"analyse-iron-btn-{key}", n_clicks=0, style=_base_toggle_button_style(False, IRON_ERROR_CONFIG[key]["fill"])) for key in IRON_FILTER_ORDER
                        ]),
                        html.Label("Punkteanzahl", style={"fontSize": "0.9em", "display": "block", "margin": "10px 0 4px"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                            html.Button(str(value), id=f"analyse-point-btn-{value}", n_clicks=0, style=_base_toggle_button_style(False)) for value in POINT_FILTER_ORDER
                        ]),
                        html.Label("Punktedichte beibehalten", style={"fontSize": "0.9em", "display": "block", "margin": "10px 0 4px"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                            html.Button("True", id="analyse-density-btn-true", n_clicks=0, style=_base_toggle_button_style(False)),
                            html.Button("False", id="analyse-density-btn-false", n_clicks=0, style=_base_toggle_button_style(False)),
                        ]),
                        html.Label("Achseneinschränkung", style={"fontSize": "0.9em", "display": "block", "margin": "10px 0 4px"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                            html.Button("Nicken und Rollen", id="analyse-axis-constraint-btn-pitch-roll", n_clicks=0, style=_base_toggle_button_style(False)),
                            html.Button("Nicken ohne Rollen", id="analyse-axis-constraint-btn-pitch-only", n_clicks=0, style=_base_toggle_button_style(False)),
                        ]),
                    ]),
                    html.Div(style=SECTION_STYLE, children=[
                        html.Label("Anzeige", style={"fontWeight": "bold", "display": "block", "marginBottom": "8px"}),
                        html.Div("Anzeige Modus", style={"fontSize": "0.9em", "marginBottom": "4px"}),
                        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                            html.Button("Achsenfehler", id="analyse-view-btn-axes", n_clicks=0, style=_mode_toggle_style(False)),
                            html.Button("Radiusfehler", id="analyse-view-btn-radius", n_clicks=0, style=_mode_toggle_style(False)),
                            html.Button("Winkelfehler", id="analyse-view-btn-angle", n_clicks=0, style=_mode_toggle_style(True)),
                        ]),
                        html.Div(id="analyse-axes-controls", children=[
                            html.Div("Iron Fehler", style={"fontSize": "0.9em", "margin": "10px 0 4px"}),
                            dcc.RadioItems(
                                id="analyse-display-mode",
                                options=[
                                    {"label": " Hard Iron Fehler", "value": "hard_iron"},
                                    {"label": " Soft Iron Verzerrung", "value": "soft_iron_distortion"},
                                ],
                                value="hard_iron",
                                style={"display": "grid", "gap": "6px"},
                            ),
                            html.Div("Fehlerberechnung", style={"fontSize": "0.9em", "margin": "10px 0 4px"}),
                            dcc.RadioItems(
                                id="analyse-axis-error-mode",
                                options=[
                                    {"label": " Absolut: F_kalibriert - F_wahr", "value": "absolute"},
                                    {"label": " Verhältnis: F_kalibriert / F_wahr", "value": "ratio"},
                                    {"label": " Normalisiert: (F_kalibriert - F_wahr) / F_wahr", "value": "normalized"},
                                ],
                                value="normalized",
                                style={"display": "grid", "gap": "6px"},
                            ),
                            html.Div("Indikator", style={"fontSize": "0.9em", "margin": "10px 0 4px"}),
                            dcc.RadioItems(
                                id="analyse-indicator-mode",
                                options=[
                                    {"label": " Box Plot", "value": "boxplot"},
                                    {"label": " Punkte", "value": "points"},
                                    {"label": " Box Points", "value": "boxpoints"},
                                ],
                                value="boxplot",
                                inline=True,
                            ),
                        ]),
                        html.Div(id="analyse-radius-controls", style={"display": "none"}, children=[
                            html.Div("Fehlerberechnung", style={"fontSize": "0.9em", "margin": "10px 0 4px"}),
                            dcc.RadioItems(
                                id="analyse-radius-error-mode",
                                options=[
                                    {"label": " RMSE", "value": "rmse"},
                                    {"label": " MAE", "value": "mae"},
                                    {"label": " Mittelwert (Normalisiert)", "value": "mean"},
                                ],
                                value="rmse",
                                style={"display": "grid", "gap": "6px"},
                            ),
                            html.Div("Indikator", style={"fontSize": "0.9em", "margin": "10px 0 4px"}),
                            dcc.RadioItems(
                                id="analyse-radius-indicator-mode",
                                options=[
                                    {"label": " Box Plot", "value": "boxplot"},
                                    {"label": " Punkte", "value": "points"},
                                    {"label": " Box Points", "value": "boxpoints"},
                                ],
                                value="boxplot",
                                inline=True,
                            ),
                            html.Div("Dimension", style={"fontSize": "0.9em", "margin": "10px 0 4px"}),
                            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                                html.Button(
                                    "z-Werte Ignorieren",
                                    id="analyse-radius-ignore-z-btn",
                                    n_clicks=0,
                                    style=_base_toggle_button_style(False),
                                ),
                            ]),
                        ]),
                        html.Div(id="analyse-angle-controls", style={"display": "none"}, children=[
                            html.Div("Fehlerberechnung", style={"fontSize": "0.9em", "margin": "10px 0 4px"}),
                            dcc.RadioItems(
                                id="analyse-angle-error-mode",
                                options=[
                                    {"label": " Maximaler Fehler", "value": "max"},
                                    {"label": " Mittlerer Fehler", "value": "mean"},
                                    {"label": " MAE", "value": "mae"},
                                ],
                                value="mean",
                                style={"display": "grid", "gap": "6px"},
                            ),
                            html.Div("Indikator", style={"fontSize": "0.9em", "margin": "10px 0 4px"}),
                            dcc.RadioItems(
                                id="analyse-angle-indicator-mode",
                                options=[
                                    {"label": " Box Plot", "value": "boxplot"},
                                    {"label": " Punkte", "value": "points"},
                                    {"label": " Box Points", "value": "boxpoints"},
                                ],
                                value="boxplot",
                                inline=True,
                            ),
                        ]),
                    ]),
                    html.Div(style=SECTION_STYLE, children=[
                        html.Label("Graphen Titel", style={"fontWeight": "bold", "display": "block", "marginBottom": "8px"}),
                        html.Div(id="analyse-axis-title-inputs", children=[
                            html.Div("Graph 1", style={"fontSize": "0.85em", "marginBottom": "4px"}),
                            dcc.Input(id="analyse-title-x", type="text", value=DEFAULT_TITLES["x"], style={"width": "100%", "marginBottom": "8px"}),
                            html.Div("Graph 2", style={"fontSize": "0.85em", "marginBottom": "4px"}),
                            dcc.Input(id="analyse-title-y", type="text", value=DEFAULT_TITLES["y"], style={"width": "100%", "marginBottom": "8px"}),
                            html.Div("Graph 3", style={"fontSize": "0.85em", "marginBottom": "4px"}),
                            dcc.Input(id="analyse-title-z", type="text", value=DEFAULT_TITLES["z"], style={"width": "100%", "marginBottom": "8px"}),
                        ]),
                        html.Div(id="analyse-radius-title-input", style={"display": "none"}, children=[
                            html.Div("Radiuswerte", style={"fontSize": "0.85em", "marginBottom": "4px"}),
                            dcc.Input(id="analyse-title-radius", type="text", value=DEFAULT_TITLES["radius"], style={"width": "100%", "marginBottom": "8px"}),
                        ]),
                        html.Div(id="analyse-angle-title-inputs", style={"display": "none"}, children=[
                            html.Div("Azimutfehler (xy-Ebene)", style={"fontSize": "0.85em", "marginBottom": "4px"}),
                            dcc.Input(id="analyse-title-angle-azimuth", type="text", value=DEFAULT_TITLES["angle_azimuth"], style={"width": "100%", "marginBottom": "8px"}),
                            html.Div("Zenitfehler (z-Richtung)", style={"fontSize": "0.85em", "marginBottom": "4px"}),
                            dcc.Input(id="analyse-title-angle-zenith", type="text", value=DEFAULT_TITLES["angle_zenith"], style={"width": "100%", "marginBottom": "8px"}),
                        ]),
                        html.Button("Titel anwenden", id="analyse-apply-titles", n_clicks=0, style=BUTTON_STYLE_INLINE),
                    ]),
                ],
            ),
            html.Div(
                style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden", "padding": "16px"},
                children=[
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "10px"}, children=[
                        html.Button("❮", id="analyse-toggle-settings", n_clicks=0, style={
                            "width": "28px", "height": "28px", "border": "1px solid #adb5bd", "borderRadius": "6px", "backgroundColor": "white", "cursor": "pointer"
                        }),
                        _build_nav("analyse"),
                    ]),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "14px", "overflowY": "auto", "paddingRight": "8px"},
                        children=[
                            _build_graph_card("analyse-graph-x", "analyse-card-x"),
                            _build_graph_card("analyse-graph-y", "analyse-card-y"),
                            _build_graph_card("analyse-graph-z", "analyse-card-z"),
                            _build_graph_card("analyse-graph-radius", "analyse-card-radius", hidden=True),
                        ],
                    ),
                ],
            ),
        ],
    )


def register_callbacks(app):
    @app.callback(
        Output("analysis-export-subdir-store", "data"),
        Input("analyse-export-subdir-dropdown", "value"),
        prevent_initial_call=True,
    )
    def set_analysis_export_subdir(selected_subdir):
        return selected_subdir

    @app.callback(
        Output("analysis-results-file-store", "data"),
        Input("analyse-results-file-dropdown", "value"),
        prevent_initial_call=True,
    )
    def set_analysis_results_file(selected_results_file):
        return selected_results_file

    @app.callback(
        Output("analyse-export-subdir-dropdown", "options"),
        Output("analyse-export-subdir-dropdown", "value"),
        Input("analysis-refresh-trigger", "data"),
        Input("analyse-reload-button", "n_clicks"),
        State("analyse-export-subdir-dropdown", "value"),
    )
    def refresh_export_subdir_dropdown(_, __, current_value):
        options = _build_export_subdir_options()
        valid_values = {opt.get("value") for opt in options}
        selected = current_value if current_value in valid_values else _default_export_subdir(options)
        return options, selected

    @app.callback(
        Output("analyse-results-file-dropdown", "options"),
        Output("analyse-results-file-dropdown", "value"),
        Input("analysis-refresh-trigger", "data"),
        Input("analyse-reload-button", "n_clicks"),
        Input("analyse-export-subdir-dropdown", "value"),
        State("analyse-results-file-dropdown", "value"),
    )
    def refresh_results_file_dropdown(_, __, selected_subdir, current_value):
        options = _build_results_file_options()
        valid_values = {opt.get("value") for opt in options}
        preferred = results_filename_for_subdir(selected_subdir) if selected_subdir else None
        if current_value in valid_values:
            selected = current_value
        else:
            selected = _default_results_file(options, selected_subdir)
        if preferred in valid_values:
            selected = preferred
        return options, selected

    @app.callback(
        Output("analysis-settings-open-store", "data"),
        Output("analyse-settings-sidebar", "style"),
        Output("analyse-toggle-settings", "children"),
        Input("analyse-toggle-settings", "n_clicks"),
        State("analysis-settings-open-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_settings_sidebar(_, is_open):
        new_open = not bool(is_open)
        if new_open:
            return new_open, {**SIDEBAR_STYLE, "width": "360px"}, "❮"
        return new_open, {"width": "0px", "padding": "0px", "overflow": "hidden", "transition": "all 0.4s", "border": "none"}, "❯"

    @app.callback(
        Output("analysis-view-mode-store", "data"),
        Output("analyse-view-btn-axes", "style"),
        Output("analyse-view-btn-radius", "style"),
        Output("analyse-view-btn-angle", "style"),
        Output("analyse-axes-controls", "style"),
        Output("analyse-radius-controls", "style"),
        Output("analyse-angle-controls", "style"),
        Output("analyse-axis-title-inputs", "style"),
        Output("analyse-radius-title-input", "style"),
        Output("analyse-angle-title-inputs", "style"),
        Output("analyse-card-x", "style"),
        Output("analyse-card-y", "style"),
        Output("analyse-card-z", "style"),
        Output("analyse-card-radius", "style"),
        Input("analyse-view-btn-axes", "n_clicks"),
        Input("analyse-view-btn-radius", "n_clicks"),
        Input("analyse-view-btn-angle", "n_clicks"),
        Input("analyse-display-mode", "value"),
        State("analysis-view-mode-store", "data"),
    )
    def toggle_view_mode(_, __, ___, display_mode, current_mode):
        mode = current_mode or "radius"
        ctx = callback_context
        if ctx.triggered:
            triggered = ctx.triggered[0]["prop_id"].split(".")[0]
            if triggered == "analyse-view-btn-radius":
                mode = "radius"
            elif triggered == "analyse-view-btn-axes":
                mode = "axes"
            elif triggered == "analyse-view-btn-angle":
                mode = "angle"

        base_card_style = {
            "backgroundColor": "white",
            "border": "1px solid #dee2e6",
            "borderRadius": "10px",
            "padding": "10px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.04)",
        }
        hidden_card_style = {**base_card_style, "display": "none"}

        axes_btn_style = _mode_toggle_style(mode == "axes")
        radius_btn_style = _mode_toggle_style(mode == "radius")
        angle_btn_style = _mode_toggle_style(mode == "angle")

        if mode == "radius":
            return (
                mode,
                axes_btn_style,
                radius_btn_style,
                angle_btn_style,
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                hidden_card_style,
                hidden_card_style,
                hidden_card_style,
                base_card_style,
            )

        if mode == "angle":
            return (
                mode,
                axes_btn_style,
                radius_btn_style,
                angle_btn_style,
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                base_card_style,
                base_card_style,
                hidden_card_style,
                hidden_card_style,
            )

        return (
            mode,
            axes_btn_style,
            radius_btn_style,
            angle_btn_style,
            {"display": "block"},
            {"display": "none"},
            {"display": "none"},
            {"display": "block"},
            {"display": "none"},
            {"display": "none"},
            base_card_style,
            base_card_style,
            base_card_style,
            base_card_style if display_mode == "hard_iron" else hidden_card_style,
        )

    @app.callback(
        Output("analysis-iron-filter-store", "data"),
        Output("analyse-iron-btn-no_error", "style"),
        Output("analyse-iron-btn-hi_only", "style"),
        Output("analyse-iron-btn-si_dist_only", "style"),
        Output("analyse-iron-btn-hi_si_dist", "style"),
        Output("analyse-iron-btn-si_dist_rot", "style"),
        Output("analyse-iron-btn-hi_si_dist_rot", "style"),
        Input("analyse-iron-btn-no_error", "n_clicks"),
        Input("analyse-iron-btn-hi_only", "n_clicks"),
        Input("analyse-iron-btn-si_dist_only", "n_clicks"),
        Input("analyse-iron-btn-hi_si_dist", "n_clicks"),
        Input("analyse-iron-btn-si_dist_rot", "n_clicks"),
        Input("analyse-iron-btn-hi_si_dist_rot", "n_clicks"),
        State("analysis-iron-filter-store", "data"),
    )
    def toggle_iron_filters(*args):
        selected = [] if args[-1] is None else list(args[-1])
        button_map = {f"analyse-iron-btn-{key}": key for key in IRON_FILTER_ORDER}
        ctx = callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            selected_key = button_map.get(button_id)
            if selected_key is not None:
                if selected_key in selected:
                    selected.remove(selected_key)
                else:
                    selected.append(selected_key)

        selected_set = set(selected)
        styles = [_base_toggle_button_style(key in selected_set, IRON_ERROR_CONFIG[key]["fill"]) for key in IRON_FILTER_ORDER]
        return [selected] + styles

    @app.callback(
        Output("analysis-point-filter-store", "data"),
        Output("analyse-point-btn-100", "style"),
        Output("analyse-point-btn-1000", "style"),
        Output("analyse-point-btn-10000", "style"),
        Input("analyse-point-btn-100", "n_clicks"),
        Input("analyse-point-btn-1000", "n_clicks"),
        Input("analyse-point-btn-10000", "n_clicks"),
        State("analysis-point-filter-store", "data"),
    )
    def toggle_point_filters(*args):
        selected = [] if args[-1] is None else list(args[-1])
        button_map = {
            "analyse-point-btn-100": 100,
            "analyse-point-btn-1000": 1000,
            "analyse-point-btn-10000": 10000,
        }
        ctx = callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            selected_key = button_map.get(button_id)
            if selected_key is not None:
                if selected_key in selected:
                    selected.remove(selected_key)
                else:
                    selected.append(selected_key)

        selected_set = set(selected)
        styles = [_base_toggle_button_style(value in selected_set, "#495057") for value in POINT_FILTER_ORDER]
        return [selected] + styles

    @app.callback(
        Output("analysis-density-filter-store", "data"),
        Output("analyse-density-btn-true", "style"),
        Output("analyse-density-btn-false", "style"),
        Input("analyse-density-btn-true", "n_clicks"),
        Input("analyse-density-btn-false", "n_clicks"),
        State("analysis-density-filter-store", "data"),
    )
    def toggle_density_filters(*args):
        selected = [] if args[-1] is None else list(args[-1])
        button_map = {
            "analyse-density-btn-true": True,
            "analyse-density-btn-false": False,
        }
        ctx = callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            selected_key = button_map.get(button_id)
            if selected_key is not None:
                if selected_key in selected:
                    selected.remove(selected_key)
                else:
                    selected.append(selected_key)

        selected_set = set(selected)
        styles = [_base_toggle_button_style(value in selected_set, "#495057") for value in DENSITY_FILTER_ORDER]
        return [selected] + styles

    @app.callback(
        Output("analysis-axis-constraint-filter-store", "data"),
        Output("analyse-axis-constraint-btn-pitch-roll", "style"),
        Output("analyse-axis-constraint-btn-pitch-only", "style"),
        Input("analyse-axis-constraint-btn-pitch-roll", "n_clicks"),
        Input("analyse-axis-constraint-btn-pitch-only", "n_clicks"),
        State("analysis-axis-constraint-filter-store", "data"),
    )
    def toggle_axis_constraint_filters(*args):
        selected = [] if args[-1] is None else list(args[-1])
        button_map = {
            "analyse-axis-constraint-btn-pitch-roll": "pitch_roll",
            "analyse-axis-constraint-btn-pitch-only": "pitch_only",
        }
        ctx = callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            selected_key = button_map.get(button_id)
            if selected_key is not None:
                if selected_key in selected:
                    selected.remove(selected_key)
                else:
                    selected.append(selected_key)

        selected_set = set(selected)
        styles = [_base_toggle_button_style(value in selected_set, "#495057") for value in AXIS_CONSTRAINT_FILTER_ORDER]
        return [selected] + styles

    @app.callback(
        Output("analysis-radius-ignore-z-store", "data"),
        Output("analyse-radius-ignore-z-btn", "style"),
        Input("analyse-radius-ignore-z-btn", "n_clicks"),
        State("analysis-radius-ignore-z-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_radius_dimension(_, ignore_z_values):
        ignore_z = not bool(ignore_z_values)
        return ignore_z, _base_toggle_button_style(ignore_z)

    @app.callback(
        Output("analysis-refresh-trigger", "data"),
        Output("analyse-action-status", "children"),
        Input("analyse-fill-results-button", "n_clicks"),
        Input("analyse-calc-errors-button", "n_clicks"),
        State("analyse-export-subdir-dropdown", "value"),
        State("analyse-magnetic-field-input", "value"),
        State("analysis-refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def run_analysis_actions(_, __, export_subdir, magnetic_field_strength_nt, refresh_counter):
        ctx = callback_context
        if not ctx.triggered:
            return refresh_counter, ""

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        current_counter = int(refresh_counter or 0)

        if button_id == "analyse-fill-results-button":
            result = build_results_csv(export_subdir)
            return current_counter + 1, result.get("message", "Results füllen abgeschlossen.")

        if button_id == "analyse-calc-errors-button":
            result = calculate_and_write_radius_errors(
                export_subdir=export_subdir,
                magnetic_field_strength_nt=magnetic_field_strength_nt,
            )
            return current_counter + 1, result.get("message", "Kalibrierfehler berechnet.")

        return current_counter, ""

    @app.callback(
        Output("analysis-title-store", "data"),
        Input("analyse-apply-titles", "n_clicks"),
        State("analyse-title-x", "value"),
        State("analyse-title-y", "value"),
        State("analyse-title-z", "value"),
        State("analyse-title-radius", "value"),
        State("analyse-title-angle-azimuth", "value"),
        State("analyse-title-angle-zenith", "value"),
        prevent_initial_call=True,
    )
    def update_titles(_, title_x, title_y, title_z, title_radius, title_angle_azimuth, title_angle_zenith):
        return {
            "x": title_x or DEFAULT_TITLES["x"],
            "y": title_y or DEFAULT_TITLES["y"],
            "z": title_z or DEFAULT_TITLES["z"],
            "radius": title_radius or DEFAULT_TITLES["radius"],
            "angle_azimuth": title_angle_azimuth or DEFAULT_TITLES["angle_azimuth"],
            "angle_zenith": title_angle_zenith or DEFAULT_TITLES["angle_zenith"],
        }

    @app.callback(
        Output("analyse-graph-x", "figure"),
        Output("analyse-graph-y", "figure"),
        Output("analyse-graph-z", "figure"),
        Output("analyse-graph-radius", "figure"),
        Output("analyse-status", "children"),
        Input("analyse-reload-button", "n_clicks"),
        Input("analysis-refresh-trigger", "data"),
        Input("analysis-iron-filter-store", "data"),
        Input("analysis-point-filter-store", "data"),
        Input("analysis-density-filter-store", "data"),
        Input("analysis-axis-constraint-filter-store", "data"),
        Input("analysis-view-mode-store", "data"),
        Input("analyse-display-mode", "value"),
        Input("analyse-axis-error-mode", "value"),
        Input("analyse-indicator-mode", "value"),
        Input("analyse-radius-error-mode", "value"),
        Input("analyse-radius-indicator-mode", "value"),
        Input("analyse-angle-error-mode", "value"),
        Input("analyse-angle-indicator-mode", "value"),
        Input("analysis-radius-ignore-z-store", "data"),
        Input("analyse-radius-ignore-z-btn", "n_clicks"),
        Input("analysis-title-store", "data"),
        Input("analysis-results-file-store", "data"),
    )
    def render_analysis(_, __, iron_filters, point_filters, density_filters, axis_constraint_filters,
                        view_mode, display_mode, axis_error_mode, indicator_mode,
                        radius_error_mode, radius_indicator_mode, angle_error_mode, angle_indicator_mode,
                        radius_ignore_z_values,
                        radius_ignore_z_clicks, title_store, results_file):
        titles = title_store or DEFAULT_TITLES
        results_df, error_message = _load_results_dataframe(results_file)

        # Robust gegen Timing beim Umschalten: n_clicks-Parität als zusätzliche Quelle.
        ignore_z_active = bool(radius_ignore_z_values) or (int(radius_ignore_z_clicks or 0) % 2 == 1)

        title_x = titles.get("x", DEFAULT_TITLES["x"])
        title_y = titles.get("y", DEFAULT_TITLES["y"])
        title_z = titles.get("z", DEFAULT_TITLES["z"])
        title_radius = titles.get("radius", DEFAULT_TITLES["radius"])
        title_angle_azimuth = titles.get("angle_azimuth", DEFAULT_TITLES["angle_azimuth"])
        title_angle_zenith = titles.get("angle_zenith", DEFAULT_TITLES["angle_zenith"])

        if error_message:
            empty_x = _empty_figure(title_x, error_message)
            empty_y = _empty_figure(title_y, error_message)
            empty_z = _empty_figure(title_z, error_message)
            empty_r = _empty_figure(title_radius, error_message)
            return empty_x, empty_y, empty_z, empty_r, error_message

        filtered = results_df.copy()
        if iron_filters:
            filtered = filtered[filtered["iron_error_category"].isin(iron_filters)]
        else:
            filtered = filtered.iloc[0:0]

        if point_filters:
            filtered = filtered[filtered["point_amount_setting"].isin(point_filters)]
        else:
            filtered = filtered.iloc[0:0]

        if density_filters:
            filtered = filtered[filtered["keep_point_density_setting"].isin(density_filters)]
        else:
            filtered = filtered.iloc[0:0]

        if axis_constraint_filters:
            filtered = filtered[filtered["axis_constraint_setting"].isin(axis_constraint_filters)]
        else:
            filtered = filtered.iloc[0:0]

        filtered = filtered.sort_values(["angular_constraint_deg", "datasetname"], ascending=[False, True])
        batch_count = filtered["batch_key"].nunique() if not filtered.empty else 0
        combo_count = filtered[["iron_error_category", "point_amount_setting", "keep_point_density_setting", "axis_constraint_setting"]].drop_duplicates().shape[0] if not filtered.empty else 0

        if view_mode == "radius":
            fig_radius, rendered = _build_radius_analysis_figure(
                filtered,
                title_radius,
                radius_indicator_mode,
                radius_error_mode,
                ignore_z_active,
            )
            empty_x = _empty_figure(title_x, "Achsenfehler-Modus aktivieren, um diesen Graphen zu sehen.")
            empty_y = _empty_figure(title_y, "Achsenfehler-Modus aktivieren, um diesen Graphen zu sehen.")
            empty_z = _empty_figure(title_z, "Achsenfehler-Modus aktivieren, um diesen Graphen zu sehen.")
            indicator_text = INDICATOR_LABELS.get(radius_indicator_mode, "Box Plot")
            metric_text = {
                "rmse": "RMSE",
                "mae": "MAE",
                "mean": "Mittelwert (Normalisiert)",
            }.get(radius_error_mode, "RMSE")
            use_xy_dimension = ignore_z_active
            dimension_text = "XY (z ignoriert)" if use_xy_dimension else "XYZ"
            status = (
                f"Geladen: {len(filtered)} Datensätze | Kombinationen: {combo_count} | Batches: {batch_count} | "
                f"Anzeige: Radiusfehler | Fehlerberechnung: {metric_text} | Indikator: {indicator_text} | "
                f"Dimension: {dimension_text} | Werte: {rendered}"
            )
            return empty_x, empty_y, empty_z, fig_radius, status

        if view_mode == "angle":
            if angle_error_mode == "max":
                azimuth_column = "Azimut-Max"
                zenith_column = "Polar-Max"
            elif angle_error_mode == "mae":
                azimuth_column = "Azimut-MAE"
                zenith_column = "Polar-MAE"
            else:
                azimuth_column = "Azimut-Mean"
                zenith_column = "Polar-Mean"

            angle_title_azimuth = _angle_title_with_prefix(title_angle_azimuth, angle_error_mode)
            angle_title_zenith = _angle_title_with_prefix(title_angle_zenith, angle_error_mode)

            fig_azimuth, rendered_azimuth = _build_angle_analysis_figure(
                filtered,
                angle_title_azimuth,
                angle_indicator_mode,
                angle_error_mode,
                azimuth_column,
                "Azimutfehler",
            )
            fig_zenith, rendered_zenith = _build_angle_analysis_figure(
                filtered,
                angle_title_zenith,
                angle_indicator_mode,
                angle_error_mode,
                zenith_column,
                "Zenitfehler",
            )
            empty_z = _empty_figure(title_z, "Winkelfehler-Modus zeigt nur Azimut und Zenit.")
            empty_r = _empty_figure(title_radius, "Winkelfehler-Modus zeigt nur Azimut und Zenit.")
            indicator_text = INDICATOR_LABELS.get(angle_indicator_mode, "Box Plot")
            metric_text = {
                "max": "Maximaler Fehler",
                "mean": "Mittlerer Fehler",
                "mae": "MAE",
            }.get(angle_error_mode, "Mittlerer Fehler")
            status = (
                f"Geladen: {len(filtered)} Datensätze | Kombinationen: {combo_count} | Batches: {batch_count} | "
                f"Anzeige: Winkelfehler | Fehlerberechnung: {metric_text} | Indikator: {indicator_text} | "
                f"Werte: Azimut={rendered_azimuth}, Zenit={rendered_zenith}"
            )
            return fig_azimuth, fig_zenith, empty_z, empty_r, status

        axis_title_x = _axis_title(display_mode, "x", axis_error_mode)
        axis_title_y = _axis_title(display_mode, "y", axis_error_mode)
        axis_title_z = _axis_title(display_mode, "z", axis_error_mode)
        axis_title_total = _axis_title(display_mode, "total", axis_error_mode)

        fig_x, skipped_x, rendered_x = _build_axis_analysis_figure(filtered, display_mode, "x", axis_title_x, indicator_mode, axis_error_mode)
        fig_y, skipped_y, rendered_y = _build_axis_analysis_figure(filtered, display_mode, "y", axis_title_y, indicator_mode, axis_error_mode)
        fig_z, skipped_z, rendered_z = _build_axis_analysis_figure(filtered, display_mode, "z", axis_title_z, indicator_mode, axis_error_mode)

        if display_mode == "hard_iron":
            fig_total, skipped_total_axis, rendered_total_axis = _build_axis_analysis_figure(
                filtered,
                display_mode,
                "total",
                axis_title_total,
                indicator_mode,
                axis_error_mode,
            )
        else:
            fig_total = _empty_figure(axis_title_total, "Gesamtachsenverschiebung ist nur für Hard Iron verfügbar.")
            skipped_total_axis = 0
            rendered_total_axis = 0

        indicator_text = INDICATOR_LABELS.get(indicator_mode, "Box Plot")
        calc_label = {
            "absolute": "Absolut",
            "ratio": "Verhältnis",
            "normalized": "Normalisiert",
        }.get(axis_error_mode, "Normalisiert")

        skipped_total = skipped_x + skipped_y + skipped_z + skipped_total_axis
        rendered_total = rendered_x + rendered_y + rendered_z + rendered_total_axis
        warning = ""
        if rendered_total == 0 and skipped_total > 0:
            warning = " | Hinweis: Keine Werte darstellbar (Division durch 0 bei Bruchfunktionen)."

        status = (
            f"Geladen: {len(filtered)} Datensätze | Kombinationen: {combo_count} | Batches: {batch_count} | "
            f"Anzeige: {DISPLAY_MODE_CONFIG[display_mode]['label']} | Fehlerberechnung: {calc_label} | Indikator: {indicator_text}"
            f"{warning}"
        )

        return fig_x, fig_y, fig_z, fig_total, status
