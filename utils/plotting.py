import numpy as np
import plotly.graph_objects as go
from dash import html


LEGEND_TEXT_SIZE = 18
LEGEND_DOT_SIZE = 10
POINT_DOT_SIZE = 2
AXIS_TICK_FONT_SIZE = 12
AXIS_TITLE_FONT_SIZE = 14
AXIS_TICK_COUNT = 20
# Skalierungsfaktor für den S-Button (Achsenbeschriftungen + Legende)
AXIS_LEGEND_SCALE_FACTOR = 1.55
# Relativer Skalierungseinfluss nur für Tick-Zahlen (kleiner = schwächerer Effekt)
AXIS_TICK_SCALE_RELATIVE = 0.3


def apply_axes_legend_scale(fig, scale_enabled, scale_factor=AXIS_LEGEND_SCALE_FACTOR):
    """Skaliert Achsen-Ticks/-Titel sowie Legenden-Text für bessere Export-Lesbarkeit."""
    factor = float(scale_factor) if scale_enabled else 1.0
    tick_factor = 1.0 + (factor - 1.0) * AXIS_TICK_SCALE_RELATIVE
    tick_size = int(round(AXIS_TICK_FONT_SIZE * tick_factor))
    title_size = int(round(AXIS_TITLE_FONT_SIZE * factor*1.1))
    legend_size = int(round(LEGEND_TEXT_SIZE * factor))

    fig.update_layout(
        scene=dict(
            xaxis=dict(tickfont=dict(size=tick_size), title_font=dict(size=title_size)),
            yaxis=dict(tickfont=dict(size=tick_size), title_font=dict(size=title_size)),
            zaxis=dict(tickfont=dict(size=tick_size), title_font=dict(size=title_size)),
        ),
        legend=dict(font=dict(size=legend_size)),
    )
    return fig


def build_figure_with_points(data_points, center_offset, sphere_vertices, sphere_triangles,
                             alpha, noise_value, x_distortion, y_distortion, z_distortion,
                             mesh_opacity, show_mesh=True, point_color='darkred', calibrated_points=None,
                             show_origin=True, optimal_points=None,
                             uncalibrated_ids=None, calibrated_ids=None, optimal_ids=None,
                             show_uncalibrated=True, show_calibrated=True, show_optimal=False,
                             uncalibrated_label='Unkalibrierte Punkte',
                             calibrated_label='Kalibrierte Punkte',
                             optimal_label='Optimale Punkte'):
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
        if calibrated_points is not None:
            all_points = np.vstack([all_points, calibrated_points])
        if optimal_points is not None and len(optimal_points) > 0:
            all_points = np.vstack([all_points, optimal_points])
    else:
        # Ohne Mesh: Berechne BBox nur mit Datenpunkten
        all_points = data_points
        if calibrated_points is not None:
            all_points = np.vstack([all_points, calibrated_points])
        if optimal_points is not None and len(optimal_points) > 0:
            all_points = np.vstack([all_points, optimal_points])
    
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

    def _normalize_ids(raw_ids, count):
        if count <= 0:
            return []
        if raw_ids is None:
            return list(range(1, count + 1))
        values = list(raw_ids)
        if len(values) != count:
            return list(range(1, count + 1))
        normalized = []
        for index, value in enumerate(values, start=1):
            try:
                normalized.append(int(value))
            except Exception:
                normalized.append(index)
        return normalized

    uncal_ids = _normalize_ids(uncalibrated_ids, len(data_points))
    cal_ids = _normalize_ids(calibrated_ids if calibrated_ids is not None else uncal_ids, len(calibrated_points) if calibrated_points is not None else 0)
    opt_ids = _normalize_ids(optimal_ids, len(optimal_points) if optimal_points is not None else 0)

    # Legendeneintrag fuer den Ursprung mit grossem Symbol (wie Unkalibriert).
    data_traces.append(
        go.Scatter3d(
            x=[float('nan')],
            y=[float('nan')],
            z=[float('nan')],
            mode='markers',
            marker=dict(size=LEGEND_DOT_SIZE, color='black', opacity=1.0),
            name='Koordinatenursprung',
            meta='origin-legend',
            showlegend=True if show_origin else False,
            hoverinfo='none'
        )
    )

    # Fester Ursprungspunkt fuer Orientierung im Plot.
    # Dieser Punkt wird absichtlich nicht fuer die Bereichsberechnung verwendet.
    data_traces.append(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode='markers',
            marker=dict(size=4, color='black', opacity=1.0),
            name='Koordinatenursprung',
            meta='origin-point',
            visible=True if show_origin else False,
            showlegend=False,
            hovertemplate='Ursprung (0,0,0)<extra></extra>'
        )
    )
    
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
    
    # Datenpunkte: Legende-Dummy mit grossem Symbol + Datentrace ohne Legende
    data_traces.append(
        go.Scatter3d(
            x=[float('nan')], y=[float('nan')], z=[float('nan')],
            mode='markers',
            marker=dict(size=LEGEND_DOT_SIZE, color=point_color, opacity=1.0),
            name=uncalibrated_label,
            meta='uncal-legend',
            visible=True if show_uncalibrated else False,
            showlegend=True,
            hoverinfo='none'
        )
    )
    data_traces.append(
        go.Scatter3d(
            x=data_points[:, 0], 
            y=data_points[:, 1], 
            z=data_points[:, 2], 
            mode='markers', 
            customdata=uncal_ids,
            marker=dict(size=POINT_DOT_SIZE, color=point_color, opacity=1.0), 
            name=uncalibrated_label,
            meta='uncal-points',
            visible=True if show_uncalibrated else False,
            showlegend=False,
            hovertemplate='x: %{x:.2f} nT<br>y: %{y:.2f} nT<br>z: %{z:.2f} nT<br>ID: %{customdata}<extra></extra>'
        )
    )

    # Kalibrierte Punkte (grün) hinzufügen, wenn vorhanden
    if calibrated_points is not None:
        data_traces.append(
            go.Scatter3d(
                x=[float('nan')], y=[float('nan')], z=[float('nan')],
                mode='markers',
                marker=dict(size=LEGEND_DOT_SIZE, color='green', opacity=1.0),
                name=calibrated_label,
                meta='cal-legend',
                visible=True if show_calibrated else False,
                showlegend=True,
                hoverinfo='none'
            )
        )
        data_traces.append(
            go.Scatter3d(
                x=calibrated_points[:, 0],
                y=calibrated_points[:, 1],
                z=calibrated_points[:, 2],
                mode='markers',
                customdata=cal_ids,
                marker=dict(size=POINT_DOT_SIZE, color='green', opacity=1.0),
                name=calibrated_label,
                meta='cal-points',
                visible=True if show_calibrated else False,
                showlegend=False,
                hovertemplate='x: %{x:.2f} nT<br>y: %{y:.2f} nT<br>z: %{z:.2f} nT<br>ID: %{customdata}<extra></extra>'
            )
        )

    # Optimale Punkte (blau) hinzufügen, wenn vorhanden
    if optimal_points is not None and len(optimal_points) > 0:
        data_traces.append(
            go.Scatter3d(
                x=[float('nan')], y=[float('nan')], z=[float('nan')],
                mode='markers',
                marker=dict(size=LEGEND_DOT_SIZE, color='royalblue', opacity=1.0),
                name=optimal_label,
                meta='optimal-legend',
                visible=True if show_optimal else False,
                showlegend=True,
                hoverinfo='none'
            )
        )
        data_traces.append(
            go.Scatter3d(
                x=optimal_points[:, 0],
                y=optimal_points[:, 1],
                z=optimal_points[:, 2],
                mode='markers',
                customdata=opt_ids,
                marker=dict(size=POINT_DOT_SIZE, color='royalblue', opacity=1.0),
                name=optimal_label,
                meta='optimal-points',
                visible=True if show_optimal else False,
                showlegend=False,
                hovertemplate='x: %{x:.2f} nT<br>y: %{y:.2f} nT<br>z: %{z:.2f} nT<br>ID: %{customdata}<extra></extra>'
            )
        )
    
    fig = go.Figure(data=data_traces)
    fig.update_layout(
        uirevision='constant',  # Verhindert, dass die Kamera beim Klick aktualisiert/zurückgesetzt wird
        scene=dict(
            aspectmode='cube',  # Wichtig: gleiche Skala auf allen Achsen
            xaxis_title='X (nT)', 
            yaxis_title='Y (nT)', 
            zaxis_title='Z (nT)',
            # ALLE ACHSEN HABEN DEN GLEICHEN BEREICH (um den gemeinsamen Center)
            xaxis=dict(
                range=[center_x - max_range, center_x + max_range],
                nticks=AXIS_TICK_COUNT,
                tickfont=dict(size=AXIS_TICK_FONT_SIZE),
                title_font=dict(size=AXIS_TITLE_FONT_SIZE),
            ), 
            yaxis=dict(
                range=[center_y - max_range, center_y + max_range],
                nticks=AXIS_TICK_COUNT,
                tickfont=dict(size=AXIS_TICK_FONT_SIZE),
                title_font=dict(size=AXIS_TITLE_FONT_SIZE),
            ), 
            zaxis=dict(
                range=[center_z - max_range, center_z + max_range],
                nticks=AXIS_TICK_COUNT,
                tickfont=dict(size=AXIS_TICK_FONT_SIZE),
                title_font=dict(size=AXIS_TITLE_FONT_SIZE),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1), 
            center=dict(x=0, y=0, z=0), 
            eye=dict(x=1.25, y=1.25, z=1.25)
        ),
        hovermode='closest'
    )
    fig.update_layout(
        legend=dict(
            font=dict(size=LEGEND_TEXT_SIZE),
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.5)'
        )
    )
    
    return fig


def build_results_display(data_points, alpha, generation_mode, noise_value, center_offset, 
                         x_distortion, y_distortion, z_distortion, x_rot=0, y_rot=0, z_rot=0,
                         iron_error_matrix=None, sample_count=None, raw_metadata=None,
                         calibration_data=None, calibration_errors=None,
                         keep_point_density=None, axis_constraint_mode='pitch_roll'):
    """
    Erstellt die Ergebnis-Anzeige für die rechte Sidebar mit Metadaten und Transformationsmatrix.
    
    Args:
        raw_metadata: Dict mit Original-CSV-Metadaten (können None-Werte enthalten für fehlende Felder)
    """
    p_style = {'margin': '4px 0', 'fontSize': '0.9em', 'lineHeight': '1.2'}
    title_style = {'margin': '8px 0 4px 0', 'fontSize': '1em', 'fontWeight': 'bold'}
    hr_style = {'margin': '8px 0'}
    matrix_style = {'margin': '2px 0', 'fontSize': '12px', 'fontFamily': 'monospace', 'lineHeight': '1.4', 'letterSpacing': '0.02em'}
    error_label_style = {'margin': '8px 0 0 0', 'fontSize': '0.9em', 'lineHeight': '1.2'}
    error_value_style = {'margin': '2px 0 0 0', 'fontSize': '0.9em', 'lineHeight': '1.2', 'fontFamily': 'monospace'}
    
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
    
    # Eingestellte Punktezahl
    if raw_metadata and 'POINT_AMOUNT' in raw_metadata:
        point_val = raw_metadata.get('POINT_AMOUNT')
        if point_val is None:
            point_display = 'N/A'
        else:
            point_display = str(int(point_val))
    else:
        point_count = sample_count if sample_count is not None else len(data_points)
        point_display = str(point_count)
    result.append(html.P(f"Eingestellte Punktezahl: {point_display}", style=p_style))

    # Tatsächlich verwendete Punktezahl
    actual_points = len(data_points) if data_points is not None else None
    actual_points_display = 'N/A' if actual_points is None else str(int(actual_points))
    result.append(html.P(f"Tatsächliche Punktezahl: {actual_points_display}", style=p_style))
    
    # Winkeleinschränkung
    if raw_metadata and 'ANGULAR_CONSTRAINT_DEG' in raw_metadata:
        alpha_val = raw_metadata.get('ANGULAR_CONSTRAINT_DEG')
    else:
        alpha_val = alpha
    result.append(html.P(f"Winkeleinschränkung: {format_value(alpha_val, '{:.1f}')}°", style=p_style))

    if raw_metadata and 'KEEP_POINT_DENSITY' in raw_metadata:
        keep_density_val = raw_metadata.get('KEEP_POINT_DENSITY')
    else:
        keep_density_val = keep_point_density
    if keep_density_val is None:
        keep_density_text = 'N/A'
    else:
        keep_density_text = 'An' if bool(keep_density_val) else 'Aus'
    result.append(html.P(f"Punktedichte beibehalten: {keep_density_text}", style=p_style))

    if raw_metadata and 'AXIS_CONSTRAINT' in raw_metadata and raw_metadata.get('AXIS_CONSTRAINT'):
        axis_mode = str(raw_metadata.get('AXIS_CONSTRAINT'))
    else:
        axis_mode = str(axis_constraint_mode or 'pitch_roll')
    axis_mode_norm = axis_mode.strip().lower()
    axis_mode_label = 'Nicken ohne Rollen' if axis_mode_norm in ('pitch_only', 'nicken_ohne_rollen', 'ohne_rollen') else 'Nicken und Rollen'
    result.append(html.P(f"Achseneinschränkung: {axis_mode_label}", style=p_style))
    
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
    result.append(html.P(f"Hard Iron in nT (X,Y,Z): {hi_text}", style=p_style))
    
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
            row_text = "[" + ", ".join([f"{val:.6f}" for val in row]) + "]"
            result.append(html.P(row_text, style=matrix_style))
    else:
        result.append(html.P("Transformationsmatrix: N/A (keine Daten)", style=p_style))
    
    result.append(html.Hr(style=hr_style))

    # Kalibrierwerte anzeigen, wenn vorhanden
    if calibration_data is not None:
        import math
        result.append(html.H4("Kalibrierwerte", style=title_style))

        def fmt_calib(val, fmt='{:.7f}'):
            try:
                return fmt.format(float(val))
            except Exception:
                return str(val)

        # Kalibrierdaten koennen je nach Quelle in Tesla oder nT vorliegen.
        # Bei Tesla-Groessenordnungen (< 1) in nT fuer die Anzeige umrechnen.
        center_scale = 1e9 if max(
            abs(float(calibration_data['fit_center_x'])),
            abs(float(calibration_data['fit_center_y'])),
            abs(float(calibration_data['fit_center_z']))
        ) < 1.0 else 1.0

        # Hard Iron Kalibrierwerte in nT anzeigen
        hi_calib = (
            f"({fmt_calib(float(calibration_data['fit_center_x']) * center_scale, '{:.0f}')}, "
            f"{fmt_calib(float(calibration_data['fit_center_y']) * center_scale, '{:.0f}')}, "
            f"{fmt_calib(float(calibration_data['fit_center_z']) * center_scale, '{:.0f}')})"
        )
        result.append(html.P(f"Hard Iron in nT (X,Y,Z): {hi_calib}", style=p_style))

        si_dist_calib = (
            f"({fmt_calib(calibration_data['fit_radius_x'])}, "
            f"{fmt_calib(calibration_data['fit_radius_y'])}, "
            f"{fmt_calib(calibration_data['fit_radius_z'])})"
        )
        result.append(html.P(f"SI Verzerrung (X,Y,Z): {si_dist_calib}", style=p_style))

        roll_deg = math.degrees(calibration_data['fit_rotation_roll'])
        pitch_deg = math.degrees(calibration_data['fit_rotation_pitch'])
        yaw_deg = math.degrees(calibration_data['fit_rotation_yaw'])
        si_rot_calib = f"({roll_deg:.2f}°, {pitch_deg:.2f}°, {yaw_deg:.2f}°)"
        result.append(html.P(f"SI Rotation (Roll,Pitch,Yaw): {si_rot_calib}", style=p_style))

        result.append(html.P("Kalibriermatrix (4x4):", style={'margin': '8px 0 2px 0', 'fontSize': '0.9em', 'fontWeight': 'bold'}))
        calib_matrix_display = np.array(calibration_data['matrix'], dtype=float)
        if np.max(np.abs(calib_matrix_display[:3, 3])) < 1.0:
            calib_matrix_display[:3, 3] = calib_matrix_display[:3, 3] * 1e9
        for row in calib_matrix_display:
            row_text = "[" + ", ".join([f"{val:.6f}" for val in row]) + "]"
            result.append(html.P(row_text, style=matrix_style))

        result.append(html.Hr(style=hr_style))

    # Kalibrierfehler aus Results.csv anzeigen, falls vorhanden
    if calibration_errors is not None:
        result.append(html.H4("Kalibrier Fehler", style=title_style))

        angle_errors = calibration_errors.get('angle') or {}
        radius_errors = calibration_errors.get('radius') or {}

        result.append(html.P("Winkelfehler", style={'margin': '8px 0 0 0', 'fontSize': '0.95em', 'fontWeight': 'bold'}))
        result.append(html.P(
            f"Azimutfehlermittelwert: {format_value(angle_errors.get('azimuth_mean_deg'), '{:.6f}')}°",
            style=error_value_style,
        ))
        result.append(html.P(
            f"Zenitfehlermittelwert: {format_value(angle_errors.get('zenith_mean_deg'), '{:.6f}')}°",
            style=error_value_style,
        ))
        result.append(html.P(
            f"maximaler Azimutfehler: {format_value(angle_errors.get('azimuth_max_deg'), '{:.6f}')}°",
            style=error_value_style,
        ))
        result.append(html.P(
            f"maximaler Zenitfehler: {format_value(angle_errors.get('zenith_max_deg'), '{:.6f}')}°",
            style=error_value_style,
        ))

        mean_t = radius_errors.get('mean_t')
        mae_t = radius_errors.get('mae_t')
        rmse_t = radius_errors.get('rmse_t')

        # Mean-xyz ist in Results als normalisierter Fehler gespeichert (dimensionslos).
        mean_normalized = None if mean_t is None else float(mean_t)
        mae_nt = None if mae_t is None else float(mae_t) * 1e9
        rmse_nt = None if rmse_t is None else float(rmse_t) * 1e9

        result.append(html.P("Radiusfehler", style={'margin': '10px 0 0 0', 'fontSize': '0.95em', 'fontWeight': 'bold'}))
        result.append(html.P(f"Mittelwert (normalisiert): {format_value(mean_normalized, '{:.6f}')}", style=error_value_style))
        result.append(html.P(f"MAE: {format_value(mae_nt, '{:.6f}')}nT", style=error_value_style))
        result.append(html.P(f"RSME: {format_value(rmse_nt, '{:.6f}')}nT", style=error_value_style))

        result.append(html.Hr(style=hr_style))

    return result
