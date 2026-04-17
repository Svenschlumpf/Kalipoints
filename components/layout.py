from dash import dcc, html

from components.styles import (
    SECTION_STYLE, BUTTON_STYLE_INLINE, TAB_STYLE, TAB_SELECTED_STYLE,
    SIDEBAR_STYLE, BUTTON_STYLE
)
from utils.csv_io import get_available_seeds, get_calibration_dirs, CALIBRATION_DIR



def build_left_sidebar():
    """Erstellt die linke Sidebar mit allen Einstellungen."""
    return html.Div(id='left-sidebar', style=SIDEBAR_STYLE, children=[
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
        
        # 1b. Magnetische Flussdichte
        html.Div(style=SECTION_STYLE, children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                html.Label("Magnetische Flussdichte (in nT):", style={'fontWeight': 'bold', 'flexShrink': 0, 'whiteSpace': 'nowrap'}),
                html.Div(style={'flexGrow': 1}, children=[
                    dcc.Input(id='flux-density-input', type='number', min=0, step=1, value=49750, placeholder='Standard: 49750', style={'width': '100%'})
                ]),
            ]),
        ]),

        # 2. Fehlerabweichung/Rauschen
        html.Div(style=SECTION_STYLE, children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[ 
                html.Label("Rauschen:", style={'fontWeight': 'bold', 'marginRight': '5px', 'whiteSpace': 'nowrap'}),
                html.Div(style={'flexGrow': 1}, children=[
                    dcc.Input(id='noise-input', type='number', step=0.0001, placeholder='Fehlerabweichung (Standard: 0 bzw. 0.001)', style={'width': '100%'})
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
            html.Label("Winkeleinschränkung", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '8px'}),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                html.Label("Bewegungswinkel (+/- in Grad):", style={'fontWeight': 'bold', 'flexShrink': 0, 'marginBottom': '0'}),
                dcc.Input(
                    id='angular-constraint-input',
                    type='number',
                    min=0,
                    max=90,
                    step=1,
                    placeholder='90',
                    style={'width': '90px', 'marginLeft': 'auto'}
                ),
            ]),
            html.P(
                "Als +- Wert aus der horizontalen Lage, wobei die Einschränkung für Nicken und Rollen als gleichgross angesehen wird",
                style={'fontSize': '0.78em', 'color': '#666', 'marginTop': '6px', 'marginBottom': '0'}
            ),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginTop': '8px'}, children=[
                dcc.Checklist(
                    id='batch-mode-toggle',
                    options=[{'label': ' Batch erstellen', 'value': 'batch'}],
                    value=[],
                    style={'flexShrink': 0}
                ),
                dcc.Input(
                    id='batch-step-input',
                    type='number',
                    min=1,
                    max=90,
                    step=1,
                    placeholder='Gradschritt',
                    disabled=True,
                    style={'flexGrow': 1, 'minWidth': '0', 'backgroundColor': '#e9ecef', 'color': '#6c757d'}
                ),
            ]),
            html.Div(style={'marginTop': '8px'}, children=[
                dcc.Checklist(
                    id='density-mode-toggle',
                    options=[{'label': ' Punktedichte beibehalten', 'value': 'density'}],
                    value=['density'],
                ),
                html.P(
                    "Achtung: bei Aktivierung verringert sich die tatsächliche Punkteanzahl von der eingegebenen",
                    style={'fontSize': '0.78em', 'color': '#999', 'marginTop': '4px', 'marginBottom': '0'}
                ),
                html.Div(style={'marginTop': '8px'}, children=[
                    html.Label('Achseneinschränkung:', style={'fontSize': '0.8em', 'fontWeight': 'bold', 'marginBottom': '3px'}),
                    dcc.RadioItems(
                        id='axis-constraint-mode',
                        options=[
                            {'label': ' Nicken und Rollen', 'value': 'pitch_roll'},
                            {'label': ' Nicken ohne Rollen', 'value': 'pitch_only'},
                        ],
                        value='pitch_roll',
                        style={'fontSize': '0.8em'}
                    ),
                ]),
            ]),
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
        _build_hard_iron_section(),
        
        # 6. Soft Iron (Verzerrung) mit Mode-Tabs
        _build_soft_iron_section(),
        
        # 7. Metadaten & Buttons
        html.Div([
            html.Hr(style={'marginTop': '5px'}),
            html.P("Uhrzeit, der Erzeugung der Daten: (hh:mm:ss)", id='display-time', style={'fontSize': '0.8em', 'color': '#555'}),
            
            html.Button('Punkte Erzeugen', id='submit-button', n_clicks=0, style={'width': '100%', 'marginBottom': '15px'}),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '10px'}, children=[
                html.Label("Custom:", style={'flexShrink': 0, 'fontSize': '0.9em'}),
                dcc.Input(id='custom-filename-input', type='text', placeholder='-', style={'flexGrow': 1}),
            ]),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '15px'}, children=[
                html.Label("Anzahl der zu erzeugenden Datensets:", style={'flexShrink': 0, 'fontSize': '0.9em'}),
                dcc.Input(id='dataset-count-input', type='number', min=1, step=1, placeholder='1', style={'width': '60px', 'flexShrink': 0}),
            ]),
            html.Button('Datenset(s) exportieren', id='export-dataset', n_clicks=0, style={'width': '100%', 'marginBottom': '15px'}),
            html.Button('Plot als HTML exportieren', id='export-plot-html', n_clicks=0, style={'width': '100%', 'marginBottom': '10px'}),
            html.P(
                'Exportiert den aktuell angezeigten 3D-Plot als eigenstaendige HTML-Datei.',
                style={'fontSize': '0.8em', 'color': '#666', 'marginTop': '0', 'marginBottom': '15px'}
            ),
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
    ])


def build_center_area():
    """Erstellt den mittleren Bereich mit dem 3D-Plot."""
    _NAV_LINK = {
        'textDecoration': 'none',
        'padding': '3px 10px',
        'borderRadius': '4px',
        'fontSize': '0.8em',
        'color': '#495057',
    }
    _NAV_LINK_ACTIVE = {
        **_NAV_LINK,
        'backgroundColor': '#343a40',
        'color': 'white',
    }
    _VIEW_BTN = {
        'padding': '4px 10px',
        'border': '1px solid #ced4da',
        'borderRadius': '4px',
        'fontSize': '0.78em',
        'color': '#495057',
        'backgroundColor': '#f8f9fa',
        'cursor': 'pointer',
    }
    _VIEW_BTN_ACTIVE = {
        **_VIEW_BTN,
        'backgroundColor': '#343a40',
        'border': '1px solid #343a40',
        'color': 'white',
    }
    return html.Div([
        html.Div("❮", id="btn-toggle-left", n_clicks=0, style={**BUTTON_STYLE, "left": "0", "borderRadius": "0 8px 8px 0"}),
        html.Div("❯", id="btn-toggle-right", n_clicks=0, style={**BUTTON_STYLE, "right": "0", "borderRadius": "8px 0 0 8px"}),
        # Floating Seitennavigation – oben links im Plot-Bereich
        html.Div(
            style={
                'position': 'absolute',
                'top': '10px',
                'left': '28px',
                'zIndex': 150,
                'display': 'flex',
                'alignItems': 'center',
                'gap': '2px',
                'backgroundColor': 'rgba(248,249,250,0.92)',
                'border': '1px solid #ced4da',
                'borderRadius': '6px',
                'padding': '3px 3px',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.12)',
            },
            children=[
                html.Span("", style={'fontSize': '0.8em', 'color': '#adb5bd', 'marginRight': '4px'}),
                dcc.Link('Kalipoints', href='/', style=_NAV_LINK_ACTIVE),
                html.Span("|", style={'color': '#ced4da', 'fontSize': '0.75em', 'margin': '0 2px'}),
                dcc.Link('Analyse', href='/analyse', style=_NAV_LINK),
            ]
        ),
        dcc.Graph(id='sphere-plot', style={'width': '100%', 'height': '100%'}, config={
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'kalipoints_plot',
                'scale': 2
            }
        }),
        # Floating View-Steuerung – unten links im Plot-Bereich
        html.Div(
            style={
                'position': 'absolute',
                'bottom': '12px',
                'left': '28px',
                'zIndex': 160,
                'display': 'flex',
                'alignItems': 'center',
                'gap': '8px',
            },
            children=[
                html.Button(
                    'P',
                    id='btn-view-projection-toggle',
                    n_clicks=0,
                    title='Perspektive / Isometrie umschalten',
                    style={
                        'width': '36px',
                        'height': '36px',
                        'borderRadius': '50%',
                        'border': '1px solid #007bff',
                        'backgroundColor': '#007bff',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'fontSize': '0.85em',
                        'cursor': 'pointer',
                        'boxShadow': '0 1px 4px rgba(0,0,0,0.18)',
                    }
                ),
                html.Div(
                    style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'gap': '3px',
                        'backgroundColor': 'rgba(248,249,250,0.92)',
                        'border': '1px solid #ced4da',
                        'borderRadius': '6px',
                        'padding': '3px',
                        'boxShadow': '0 1px 4px rgba(0,0,0,0.12)',
                    },
                    children=[
                        html.Button('iso', id='btn-view-plane-iso', n_clicks=0, style=_VIEW_BTN_ACTIVE),
                        html.Button('xz', id='btn-view-plane-xz', n_clicks=0, style=_VIEW_BTN),
                        html.Button('yz', id='btn-view-plane-yz', n_clicks=0, style=_VIEW_BTN),
                        html.Button('xy', id='btn-view-plane-xy', n_clicks=0, style=_VIEW_BTN),
                    ],
                ),
                html.Button(
                    'O',
                    id='btn-view-origin-toggle',
                    n_clicks=0,
                    title='Ursprung (0,0,0) ein/aus',
                    style={
                        'width': '25px',
                        'height': '25px',
                        'borderRadius': '50%',
                        'border': '1px solid #343a40',
                        'backgroundColor': '#343a40',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'fontSize': '0.78em',
                        'cursor': 'pointer',
                        'boxShadow': '0 1px 4px rgba(0,0,0,0.18)',
                    }
                ),
                html.Button(
                    'S',
                    id='btn-view-scale-toggle',
                    n_clicks=0,
                    title='Achsen- und Legendenbeschriftung skalieren',
                    style={
                        'width': '25px',
                        'height': '25px',
                        'borderRadius': '50%',
                        'border': '1px solid #6c757d',
                        'backgroundColor': '#6c757d',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'fontSize': '0.78em',
                        'cursor': 'pointer',
                        'boxShadow': '0 1px 4px rgba(0,0,0,0.18)',
                    }
                ),
            ],
        ),
    ], style={'flex': '1', 'position': 'relative', 'height': '100vh', 'overflow': 'hidden'})


def build_right_sidebar():
    """Erstellt die rechte Sidebar mit Resultaten und Datenset-Laden."""
    return html.Div(id='right-sidebar', style={**SIDEBAR_STYLE, "border-right": "none", "border-left": "1px solid #ddd", "width": "400px"}, children=[
        html.H3("Resultate", style={'marginBottom': '20px', 'whiteSpace': 'nowrap'}),
        
        # Datenset Laden Section
        html.Div(style={'marginBottom': '20px', 'paddingBottom': '15px', 'borderBottom': '1px solid #ddd'}, children=[
            html.Label("Datenset laden:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
            # Quelle wechseln
            html.Div(style={'display': 'flex', 'gap': '6px', 'marginBottom': '8px'}, children=[
                html.Button(
                    'Simuliert',
                    id='btn-source-simulated',
                    n_clicks=0,
                    style={
                        'flex': '1', 'backgroundColor': '#007bff', 'color': 'white',
                        'border': '2px solid #007bff', 'borderRadius': '4px',
                        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
                    }
                ),
                html.Button(
                    'Echtdaten',
                    id='btn-source-reallife',
                    n_clicks=0,
                    style={
                        'flex': '1', 'backgroundColor': '#f8f9fa', 'color': '#6c757d',
                        'border': '2px solid #adb5bd', 'borderRadius': '4px',
                        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
                    }
                ),
            ]),
            html.Div(style={'marginBottom': '8px'}, children=[
                html.Label("Datenordner:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '3px', 'fontSize': '0.85em'}),
                dcc.Dropdown(
                    id='dataset-folder-dropdown',
                    options=[],
                    value=None,
                    clearable=False,
                    style={'fontSize': '0.8em'}
                ),
            ]),
            html.Div(style={'marginBottom': '8px'}, children=[
                html.Label("Kalibrierordner:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '3px', 'fontSize': '0.85em'}),
                dcc.Dropdown(
                    id='calibration-dir-dropdown',
                    options=get_calibration_dirs(),
                    value=CALIBRATION_DIR,
                    clearable=False,
                    style={'fontSize': '0.8em'}
                ),
            ]),
            html.Details(
                id='simulated-seed-filter-details',
                open=False,
                style={'marginBottom': '8px'},
                children=[
                    html.Summary('Filter (nur Simuliert)', style={'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '0.85em'}),
                    html.Div(id='simulated-seed-filter-container', style={'marginTop': '8px', 'display': 'block'}, children=[
                        html.Label('Iron Fehler:', style={'fontSize': '0.8em', 'fontWeight': 'bold', 'marginBottom': '3px'}),
                        dcc.Dropdown(
                            id='sim-filter-iron-error',
                            options=[
                                {'label': 'Kein Fehler', 'value': 'none'},
                                {'label': 'Nur Hard Iron', 'value': 'hi_only'},
                                {'label': 'Nur Soft Iron Verzerrung', 'value': 'si_dist_only'},
                                {'label': 'Nur Soft Iron Rotation', 'value': 'si_rot_only'},
                                {'label': 'Hard Iron + Soft Iron Verzerrung', 'value': 'hi_si_dist'},
                                {'label': 'Hard Iron + Soft Iron Rotation', 'value': 'hi_si_rot'},
                                {'label': 'Soft Iron Verzerrung + Rotation', 'value': 'si_dist_si_rot'},
                                {'label': 'Hard Iron + Soft Iron Verzerrung + Rotation', 'value': 'all'},
                            ],
                            value=None,
                            clearable=True,
                            placeholder='Iron Fehler auswählen',
                            style={'fontSize': '0.8em', 'marginBottom': '8px'}
                        ),
                        html.Label('Punkteanzahl:', style={'fontSize': '0.8em', 'fontWeight': 'bold', 'marginBottom': '3px'}),
                        dcc.Dropdown(
                            id='sim-filter-point-amount',
                            options=[
                                {'label': '100', 'value': 100},
                                {'label': '1000', 'value': 1000},
                                {'label': '10000', 'value': 10000},
                            ],
                            value=None,
                            clearable=True,
                            placeholder='Punkteanzahl auswählen',
                            style={'fontSize': '0.8em', 'marginBottom': '8px'}
                        ),
                        html.Label('Punktedichte beibehalten:', style={'fontSize': '0.8em', 'fontWeight': 'bold', 'marginBottom': '3px'}),
                        dcc.RadioItems(
                            id='sim-filter-keep-density',
                            options=[
                                {'label': 'An', 'value': 'on'},
                                {'label': 'Aus', 'value': 'off'},
                            ],
                            value=None,
                            inline=True,
                            style={'fontSize': '0.8em', 'marginBottom': '8px'}
                        ),
                        html.Label('Achseneinschränkung:', style={'fontSize': '0.8em', 'fontWeight': 'bold', 'marginBottom': '3px'}),
                        dcc.Dropdown(
                            id='sim-filter-axis-constraint',
                            options=[
                                {'label': 'Nicken und Rollen', 'value': 'pitch_roll'},
                                {'label': 'Nicken ohne Rollen', 'value': 'pitch_only'},
                            ],
                            value=None,
                            clearable=True,
                            placeholder='Achseneinschränkung auswählen',
                            style={'fontSize': '0.8em', 'marginBottom': '8px'}
                        ),
                        html.Label('Winkeleinschränkung:', style={'fontSize': '0.8em', 'fontWeight': 'bold', 'marginBottom': '3px'}),
                        dcc.RangeSlider(
                            id='sim-filter-angle-range',
                            min=5,
                            max=90,
                            step=5,
                            value=[5, 90],
                            allowCross=False,
                            marks={5: '5', 15: '15', 25: '25', 35: '35', 45: '45', 55: '55', 65: '65', 75: '75', 85: '85', 90: '90'},
                            tooltip={'always_visible': False, 'placement': 'bottom'}
                        ),
                        html.Div(style={'marginTop': '10px', 'display': 'flex', 'justifyContent': 'flex-end'}, children=[
                            html.Button('Zurücksetzen', id='sim-filter-reset-button', n_clicks=0, style=BUTTON_STYLE_INLINE),
                            html.Button('Anwenden', id='sim-filter-apply-button', n_clicks=0, style=BUTTON_STYLE_INLINE)
                        ]),
                    ])
                ]
            ),
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

            # Toggle-Buttons: Datenpunkte anzeigen/ausblenden
            html.Div(style={'display': 'flex', 'gap': '8px', 'marginTop': '10px'}, children=[
                html.Button(
                    'Unkalibriert',
                    id='btn-toggle-uncalibrated',
                    n_clicks=0,
                    style={
                        'flex': '1', 'backgroundColor': '#dc3545', 'color': 'white',
                        'border': '2px solid #dc3545', 'borderRadius': '4px',
                        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
                    }
                ),
                html.Button(
                    'Optimal',
                    id='btn-toggle-optimal',
                    n_clicks=0,
                    style={
                        'flex': '1', 'backgroundColor': '#dbeafe', 'color': '#0d6efd',
                        'border': '2px solid #0d6efd', 'borderRadius': '4px',
                        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
                    }
                ),
                html.Button(
                    'Kalibriert',
                    id='btn-toggle-calibrated',
                    n_clicks=0,
                    style={
                        'flex': '1', 'backgroundColor': '#28a745', 'color': 'white',
                        'border': '2px solid #28a745', 'borderRadius': '4px',
                        'padding': '4px 8px', 'cursor': 'pointer', 'fontSize': '0.85em'
                    }
                ),
            ]),
        ]),
        
        # Results Container
        html.Div(id='results-container', children="Hier erscheinen später die Ergebnisse...", style={'padding': '10px', 'border': '1px dashed #ccc', 'height': '100%'})
    ])


# ============================================================
# Private Hilfsfunktionen für die Sidebar-Sektionen
# ============================================================

def _build_hard_iron_section():
    """Erstellt die Hard Iron (Versatz) Sektion."""
    return html.Div(style=SECTION_STYLE, children=[
        html.Label("Hard Iron (Versatz in nT):", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
        dcc.Tabs(id="hard-iron-mode-tabs", value='hard-iron-random', style={'height': '30px'}, children=[
            dcc.Tab(label='Manuell', value='hard-iron-manual', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE, children=[
                html.Div(style={'paddingTop': '10px'}, children=[
                    html.Label("Achsenabweichung (in nT):"),
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
                                html.Label("Zufallsbereich (in nT):"),
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
    ])


def _build_soft_iron_section():
    """Erstellt die Soft Iron (Verzerrung/Rotation) Sektion."""
    return html.Div(style=SECTION_STYLE, children=[
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
    ])
