import os

# ============================================================
# KONSTANTEN & KONFIGURATION
# ============================================================
OUTPUT_DIR = os.path.join("datasets", "1-kalipoints_exports", "neu")
DEFAULT_SAMPLES = 1000
DEFAULT_ALPHA = 90
DEFAULT_NOISE = 0.0
DEFAULT_OFFSET = [0.0, 0.0, 0.0]
DEFAULT_DISTORTION = [1.0, 1.0, 1.0]
SIDEBAR_WIDTH = "400px"

# ============================================================
# STYLES
# ============================================================
SECTION_STYLE = {'marginBottom': '20px', 'paddingBottom': '10px', 'borderBottom': '1px dotted #ccc'}
BUTTON_STYLE_INLINE = {'width': 'auto', 'margin': '0 5px', 'flexShrink': 0}

TAB_STYLE = {
    'padding': '5px 9px', 'fontWeight': 500, 'fontSize': '0.8em',
    'border': '1px solid #ddd', 'backgroundColor': '#f0f0f0',
    'borderRadius': '4px 4px 0 0', 'marginRight': '2px'
}
TAB_SELECTED_STYLE = {
    'padding': '5px 9px', 'fontWeight': 'bold',
    'borderTop': '2px solid #007bff', 'borderLeft': '1px solid #ddd',
    'borderRight': '1px solid #ddd', 'borderBottom': '1px solid white',
    'backgroundColor': 'white', 'fontSize': '0.9em', 'borderRadius': '4px 4px 0 0'
}

SIDEBAR_STYLE = {
    "position": "relative", "width": SIDEBAR_WIDTH, "background-color": "#f8f9fa",
    "padding": "20px", "transition": "all 0.5s", "overflow-y": "auto",
    "border-right": "1px solid #ddd", "display": "flex", "flex-direction": "column", "box-sizing": "border-box"
}

COLLAPSED_STYLE = {
    "width": "0px", "padding": "0px", "overflow": "hidden",
    "transition": "all 0.5s", "border": "none"
}

BUTTON_STYLE = {
    "position": "absolute", "top": "50%", "zIndex": "100", "width": "20px", "height": "60px",
    "border": "1px solid #ccc", "backgroundColor": "white", "cursor": "pointer",
    "display": "flex", "alignItems": "center", "justifyContent": "center",
    "fontWeight": "bold", "color": "#555", "borderRadius": "0 5px 5px 0"
}
