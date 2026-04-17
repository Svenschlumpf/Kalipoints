from dash import Dash, dcc, html, Input, Output

from pages import kalipoints, analyse

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    style={'height': '100vh', 'overflow': 'hidden'},
    children=[
        dcc.Location(id='url', refresh=False),
        html.Div(
            id='page-content',
            style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'}
        ),
    ]
)


@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/analyse':
        return analyse.create_layout()
    return kalipoints.create_layout()


kalipoints.register_callbacks(app)
analyse.register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)
