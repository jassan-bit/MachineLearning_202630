"""Dashboard for the executed daily experiment. python dashboard.py."""
from pathlib import Path
import os
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
app = Dash(__name__, title='SVR | Volatilidad diaria')
server = app.server

if not (RESULTS/'metrics.csv').exists():
    app.layout = html.Main([html.H1('Experimento diario'), html.P('Resultados pendientes de ejecución.')])
else:
    metrics = pd.read_csv(RESULTS/'metrics.csv')
    app.layout = html.Main([
        html.H1('Volatilidad diaria: SVR lineal'),
        html.P('BTC · ETH · BNB · XRP | Binance 1 minuto → cierre diario UTC | 2020–2025'),
        html.P('Entrenamiento 2020–2022 · Validación 2023 · Test 2024–2025. '
               'tsxv GroupKFold con filtros cronológicos; folds dependientes y pocas muestras de evaluación.'),
        html.Label('Criptomoneda'), dcc.Dropdown(metrics.symbol.unique(), 'BTCUSDT', id='symbol', clearable=False),
        html.Label('Ventana de volatilidad (días)'), dcc.Dropdown([7,14,21,28], 7, id='volatility', clearable=False),
        html.Label('Ventana de precios (días)'), dcc.Dropdown([7,14,21,28], 7, id='lag', clearable=False),
        html.Div(id='summary'), dcc.Graph(id='folds'), dcc.Graph(id='horizons'),
        html.P('Compare ventanas de entrada manteniendo fija la definición de volatilidad. '
               'El menor RMSE entre objetivos diferentes no identifica el mejor modelo. '
               'Persistence repite la última volatilidad conocida. Los valores negativos del SVR se conservan.'),
        html.A('Informe', href=os.environ.get('BOOK_URL', 'https://jassan-bit.github.io/MachineLearning_202630/')),
    ], style={'maxWidth':'1100px','margin':'auto','fontFamily':'Arial','padding':'24px'})

    @app.callback(Output('folds','figure'), Output('horizons','figure'), Output('summary','children'),
                  Input('symbol','value'), Input('volatility','value'), Input('lag','value'))
    def update(symbol, volatility, lag):
        data = metrics.query('symbol == @symbol and volatility_window == @volatility and input_window == @lag and split == "test"')
        folds = data[data.horizon == 0]
        by_h = data[data.horizon > 0].groupby(['model','horizon'], as_index=False).rmse.mean()
        f = px.bar(folds, x='fold', y='rmse', color='model', barmode='group', title='RMSE test por fold (puntos porcentuales)')
        h = px.line(by_h, x='horizon', y='rmse', color='model', markers=True, title='RMSE test por horizonte')
        text = ' | '.join(f'{name}: RMSE {g.rmse.mean():.4f}' for name,g in folds.groupby('model'))
        return f, h, text

import json
minute_status = RESULTS/'minute_2023_2025/status.json'
if minute_status.exists():
    from minute_dashboard import create_app, create_progress_app
    app = create_app() if json.loads(minute_status.read_text())['status'] == 'complete' else create_progress_app()
    server = app.server

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',8050)), debug=False)
