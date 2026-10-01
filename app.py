"""Dashboard del Entregable 2. Ejecutar: python app.py."""
from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dashboard_data'
summary = pd.read_csv(DATA / 'eda_target_summary.csv')
quarters = pd.read_csv(DATA / 'temporal_target_quarters.csv')
acf = pd.read_csv(DATA / 'temporal_target_correlations.csv')
metrics = pd.read_csv(DATA / 'base_metrics_by_fold.csv')
coverage = pd.read_csv(DATA / 'univariate_monthly_coverage.csv')
SYMBOLS = sorted(summary.symbol.unique())
BOOK = os.environ.get('BOOK_URL', 'https://jassan-bit.github.io/MachineLearning_202630/')
app = Dash(__name__, title='Volatilidad | Entregable 2')
server = app.server


def graph(identifier):
    return dcc.Graph(id=identifier, config={'displaylogo': False, 'responsive': True})


def note(text):
    return html.P(text, className='note')


app.layout = html.Main([
    html.Header([
        html.P('MACHINE LEARNING · ENTREGABLE 2', className='eyebrow'),
        html.H1('Pronóstico de volatilidad de criptomonedas'),
        html.P('BTC · ETH · BNB · XRP · SOL | Datos horarios de Binance Spot'),
        html.A('Abrir informe en Jupyter Book ↗', href=BOOK, target='_blank', rel='noopener noreferrer'),
    ]),
    html.Div('Validación interna en DEVELOPMENT · TEST reservado', className='banner'),
    html.Div([
        html.Label('Criptomoneda', htmlFor='symbol'),
        dcc.Dropdown(id='symbol', options=[{'label': s.replace('USDT', ''), 'value': s} for s in SYMBOLS],
                     value='BTCUSDT', clearable=False),
    ], className='filter'),
    dcc.Tabs([
        dcc.Tab(label='1. Contexto del problema', children=[
            html.Section([
                html.H2('¿Puede un SVR lineal mejorar una referencia temporal?'),
                html.P('Predecimos la dispersión de los próximos 24 retornos logarítmicos horarios a partir de 168 cierres horarios consecutivos por activo.'),
                html.P('El objetivo es 100 × desviación estándar poblacional de esos 24 retornos, centrados en su media (ddof=0). Se expresa en porcentaje y no se anualiza.'),
                html.H3('Datos y alcance'),
                html.P('Cinco pares contra USDT, un proveedor y frecuencia horaria. Las visualizaciones usan exclusivamente resultados derivados de DEVELOPMENT.'),
                graph('coverage'),
                note('La cobertura mensual cuenta registros disponibles, no observaciones independientes. Meses parciales y huecos pueden reducir el conteo; esta gráfica no identifica su causa.'),
                html.H3('Protocolo de evaluación'),
                html.P('Cinco folds cronológicos de ventana creciente. El escalado se ajusta solo en TRAIN y las ventanas de historia y etiqueta se mantienen dentro de cada bloque. Persistence usa la volatilidad de referencia del pasado.'),
            ])
        ]),
        dcc.Tab(label='2. EDA', children=[html.Section([
            html.H2('Distribución y dependencia temporal'),
            html.Div(id='eda-summary', className='banner'),
            graph('quantiles'),
            note('Los cuantiles resumen la distribución completa de DEVELOPMENT. Una distancia grande entre mediana y percentil 99 indica una cola superior extensa; no es evidencia de errores de registro.'),
            graph('quarters'),
            note('Se muestran media y mediana trimestrales del objetivo. Sus cambios describen variación temporal; no atribuyen causalidad ni prueban estabilidad futura.'),
            graph('acf'),
            note('La ACF del objetivo se calcula sobre el segmento temporal documentado en el informe. Las etiquetas contiguas comparten retornos: una ACF alta no implica por sí sola capacidad predictiva.'),
        ])]),
        dcc.Tab(label='3. Modelos base', children=[html.Section([
            html.H2('SVR lineal frente a Persistence'),
            html.Div([
                html.Label('Fold de validación', htmlFor='fold'),
                dcc.Dropdown(id='fold', options=[{'label': 'Todos (media de folds)', 'value': 'all'}] +
                             [{'label': str(f), 'value': str(f)} for f in sorted(metrics.fold.unique())],
                             value='all', clearable=False),
                html.Label('Métrica', htmlFor='metric'),
                dcc.Dropdown(id='metric', options=[{'label': label, 'value': key} for key, label in
                             [('rmse', 'RMSE (puntos porcentuales)'), ('mae', 'MAE (puntos porcentuales)'),
                              ('mape', 'MAPE (%)'), ('r2', 'R²')]], value='rmse', clearable=False),
            ], className='filter'),
            html.Div(id='model-summary', className='banner'),
            graph('comparison'),
            graph('folds'),
            note('Menor RMSE, MAE o MAPE indica menor error; mayor R² es mejor. R² negativo significa que el error supera al de la media observada en el bloque. MAPE es sensible a valores próximos a cero y no es porcentaje de exactitud.'),
            html.H3('Resultado global y límites'),
            html.P('En los 25 bloques activo-fold, el RMSE medio fue 0,389006 para Persistence y 0,623259 para SVR. El SVR no mejoró la referencia. Estas medias dan igual peso a cada bloque; no son métricas sobre predicciones concatenadas.'),
            html.P('La selección de parámetros y la evaluación usan los mismos folds: existe posible optimismo de selección. Los resultados no constituyen una evaluación independiente de TEST.'),
        ])]),
    ]),
    html.Footer('Jassan Arteta y Mateo Bernal · Universidad del Norte · 2026-30'),
])


def style(fig, ytitle):
    fig.update_layout(template='plotly_white', paper_bgcolor='white', font={'family': 'Arial'},
                      margin={'l': 55, 'r': 25, 't': 65, 'b': 60}, yaxis_title=ytitle,
                      legend_title_text='', hovermode='closest')
    return fig


@app.callback(Output('coverage', 'figure'), Output('eda-summary', 'children'),
              Output('quantiles', 'figure'), Output('quarters', 'figure'), Output('acf', 'figure'),
              Input('symbol', 'value'))
def update_eda(symbol):
    row = summary.set_index('symbol').loc[symbol]
    cov = px.bar(coverage, x='month', y=symbol, title=f'{symbol}: registros por mes')
    cov.update_xaxes(title='Mes (UTC)')
    quant = go.Figure(go.Bar(x=['Mínimo', 'Q1', 'Mediana', 'Q3', 'P95', 'P99', 'Máximo'],
                            y=[row[k] for k in ['min', 'q1', 'median', 'q3', 'p95', 'p99', 'max']],
                            marker_color='#177e89'))
    quant.update_layout(title=f'{symbol}: cuantiles del objetivo')
    q = quarters[quarters.symbol == symbol].sort_values('quarter')
    qfig = px.line(q, x='quarter', y=['mean', 'median'], markers=True,
                   title=f'{symbol}: volatilidad por trimestre', labels={'quarter': 'Trimestre', 'variable': 'Estadístico'})
    a = acf[acf.symbol == symbol].sort_values('lag_hours')
    afig = px.line(a, x='lag_hours', y='target_acf', title=f'{symbol}: autocorrelación del objetivo',
                   labels={'lag_hours': 'Rezago (horas)'})
    afig.add_hline(y=0, line_color='#aaa')
    message = f"{int(row['n']):,} etiquetas válidas · Mediana: {row['median']:.3f}% · P99: {row['p99']:.3f}%"
    return style(cov, 'Registros'), message, style(quant, 'Volatilidad (%)'), style(qfig, 'Volatilidad (%)'), style(afig, 'ACF')


@app.callback(Output('comparison', 'figure'), Output('folds', 'figure'), Output('model-summary', 'children'),
              Input('symbol', 'value'), Input('fold', 'value'), Input('metric', 'value'))
def update_models(symbol, fold, metric):
    all_rows = metrics[metrics.symbol == symbol].copy()
    selected = all_rows if fold == 'all' else all_rows[all_rows.fold == int(fold)]
    means = selected.groupby('model', as_index=False)[metric].mean()
    labels = {'rmse': 'RMSE (pp)', 'mae': 'MAE (pp)', 'mape': 'MAPE (%)', 'r2': 'R²'}
    names = {'persistence': 'Persistence', 'svr': 'SVR lineal'}
    means['Modelo'] = means.model.map(names)
    all_rows['Modelo'] = all_rows.model.map(names)
    title = 'Media de 5 folds' if fold == 'all' else f'Fold {fold}'
    comp = px.bar(means, x='Modelo', y=metric, color='Modelo', title=f'{symbol} · {title}', text_auto='.3f')
    trend = px.line(all_rows.sort_values('fold'), x='fold', y=metric, color='Modelo', markers=True,
                    title=f'{symbol}: detalle de los cinco folds (independiente del filtro de fold)')
    trend.update_xaxes(dtick=1, title='Fold cronológico')
    values = means.set_index('model')[metric]
    winner = values.idxmax() if metric == 'r2' else values.idxmin()
    tied = values.nunique() == 1
    msg = ('Empate' if tied else f'Mejor resultado: {names[winner]}') + f' · {labels[metric]} · {title} · {symbol}'
    return style(comp, labels[metric]), style(trend, labels[metric]), msg


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 8050)), debug=False)
