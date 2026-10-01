"""Dashboard for the completed 2023–2025 minute experiment."""
from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html


def create_progress_app():
    root=Path(__file__).resolve().parent/'results/minute_2023_2025'
    app=Dash(__name__,title='Binance 2023–2025 | Entrenamiento')
    app.layout=html.Main([
        html.H1('Estudio Binance 2023–2025'),
        html.P('Entradas de un minuto · BTC, ETH, BNB y XRP · SVR lineal'),
        html.P('2023 entrenamiento · 2024 validación · 2025 evaluación retrospectiva.'),
        html.P('Se prueban entradas de 7, 14, 21 y 28 días y cuatro ventanas de volatilidad. '
               'Cada pronóstico contiene siete volatilidades diarias.'),
        html.H2('Entrenamiento en curso'),html.Div(id='progress'),
        html.P('Las métricas finales se mostrarán tras completar y verificar el experimento.'),
        dcc.Interval(id='tick',interval=30000,n_intervals=0),
    ],style={'maxWidth':'1000px','margin':'auto','padding':'24px','fontFamily':'Arial'})

    @app.callback(Output('progress','children'),Input('tick','n_intervals'))
    def progress(_):
        paths=[root/'selection_all_inputs.csv',root.parent/'minute_2023_2025_parallel/selection_all_inputs.csv']
        frames=[pd.read_csv(path) for path in paths if path.exists()]
        if not frames: return 'Preparando datos y calendarios.'
        frame=max(frames,key=len)
        finished=frame.groupby('symbol').size()
        return [html.P(f'{len(frame)} de 192 combinaciones guardadas. El avance se guarda al terminar cada ventana de entrada.'),
                html.Ul([html.Li(f'{symbol}: {count} de 48 combinaciones') for symbol,count in finished.items()])]
    return app


def create_app():
    root = Path(__file__).resolve().parent/'results/minute_2023_2025'
    metrics = pd.read_csv(root/'all_metrics.csv')
    horizons = pd.read_csv(root/'horizon_metrics.csv')
    macro = pd.read_csv(root/'macro_metrics.csv')
    choices = pd.read_csv(root/'selected_inputs.csv')
    app = Dash(__name__, title='SVR | Binance 1 minuto | 2023–2025')
    app.layout = html.Main([
        html.H1('Volatilidad: SVR lineal con entradas de un minuto'),
        html.P('2023: entrenamiento · 2024: validación · 2025: evaluación retrospectiva.'),
        html.P('Cada entrada conserva todos los minutos de 7, 14, 21 o 28 días. '
               'Un origen diario; siete salidas de volatilidad de retornos diarios.'),
        dcc.Dropdown(['GroupKFold','KFold','ForwardChaining'],'ForwardChaining',id='method',clearable=False),
        dcc.Dropdown([{'label':'Fechas comunes entre métodos','value':'common_test'},
                      {'label':'Todas las fechas de cada método','value':'all_available_test'}],
                     'all_available_test',id='scope',clearable=False),
        html.H2('R² por criptomoneda y global'),html.Div(id='macro'),
        dcc.Dropdown(list(metrics.symbol.unique()),'BTCUSDT',id='symbol',clearable=False),
        html.Label('Ventana de volatilidad (días)'),
        dcc.Dropdown([7,14,21,28],7,id='volatility',clearable=False),
        html.Label('Ventana de entrada (días)'),
        dcc.Dropdown([7,14,21,28],7,id='lag',clearable=False),
        html.Div(id='summary'),dcc.Graph(id='windows'),dcc.Graph(id='horizons'),
        html.P('Los adaptadores cronológicos de KFold y ForwardChaining producen las mismas muestras. '
               'Sus cinco particiones de evaluación comparten entrenamiento; no son cinco ajustes temporales independientes. '
               'Compare métodos en fechas comunes. Las ventanas se seleccionan por RMSE de validación.'),
    ],style={'maxWidth':'1100px','margin':'auto','padding':'24px','fontFamily':'Arial'})

    @app.callback(Output('windows','figure'),Output('horizons','figure'),Output('summary','children'),Output('macro','children'),
                  Input('method','value'),Input('scope','value'),Input('symbol','value'),
                  Input('volatility','value'),Input('lag','value'))
    def update(method,scope,symbol,volatility,lag):
        query='method == @method and scope == @scope and symbol == @symbol and volatility_window == @volatility'
        data=metrics.query(query)
        h=horizons.query(query+' and input_window == @lag')
        selected=data[data.input_window==lag]
        best=int(choices.query('method == @method and symbol == @symbol and volatility_window == @volatility').input_window.iloc[0])
        summary=' | '.join(f'{r.model}: R²={r.r2:.4f}, RMSE={r.rmse:.4f}, MAE={r.mae:.4f}, n={r.n_origins}'
                           for r in selected.itertuples())
        overview=macro.query('method == @method and scope == @scope').pivot(index='symbol',columns='model',values='r2')
        table=html.Table([html.Thead(html.Tr([html.Th('Cripto'),html.Th('SVR'),html.Th('Persistencia')]))]+[
            html.Tr([html.Td(name),html.Td(f'{row.SVR:.4f}'),html.Td(f'{row.Persistence:.4f}')])
            for name,row in overview.iterrows()],style={'width':'100%','textAlign':'left'})
        return (px.bar(data,x='input_window',y='rmse',color='model',barmode='group',title='RMSE por ventana de entrada'),
                px.line(h,x='horizon',y='rmse',color='model',markers=True,title='RMSE por horizonte diario'),
                f'Entrada seleccionada en validación: {best} días. Mostrando {lag*1440:,} cierres por entrada. '+summary,
                [table,html.P('Promedio macro de las cuatro definiciones de volatilidad; entradas seleccionadas en validación. '
                              'El global promedia las 16 selecciones. Esta tabla no cambia con los filtros individuales inferiores.')])
    return app
