"""Run with python -m volatility_dashboard.app; no experiment is trained."""
from pathlib import Path
import logging
import os
from dash import Dash, Input, Output, dcc, html
from flask import send_from_directory
from .data_loader import ROOT, repository
from .tabs import contexto, eda, modelos

LOGGER = logging.getLogger(__name__)


def create_app():
    # Graphs and tables arrive through tab callbacks. Load their JavaScript
    # with the initial page so rendering does not depend on lazy chunk requests.
    app = Dash(__name__,title='Volatilidad | Machine Learning',assets_folder=str(Path(__file__).parent/'assets'),suppress_callback_exceptions=True,eager_loading=True)
    app.layout = html.Div([
        html.Header([html.Div('ML / VOLATILIDAD',className='brand'),html.H1('Predicción de la volatilidad futura de criptomonedas mediante Machine Learning'),html.P('Binance · cierres de 1 minuto · seis modelos clásicos + mejora HAR-Ridge y XGBoost · evaluación temporal retrospectiva')],className='site-header'),
        dcc.Tabs(id='main-tabs',value='contexto',children=[dcc.Tab(label='1 · Contexto del problema',value='contexto'),dcc.Tab(label='2 · EDA',value='eda'),dcc.Tab(label='3 · Comparación de modelos',value='modelos')]),
        html.Main(dcc.Loading(html.Div(id='tab-content'),type='circle'),className='main-content'),
        html.Footer('Resultados locales verificados · Unidades y protocolos explícitos · Sin entrenamiento desde el dashboard')])

    @app.callback(Output('tab-content','children'),Input('main-tabs','value'))
    def render_tab(tab):
        try:
            return {'contexto':contexto.layout,'eda':eda.layout,'modelos':modelos.layout}[tab]()
        except (OSError,ValueError,KeyError,IndexError,TypeError):
            LOGGER.exception('No se pudo cargar la pestaña %s.', tab)
            return html.Div([html.H2('Datos no disponibles'),html.P('Vuelve a seleccionar la pestaña para intentarlo de nuevo.')],className='notice')

    eda.register(app)
    modelos.register(app)

    @app.server.route('/healthz')
    def healthz():
        return {'status': 'ok'}, 200

    @app.server.route('/book-report')
    def book_report():
        return send_from_directory(str(ROOT/'book/sections'),'10_dashboard_comparativo.md',as_attachment=False,mimetype='text/plain; charset=utf-8')

    return app


app = create_app()
server = app.server

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=int(os.environ.get('PORT','8050')),debug=False)
