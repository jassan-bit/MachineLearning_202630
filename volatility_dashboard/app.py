"""Run with python -m volatility_dashboard.app; no experiment is trained."""
from pathlib import Path
import json
import logging
import os
from urllib.parse import parse_qs
from dash import Dash, Input, Output, dcc, html
from flask import abort, redirect, send_from_directory, url_for
from .data_loader import ROOT, repository
from .tabs import contexto, eda, modelos, revision_final

LOGGER = logging.getLogger(__name__)
FINAL_REVIEW_BUNDLE = ROOT / 'dashboard_data/final_review_20261018'


def create_app():
    # Graphs and tables arrive through tab callbacks. Load their JavaScript
    # with the initial page so rendering does not depend on lazy chunk requests.
    app = Dash(__name__,title='Volatilidad | Machine Learning',assets_folder=str(Path(__file__).parent/'assets'),suppress_callback_exceptions=True,eager_loading=True)
    app.layout = html.Div([
        dcc.Location(id='page-location', refresh=False),
        html.Header([html.Div('ML / VOLATILIDAD',className='brand'),html.H1('Predicción de la volatilidad futura de criptomonedas mediante Machine Learning'),html.P('Binance · cierres de 1 minuto · volatilidad diaria · modelos clásicos y HAR-Ridge + XGBoost · evaluación retrospectiva'),
            html.P(['Actualización del 18 de octubre de 2026: ', html.A('optimización y revisión final', href='/?tab=revision-final', style={'color':'#c6edff','fontWeight':'bold'}), '. El HAR original continúa vigente.'])],className='site-header'),
        dcc.Tabs(id='main-tabs',value='contexto',children=[dcc.Tab(label='1 · Contexto del problema',value='contexto'),dcc.Tab(label='2 · EDA',value='eda'),dcc.Tab(label='3 · Comparación de modelos',value='modelos'),dcc.Tab(label='4 · Optimización y revisión final',value='revision-final')]),
        html.Main(dcc.Loading(html.Div(id='tab-content'),type='circle'),className='main-content'),
        html.Footer([
            html.Div('By Jassan Arteta y Mateo Bernal'),
            html.Div('© Copyright 2026.')])])

    @app.callback(Output('tab-content','children'),Input('main-tabs','value'))
    def render_tab(tab):
        try:
            return {'contexto':contexto.layout,'eda':eda.layout,'modelos':modelos.layout,'revision-final':revision_final.layout}[tab]()
        except (OSError,ValueError,KeyError,IndexError,TypeError):
            LOGGER.exception('No se pudo cargar la pestaña %s.', tab)
            return html.Div([html.H2('Datos no disponibles'),html.P('Vuelve a seleccionar la pestaña para intentarlo de nuevo.')],className='notice')

    eda.register(app)
    modelos.register(app)
    revision_final.register(app)

    @app.callback(Output('main-tabs', 'value'), Input('page-location', 'search'))
    def select_linked_tab(search):
        requested = parse_qs((search or '').lstrip('?')).get('tab', ['contexto'])[0]
        return requested if requested in {'contexto', 'eda', 'modelos', 'revision-final'} else 'contexto'

    @app.server.route('/healthz')
    def healthz():
        return {'status': 'ok'}, 200

    @app.server.route('/review-release.json')
    def review_release():
        path = FINAL_REVIEW_BUNDLE / 'release.json'
        if not path.is_file():
            return {'status': 'unavailable'}, 404
        release = json.loads(path.read_text(encoding='utf-8'))
        release['deployed_commit'] = os.environ.get('RENDER_GIT_COMMIT', '')
        return release, 200

    @app.server.route('/final-review/')
    def final_review_index():
        return redirect(url_for('final_review_asset', filename='delivery/final_regression_local_20261018/index.html'))

    @app.server.route('/final-review/<path:filename>')
    def final_review_asset(filename):
        try:
            manifest = json.loads((FINAL_REVIEW_BUNDLE / 'bundle_manifest.json').read_text(encoding='utf-8'))
            public_root = (FINAL_REVIEW_BUNDLE / 'files').resolve()
            target = (public_root / filename).resolve()
            if filename not in manifest['files'] or not target.is_relative_to(public_root):
                abort(404)
        except (OSError, ValueError, KeyError, TypeError):
            abort(404)
        return send_from_directory(str(public_root), filename,
                                   mimetype='text/csv' if target.suffix.lower() == '.csv' else None)

    @app.server.route('/book-report')
    def book_report():
        return send_from_directory(str(ROOT/'book/sections'),'10_dashboard_comparativo.md',as_attachment=False,mimetype='text/plain; charset=utf-8')

    @app.server.route('/confidence-intervals/', defaults={'filename': 'report.html'})
    @app.server.route('/confidence-intervals/<path:filename>')
    def confidence_report(filename):
        return send_from_directory(
            str(ROOT/'results/har_original_ml_confidence_20261006'), filename)

    return app


app = create_app()
server = app.server

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=int(os.environ.get('PORT','8050')),debug=False)
