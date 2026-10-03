"""Verify and render a standalone local review. No publication operations."""
import hashlib
import json
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import nbformat
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from optimize_minute_svr import OUT, ROOT, DATA, features, minute_features, load_panel, calendars, estimator, predict
from metric_comparison import comparison_tables


def verify_and_report():
    status = json.loads((OUT/'status.json').read_text())
    assert status['status'] == 'complete'
    assert status['selection_sha256'] == hashlib.sha256((OUT/'selected_inputs.csv').read_bytes()).hexdigest()
    for name, digest in json.loads((ROOT/'results/minute_2023_2025/data_manifest.json').read_text()).items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest
    panel = load_panel()
    folds, eligible, audit = calendars(panel)
    search = pd.read_csv(OUT/'search.csv')
    choices = pd.read_csv(OUT/'selected_inputs.csv')
    assert len(search) == 640 and len(choices) == 16
    assert search.groupby(['symbol','volatility_window']).size().eq(40).all()
    assert not search.duplicated(['symbol','volatility_window','input_window','C','epsilon']).any()
    result = pd.read_csv(OUT/'selected_metrics.csv')
    pred = pd.read_csv(OUT/'predictions.csv.gz')
    oldpred = pd.read_csv(ROOT/'results/minute_2023_2025/predictions.csv.gz')
    oldpred = oldpred[oldpred.method == 'ForwardChaining']
    validation = []
    for symbol in panel.columns:
        minute = minute_features(np.load(DATA/f'{symbol}.npy'))
        for row in choices[choices.symbol == symbol].itertuples():
            options = search[(search.symbol == symbol) & (search.volatility_window == row.volatility_window)]
            assert np.isclose(row.validation_rmse, options.validation_rmse.min())
            X, y, base = features(panel[symbol], minute, row.input_window, row.volatility_window)
            artifact = joblib.load(OUT/'models'/f'{symbol}_v{row.volatility_window}.joblib')
            assert pd.Timestamp(artifact['fitted_through']) < pd.Timestamp('2025-01-01', tz='UTC')
            g = pred[(pred.symbol == symbol) & (pred.volatility_window == row.volatility_window)]
            assert not g.duplicated(['origin', 'horizon']).any()
            actual = g.pivot(index='origin', columns='horizon', values='actual')
            anchors = panel.index.get_indexer(pd.to_datetime(actual.index, utc=True))
            assert (anchors >= 0).all()
            np.testing.assert_allclose(actual, y[anchors])
            p = g.pivot(index='origin', columns='horizon', values='svr').to_numpy()
            np.testing.assert_allclose(p, predict(artifact['model'], X[anchors], base[anchors]), atol=1e-10)
            b = g.pivot(index='origin', columns='horizon', values='persistence').to_numpy()
            np.testing.assert_allclose(b, np.repeat(base[anchors,None], 7, axis=1))
            reference = oldpred[(oldpred.symbol == symbol) & (oldpred.volatility_window == row.volatility_window)]
            reference = reference.drop_duplicates(['origin', 'horizon']).pivot(index='origin', columns='horizon', values='actual')
            pd.testing.assert_index_equal(reference.index, actual.index)
            np.testing.assert_allclose(reference, actual)
            for label, values in [('SVR_optimized', p), ('Persistence', b)]:
                saved = result[(result.symbol == symbol) & (result.volatility_window == row.volatility_window) & (result.model == label)].iloc[0]
                computed = [r2_score(actual, values), np.sqrt(mean_squared_error(actual, values, multioutput='raw_values')).mean(),
                            mean_absolute_error(actual, values), mean_squared_error(actual, values),
                            np.mean(100*np.abs((values-actual.to_numpy())/actual.to_numpy()))]
                np.testing.assert_allclose(saved[['r2','rmse','mae','mse','mape']].to_numpy(float), computed, rtol=1e-10)
            vy, vp, vb = [], [], []
            for tr, va in folds:
                m = make_pipeline(StandardScaler(), estimator(row.C, row.epsilon))
                m.fit(X[tr], y[tr]/base[tr,None]-1)
                vy.append(y[va]); vp.append(predict(m, X[va], base[va])); vb.append(np.repeat(base[va,None],7,axis=1))
            vy, vp, vb = map(np.vstack, [vy, vp, vb])
            score = np.sqrt(((vy-vp)**2).mean(axis=0)).mean()
            np.testing.assert_allclose(score, row.validation_rmse, rtol=1e-9)
            validation.append(dict(symbol=symbol, volatility_window=row.volatility_window,
                svr_rmse=score, persistence_rmse=np.sqrt(((vy-vb)**2).mean(axis=0)).mean()))
    pd.DataFrame(validation).to_csv(OUT/'validation_comparison.csv', index=False)
    macro = pd.read_csv(OUT/'macro_metrics.csv')
    old = pd.read_csv(ROOT/'results/minute_2023_2025/macro_metrics.csv')
    old = old[(old.method == 'ForwardChaining') & (old.scope == 'all_available_test') & (old.model == 'SVR')].copy()
    old['model'] = 'SVR_original'
    comparison = pd.concat([old[macro.columns], macro], ignore_index=True)
    comparison.to_csv(OUT/'comparison.csv', index=False)
    horizon_rows = []
    for (symbol, window, horizon), g in pred.groupby(['symbol','volatility_window','horizon']):
        for label, column in [('SVR_optimized','svr'),('Persistence','persistence')]:
            horizon_rows.append(dict(symbol=symbol, volatility_window=int(window), horizon=int(horizon), model=label,
                r2=r2_score(g.actual,g[column]), rmse=np.sqrt(mean_squared_error(g.actual,g[column])),
                mae=mean_absolute_error(g.actual,g[column])))
    pd.DataFrame(horizon_rows).to_csv(OUT/'horizon_metrics.csv', index=False)
    wins = result.pivot(index=['symbol','volatility_window'], columns='model', values='rmse')
    wins_count = int((wins.SVR_optimized < wins.Persistence).sum())
    figure = px.bar(macro[macro.symbol != 'GLOBAL_MACRO'], x='symbol', y='r2', color='model', barmode='group', title='R² de prueba: SVR optimizado frente a persistencia')
    detail = px.bar(result, x='volatility_window', y='rmse', color='model', facet_col='symbol', barmode='group', title='RMSE por definición de volatilidad')
    counts = pd.read_csv(OUT/'calendar.csv').groupby(['fold','split']).agg(n=('origin','size'), first=('origin','min'), last=('origin','max')).reset_index()
    html = '''<!doctype html><html lang="es"><meta charset="utf-8"><title>Revisión local: SVR optimizado</title>
    <style>body{font-family:Arial;max-width:1250px;margin:35px auto;padding:15px;color:#182b44}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:9px;border:1px solid #ddd;text-align:right}h1{color:#176a8e}p{line-height:1.6}</style>
    <h1>SVR lineal optimizado · revisión local</h1>
    <p>Binance: BTC, ETH, BNB y XRP · 2023–2025 · datos de un minuto · ventanas 7, 14, 21 y 28 días · siete salidas diarias.</p>
    <p>Esta variante usa características calculadas con todos los retornos por minuto: volatilidad realizada, variación absoluta, semivolatilidad negativa y salto máximo por día. Añade retornos diarios y la parte conocida de la ventana de volatilidad que permanece en cada horizonte. No introduce cada precio minuto como una columna. El objetivo sigue siendo la desviación estándar de retornos diarios, ddof=0, en puntos porcentuales y sin anualizar.</p>
    <p>El SVR aprende una corrección relativa a persistencia. Escaladores ajustados dentro de cada entrenamiento; salidas negativas truncadas a cero. Búsqueda: 4 entradas × 5 valores de C × 2 epsilon por activo y ventana objetivo. Selección por RMSE fuera de muestra en seis bloques de 2024. Reajuste final con etiquetas anteriores a 2025; modelo fijo durante la prueba.</p>
    <p>Forward Chaining conserva los cortes de entrenamiento de tsxv y extiende cada validación a dos meses; son seis entrenamientos crecientes diferentes. K-Fold y Group K-Fold nativos se auditan pero no se usan para seleccionar: incluyen futuro respecto a validación. No se presentan como equivalentes ni como resultados de forecasting válidos.</p>
    <p>Comparación retrospectiva: 2025 ya se había consultado. La selección nueva no usa sus métricas, pero no es una prueba inédita. El original se entrenó solo con 2023; el optimizado se reajusta con 2023–2024. La mejora conjunta no puede atribuirse exclusivamente a un parámetro o a la validación.</p>'''
    html += f'<p><b>El SVR optimizado supera a persistencia en RMSE en {wins_count}/16 configuraciones seleccionadas.</b> R² global: promedio de activos, ventanas y horizontes; no R² sobre precios mezclados.</p>'
    global_comparison, asset_comparison, interpretation = comparison_tables(macro)
    html += '<h2>Tabla comparativa global: persistencia y SVR lineal</h2>' + global_comparison.round(5).to_html(index=False)
    html += '<h2>Comparación por criptomoneda</h2>' + asset_comparison.round(5).to_html(index=False)
    html += '<h2>Interpretación de las métricas</h2>' + ''.join('<p>'+p+'</p>' for p in interpretation)
    html += '<h2>Referencia histórica: modelo original</h2>' + comparison.round(5).to_html(index=False)
    html += figure.to_html(full_html=False, include_plotlyjs=True)
    html += detail.to_html(full_html=False, include_plotlyjs=False)
    html += '<h2>Selección y validación 2024</h2>' + choices.round(6).to_html(index=False)
    html += pd.DataFrame(validation).round(5).to_html(index=False)
    html += '<h2>Cortes temporales</h2>' + counts.to_html(index=False)
    html += '<p>Archivos: <a href="comparison.csv">métricas comparadas</a> · <a href="selected_metrics.csv">16 configuraciones</a> · <a href="search.csv">búsqueda completa</a> · <a href="native_cv_audit.csv">auditoría tsxv</a></p>'
    html += '<p>Referencias técnicas: <a href="https://pypi.org/project/timeseries-cv/">timeseries-cv / Filipe Ramos</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html">LinearSVR</a>.</p></html>'
    (OUT/'index.html').write_text(html, encoding='utf-8')
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Optimización local del SVR\nDatos Binance de un minuto, 2023–2025. Ver el informe HTML para metodología y limitaciones. Esta variante usa características intradía resumidas; no reemplaza las columnas de precios del experimento original.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nroot = Path.cwd()\nif not (root/'results').exists(): root = root.parent\nout = root/'results/optimized_minute_2023_2025'\npd.read_csv(out/'comparison.csv')"),
        nbformat.v4.new_code_cell("pd.read_csv(out/'validation_comparison.csv')"),
        nbformat.v4.new_code_cell("pd.read_csv(out/'selected_inputs.csv')"),
        nbformat.v4.new_markdown_cell('Reproducir desde la raíz: `python src/optimize_minute_svr.py` y después `python src/report_optimized_svr.py`. El segundo comando verifica predicciones, selección, métricas, modelos y correspondencia con las fechas del estudio original. Ningún comando publica resultados.')])
    nbformat.write(notebook, ROOT/'notebooks/Optimizacion_SVR_Minuto.ipynb')
    verification = dict(passed=True, configurations=16, forecast_rows=len(pred),
        test_origins=status['test_origins'], wins_vs_persistence=wins_count,
        source_hashes_verified=True, exported_models_verified=True, metrics_recomputed=True,
        selection_refitted_on_2024=True, original_test_dates_and_targets_equal=True)
    (OUT/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    provenance = dict(versions={p:version(p) for p in ['numpy','pandas','scikit-learn','timeseries-cv']},
        source_sha256={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
                       for name in ['optimize_minute_svr.py','report_optimized_svr.py','minute_experiment.py','metric_comparison.py']},
        artifact_sha256={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in OUT.rglob('*') if p.is_file() and p.name != 'provenance.json'})
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify_and_report()
