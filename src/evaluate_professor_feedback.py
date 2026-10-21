"""Read-only evaluation of local feedback corrections; output a reviewable HTML.

The comparison family/rule is in evaluation_protocol.json, saved before reading
the new AR test results. Never fits a model, edits original inputs or publishes.
"""
from pathlib import Path
from html import escape
from datetime import datetime, timezone
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_har_spca_experiment import FAMILIES, holm, sha
from volatility_dashboard.metrics import metric_table
from volatility_dashboard.diebold_mariano import dm_test
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

OUT = ROOT / 'results/professor_feedback_20261018'
KEYS = ['symbol', 'volatility_window', 'origin', 'horizon']
FIELDS = ['model'] + KEYS + ['actual', 'forecast']
DAILY_HAR = 'HAR diario OLS (1/7/28)'
BASE = 'HAR-Ridge + XGBoost'
AR = 'HAR-XG + AR mature errors'
LAST = 'HAR-XG + last mature error'
WEIGHTED = 'HAR-XG Ridge ponderado'
PREVIOUS = 'HAR-Ridge + XGBoost residual'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def normalized(frame, year, label=None):
    frame = frame.copy()
    if label is not None:
        frame['model'] = label
    if 'forecast' not in frame:
        frame = frame.rename(columns={c: 'forecast' for c in ('svr', 'knn') if c in frame})
    if not set(FIELDS).issubset(frame) or frame[FIELDS].isna().any().any():
        raise ValueError('Missing forecast fields or values.')
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    if 'target_date' in frame:
        target_date = pd.to_datetime(frame.target_date, utc=True)
        if not target_date.equals(frame.origin + pd.to_timedelta(frame.horizon, unit='D')):
            raise ValueError('Target date does not match origin+horizon.')
    frame = frame[FIELDS].copy()
    if not frame.origin.dt.year.eq(year).all():
        raise ValueError('Unexpected forecast year.')
    if (not np.isfinite(frame[['actual', 'forecast']]).all().all()
            or (frame[['actual', 'forecast']] < 0).any().any()
            or frame.duplicated(['model'] + KEYS).any()):
        raise ValueError('Invalid, negative or duplicate forecasts.')
    n = {2025: 358, 2026: 236}[year]
    dates = pd.date_range(f'{year}-01-01', periods=n, freq='D', tz='UTC')
    expected = pd.MultiIndex.from_product([['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT'],
        [7, 14, 21, 28], dates, range(1, 8)], names=KEYS).sort_values()
    for model, group in frame.groupby('model'):
        if not group.set_index(KEYS).sort_index().index.equals(expected):
            raise ValueError(f'Unequal/incomplete evaluation calendar: {model}')
    return frame


def load_period(year):
    paths = []
    frames = []
    if year == 2025:
        for model, folder in FAMILIES.items():
            path = ROOT / 'results' / folder / 'predictions.csv.gz'
            paths.append(path)
            frames.append(normalized(pd.read_csv(path), year, model))
    else:
        path = ROOT / 'results/har_spca_research_20261005/baseline_2026/predictions.csv.gz'
        paths.append(path)
        frames.append(normalized(pd.read_csv(path), year))
    for folder, labels in (
        (OUT / 'har_baseline', None), (OUT / 'daily_har', None),
        (OUT / 'residuals', None),
        (ROOT / 'results/har_xg_optimization_20261005', [WEIGHTED]),
        (ROOT / 'results/har_residual_xgboost_research_20261006', [PREVIOUS])):
        path = folder / f'predictions_{year}.csv.gz'
        paths.append(path)
        frame = pd.read_csv(path)
        if labels:
            frame = frame[frame.model.isin(labels)]
        if folder == OUT / 'residuals':
            frame = frame[~frame.model.eq(BASE)]  # original independently loaded above
        frames.append(normalized(frame, year))
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(['model'] + KEYS).any():
        raise ValueError('Duplicate model forecasts across sources.')
    reference = frame[frame.model.eq(BASE)].set_index(KEYS).actual.sort_index()
    audit = []
    for model, group in frame.groupby('model'):
        observed = group.set_index(KEYS).actual.sort_index()
        if not observed.index.equals(reference.index):
            raise ValueError('Keys differ across models.')
        np.testing.assert_allclose(observed, reference, rtol=1e-12, atol=1e-12)
        audit.append(dict(year=year, model=model, rows=len(group),
                          max_abs_actual_error=float(np.max(np.abs(observed-reference)))))
    return frame, paths, audit


def daily_losses(frame):
    frame = frame.assign(loss=(frame.actual - frame.forecast) ** 2)
    return frame.groupby(['origin', 'model']).loss.mean().unstack('model').sort_index()


def comparison_family(daily, candidate, references, lags):
    rows = []
    for reference in references:
        test = dm_test((daily[candidate] - daily[reference]).to_numpy(), horizon=7, lags=lags)
        rows.append(dict(candidate=candidate, reference=reference,
            mse_candidate=float(daily[candidate].mean()), mse_reference=float(daily[reference].mean()),
            reduction_percent=float(100 * (1 - daily[candidate].mean() / daily[reference].mean())), **test))
    adjusted = holm([r['p'] for r in rows], len(rows))
    for row, p in zip(rows, adjusted):
        row['p_holm'] = float(p)
        row['favorable_significant'] = bool(p < .05 and row['diferencia_mse'] < 0)
    return pd.DataFrame(rows)


def residual_diagnostics(frame, year):
    rows = []
    selected = frame[frame.model.isin([BASE, AR, LAST, 'HAR-XG + AR no online labels'])]
    for key, group in selected.groupby(['model', 'symbol', 'volatility_window', 'horizon']):
        group = group.sort_values('origin')
        error = (group.actual - group.forecast).to_numpy()
        coefficients = acf(error, nlags=28, fft=False)
        ljung = acorr_ljungbox(error, lags=[7, 14, 28], return_df=True)
        rows.append(dict(zip(['model', 'symbol', 'volatility_window', 'horizon'], key),
            year=year, n_origins=len(error), acf1=float(coefficients[1]),
            acf7=float(coefficients[7]), acf14=float(coefficients[14]), acf28=float(coefficients[28]),
            ljung_box7_p_raw=float(ljung.loc[7, 'lb_pvalue']),
            ljung_box14_p_raw=float(ljung.loc[14, 'lb_pvalue']),
            ljung_box28_p_raw=float(ljung.loc[28, 'lb_pvalue'])))
    return pd.DataFrame(rows)


def table(frame):
    return frame.to_html(index=False, border=0, escape=True,
                         float_format=lambda x: f'{x:.6g}')


def verify_freeze(folder):
    freeze = read_json(folder / 'selection_freeze.json')
    if freeze.get('test_forecasts_or_metrics_computed') is not False:
        raise ValueError(f'No pre-test freeze in {folder}.')
    if sha(folder / 'protocol.json') != freeze.get('protocol_sha256'):
        raise ValueError('Protocol changed after training freeze.')
    if 'selected_inputs_sha256' in freeze and sha(folder / 'selected_inputs.csv') != freeze['selected_inputs_sha256']:
        raise ValueError('Model choices changed after freeze.')
    if 'training_support_sha256' in freeze and sha(folder / 'training_support.csv') != freeze['training_support_sha256']:
        raise ValueError('Training support changed after freeze.')
    model_hashes = freeze.get('model_sha256', {})
    expected = {p.name for p in (folder / 'models').glob('*.joblib')}
    if len(model_hashes) not in (16, 32) or set(model_hashes) != expected:
        raise ValueError('Frozen model inventory differs from saved files.')
    for name, digest in freeze.get('model_sha256', {}).items():
        if sha(folder / 'models' / name) != digest:
            raise ValueError('Frozen model changed.')
    return freeze


def run():
    protocol = read_json(OUT / 'evaluation_protocol.json')
    if (OUT / 'summary.json').exists():
        raise RuntimeError('Feedback report already exists; inspect without replacing.')
    diag = read_json(OUT / 'diagnostics/metadata.json')
    if diag.get('status') != 'complete':
        raise ValueError('Development diagnostics are incomplete.')
    if read_json(OUT / 'residuals/summary.json').get('status') != 'complete':
        raise ValueError('Residual experiment is incomplete.')
    freezes = {folder: verify_freeze(OUT / folder) for folder in ('har_baseline', 'daily_har', 'residuals')}
    before = read_json(OUT / 'har_baseline/protected_before.json')
    if any(sha(ROOT / path) != digest for path, digest in before.items()):
        raise ValueError('Protected original dataset, model, dashboard or presentation changed.')
    all_metrics, all_windows, all_dm, base_dm, audits, inputs = [], [], [], [], [], {}
    all_residuals = []
    for year in (2025, 2026):
        frame, paths, audit = load_period(year)
        audits.extend(audit)
        all_residuals.append(residual_diagnostics(frame, year))
        inputs.update({p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
        horizon = metric_table(frame)
        horizon['year'] = year
        horizon.to_csv(OUT / f'horizon_metrics_{year}.csv', index=False)
        macro = horizon.groupby('model', as_index=False)[['mse', 'rmse', 'mae', 'r2']].mean().assign(year=year)
        all_metrics.append(macro)
        all_windows.append(horizon.groupby(['volatility_window', 'model'], as_index=False)[
            ['mse', 'rmse', 'mae', 'r2']].mean().assign(year=year))
        daily = daily_losses(frame)
        daily.to_csv(OUT / f'daily_loss_{year}.csv')
        for lags in protocol['sensitivity_hac_lags']:
            all_dm.append(comparison_family(daily, AR, protocol['primary_references'], lags).assign(
                year=year, scope='global', hac=lags))
        base_dm.append(comparison_family(daily, 'SVR Lineal', ['Persistencia', DAILY_HAR], 33).assign(year=year))
        for window in (7, 14, 21, 28):
            all_dm.append(comparison_family(daily_losses(frame[frame.volatility_window.eq(window)]),
                AR, protocol['primary_references'], window+5).assign(year=year, scope=str(window), hac=window+5))
    metrics = pd.concat(all_metrics, ignore_index=True)
    windows = pd.concat(all_windows, ignore_index=True)
    dm = pd.concat(all_dm, ignore_index=True)
    baseline_dm = pd.concat(base_dm, ignore_index=True)
    residuals = pd.concat(all_residuals, ignore_index=True)
    residuals.to_csv(OUT / 'residual_diagnostics.csv', index=False)
    residual_summary = residuals.groupby(['year', 'model'], as_index=False).agg(
        median_acf1=('acf1', 'median'), median_acf7=('acf7', 'median'),
        median_acf14=('acf14', 'median'),
        fraction_lb28_raw_p_below05=('ljung_box28_p_raw', lambda p: float((p < .05).mean())))
    residual_summary.to_csv(OUT / 'residual_diagnostics_summary.csv', index=False)
    metrics.to_csv(OUT / 'macro_metrics.csv', index=False)
    windows.to_csv(OUT / 'metrics_by_window.csv', index=False)
    dm.to_csv(OUT / 'dm_ar_holm4.csv', index=False)
    baseline_dm.to_csv(OUT / 'dm_svr_baselines_holm2.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'alignment_checks.csv', index=False)
    primary = dm[(dm.year == 2026) & dm.scope.eq('global') & (dm.hac == 33)]
    macro2025 = metrics[metrics.year.eq(2025)].set_index('model')
    success = bool(primary.favorable_significant.all() and (primary.reduction_percent > 0).all()
        and macro2025.loc[AR, 'mse'] < macro2025.loc[BASE, 'mse'])
    summary = dict(status='complete', evaluation_at_utc=datetime.now(timezone.utc).isoformat(),
        publication='LOCAL ONLY', github_or_render_updated=False, strong_retrospective_rule_passed=success,
        untouched_confirmatory_test_available=False, protected_files_unchanged=len(before),
        evaluation_protocol_sha256=sha(OUT / 'evaluation_protocol.json'), input_sha256=inputs,
        primary2026=primary.to_dict(orient='records'), macro_metrics=metrics.to_dict(orient='records'))
    labels = ['Persistencia', DAILY_HAR, 'SVR Lineal', BASE, LAST, AR]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharex=True)
    for year, ax in zip((2025, 2026), axes):
        part = metrics[metrics.year.eq(year)].set_index('model').loc[labels]
        ax.barh(labels, part.mse, color=['#8b97a3']*4+['#bc8120', '#087f8c'])
        ax.invert_yaxis()
        ax.set_title(f'{year}: MSE global, evaluación retrospectiva')
        ax.set_xlabel('Puntos porcentuales de volatilidad al cuadrado')
        ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / 'mse_comparison.png', dpi=160)
    fig.savefig(OUT / 'mse_comparison.svg')
    plt.close(fig)
    verdict = ('Se cumple la regla de mejora retrospectiva fijada antes de evaluar.' if success else
               'No se cumple la regla de mejora retrospectiva fijada antes de evaluar.')
    content = ['<!doctype html><html lang="es"><meta charset="utf-8"><title>Correcciones locales del proyecto</title>',
        '<style>body{font:17px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#17324d}'
        'table{border-collapse:collapse;font-size:13px;width:100%;margin:15px 0}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}'
        'img{width:100%}a{color:#087f8c}.status{background:#edf4f7;padding:18px}pre{white-space:pre-wrap;font-size:13px}</style>',
        '<h1>Correcciones al comentario del profesor</h1><p>18 de octubre de 2026 · resultados locales</p>',
        f'<div class="status"><b>{escape(verdict)}</b> GitHub y Render permanecen sin cambios. '
        'El modelo y la presentación vigentes se conservan. Ambos años ya habían sido explorados; '
        'ningún resultado de este informe constituye confirmación en un periodo nuevo.</div>',
        '<h2>Qué se corrigió</h2><ul><li>Información mutua marginal con objetivos absolutos y correcciones relativas; '
        'VIF sobre las seis variables diarias y las 44 entradas HAR reales. Desarrollo2023–2024 y2023 por separado.</li>'
        '<li>Perfiles intradía UTC por minuto y hora, comparación2023/2024 y bootstrap por bloques de días.</li>'
        '<li>HAR-OLS diario con tres escalas y referencia HAR intradía, además de persistencia.</li>'
        '<li>Corrección AR por horizonte usando únicamente errores ya observables; control con último residuo maduro '
        'y control sin acceso a nuevas etiquetas.</li></ul>',
        '<h2>Resultados de pronóstico</h2><p>Mismo objetivo y fechas por modelo:358 orígenes2025 y236 en2026 '
        '(1 de enero–24 de agosto). Cada origen aporta una pérdida media sobre112 salidas dependientes. '
        'RMSE macro es promedio de RMSE por activo/ventana/horizonte; no equivale a raíz del MSE global.</p>',
        '<img src="mse_comparison.svg" alt="MSE de referencias, modelo vigente y corrección AR">']
    for year in (2026, 2025):
        content += [f'<h3>{year}: métricas macro</h3>', table(metrics[metrics.year.eq(year)].sort_values('mse')),
            '<h3>Corrección AR frente a las cuatro referencias predeclaradas</h3>',
            table(dm[(dm.year == year) & dm.scope.eq('global') & (dm.hac == 33)][
                ['reference', 'mse_candidate', 'mse_reference', 'reduction_percent', 'dm', 'p', 'p_holm', 'favorable_significant']])]
    content += ['<p>Contrastes bilaterales DM-HLN7, HAC Bartlett33 y Holm4 por año;2026 es primario. '
        'Sensibilidades14/56 y tablas por ventana en los CSV. La regla requiere ventaja significativa '
        'frente a las cuatro referencias en2026 y menor MSE frente al vigente en2025. '
        'No rechazar una diferencia no demuestra equivalencia.</p>',
        '<h2>Autocorrelación residual antes y después</h2>', table(residual_summary),
        '<p>Coeficientes ACF por serie y Ljung–Box descriptivo con28 rezagos. La fracción usa valores p '
        'individuales sin ajuste y resume112 series por modelo/año; no es una prueba global. Las ventanas '
        'superpuestas inducen dependencia: una reducción de MSE no demuestra residuos blancos. '
        'Coeficientes y pruebas por serie están en residual_diagnostics.csv.</p>',
        '<h2>Referencia HAR independiente</h2><p>El HAR diario usa la volatilidad actual del objetivo y sus medias '
        'de7/28 días, con intercepto y OLS por horizonte. Su historia más larga exige excluir del train '
        'los orígenes iniciales sin entradas completas; conserva todas las fechas de evaluación. '
        'El segundo HAR usa la raíz de variación intradía y sus medias7/28. Son adaptaciones al objetivo '
        'del estudio, no reproducciones exactas del objetivo de volatilidad realizada de Corsi.</p>',
        table(baseline_dm[['year', 'reference', 'reduction_percent', 'p', 'p_holm', 'favorable_significant']]),
        '<h2>Diagnósticos de desarrollo</h2><p>Información mutua y VIF describen asociación y colinealidad; '
        'no prueban utilidad incremental. Los valores MI no son pruebas de significación y no se utilizaron '
        'para escoger entradas. Las constantes y dependencias exactas se informan sin sustituirlas '
        'por VIF aparentemente finitos.</p><p><a href="diagnostics/metadata.json">Resumen y rutas del diagnóstico</a></p>',
        '<img src="diagnostics/intraday_development.png" alt="Perfiles horarios UTC de desarrollo por activo">',
        '<details><summary>Resumen completo de diagnóstico</summary><pre>' + escape(json.dumps(diag, ensure_ascii=False, indent=2)) + '</pre></details>',
        '<h2>Interpretación temporal y límites</h2><p>La corrección usa e[s,h] solo cuando s+h≤t, después del cierre '
        'de t; las últimas etiquetas aún no observadas se excluyen. Sus coeficientes quedan fijos. '
        'Observar resultados maduros de prueba para actualizar el estado es una política online explícita, '
        'por lo que se compara con el control de último error maduro y con el control sin nuevas etiquetas.</p>'
        '<p>La validación AR es condicional a hiperparámetros base ya seleccionados sobre todo2024. '
        'No es una validación anidada independiente. El EDA se reconstruyó con desarrollo únicamente, '
        'pero fue añadido después del modelado original: esa cronología no puede deshacerse. '
        'También2026 ya se exploró en varios experimentos previos. Hace falta un periodo posterior '
        'sin inspección y una política congelada para una confirmación prospectiva.</p>',
        '<h2>Reproducir y revisar</h2><p>Usar Python del entorno .venv-repro. Los ejecutores rechazan '
        'sobrescribir resultados existentes; para repetir, elegir una carpeta nueva al modificar la constante OUT '
        'en una copia de los scripts. No ejecutar publicadores.</p><pre>'
        '.venv-repro/Scripts/python.exe src/professor_feedback_diagnostics.py\n'
        '.venv-repro/Scripts/python.exe src/professor_feedback_har_baseline.py\n'
        '.venv-repro/Scripts/python.exe src/professor_feedback_daily_har.py\n'
        '.venv-repro/Scripts/python.exe src/professor_feedback_residuals.py\n'
        '.venv-repro/Scripts/python.exe src/evaluate_professor_feedback.py</pre>',
        '<p><a href="evaluation_protocol.json">Protocolo de evaluación</a> · <a href="summary.json">Resumen y hashes</a> · '
        '<a href="macro_metrics.csv">Métricas CSV</a> · <a href="dm_ar_holm4.csv">DM y sensibilidades CSV</a> · '
        '<a href="alignment_checks.csv">Alineación de objetivos</a></p>',
        f'<p>Se verificaron {len(before)} archivos protegidos sin cambios, incluidos fuentes/datos/modelos '
        'originales, dashboard y presentación TeX. No hubo actualización remota.</p>',
        '<h2>Fuentes metodológicas</h2><ul>'
        '<li><a href="https://doi.org/10.1093/jjfinec/nbp001">Corsi2009: HAR</a></li>'
        '<li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html">scikit-learn: información mutua</a></li>'
        '<li><a href="https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html">statsmodels: VIF</a></li>'
        '<li><a href="https://doi.org/10.1016/S0927-5398(97)00004-2">Andersen y Bollerslev1997: periodicidad intradía</a></li>'
        '<li><a href="https://pkg.robjhyndman.com/forecast/reference/dm.test.html">DM, HLN y Bartlett</a></li></ul></html>']
    (OUT / 'report.html').write_text('\n'.join(content), encoding='utf-8')
    if any(sha(ROOT / p) != h for p, h in before.items()):
        raise ValueError('Protected inputs changed during evaluation.')
    if any(sha(ROOT / p) != h for p, h in inputs.items()):
        raise ValueError('Evaluation inputs changed during evaluation.')
    summary['analysis_code_sha256'] = sha(Path(__file__))
    summary['diagnostics_metadata_sha256'] = sha(OUT / 'diagnostics/metadata.json')
    summary['training_freeze_sha256'] = {folder: sha(OUT / folder / 'selection_freeze.json') for folder in freezes}
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(dict(strong_retrospective_rule_passed=success,
        report=str(OUT / 'report.html'), primary2026=summary['primary2026']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
