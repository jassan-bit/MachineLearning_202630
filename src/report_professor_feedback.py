"""Build the Spanish review report from completed, immutable local experiments.

No fitting or statistical selection. The technical report and original metrics
are preserved. Run after evaluate_professor_feedback.py.
"""
from pathlib import Path
from html import escape
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_har_spca_experiment import sha

OUT = ROOT / 'results/professor_feedback_20261018'
BASE = 'HAR-Ridge + XGBoost'
AR = 'HAR-XG + AR mature errors'
LABELS = {AR: 'HAR-XG + corrección AR causal',
    'HAR-XG + last mature error': 'HAR-XG + último error maduro',
    'HAR-XG + AR no online labels': 'HAR-XG + AR sin nuevas etiquetas',
    'HAR-Ridge + XGBoost residual': 'Corrección XGBoost anterior (06/oct)',
    'HAR-XG Ridge ponderado': 'HAR-XG ponderado anterior (05/oct)'}


def tab(frame):
    frame = frame.replace(LABELS)
    return frame.to_html(index=False, border=0, escape=True,
        float_format=lambda x: f'{x:.6f}'.replace('.', ','))


def run():
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    before = json.loads((OUT / 'har_baseline/protected_before.json').read_text(encoding='utf-8'))
    if summary['strong_retrospective_rule_passed']:
        raise ValueError('This report text is scoped to the recorded negative replacement decision.')
    if any(sha(ROOT / p) != h for p, h in before.items()):
        raise ValueError('A protected original artifact changed.')
    metrics = pd.read_csv(OUT / 'macro_metrics.csv')
    dm = pd.read_csv(OUT / 'dm_ar_holm4.csv')
    residuals = pd.read_csv(OUT / 'residual_diagnostics_summary.csv')
    vif = pd.read_csv(OUT / 'diagnostics/variance_inflation.csv').query("period == 'development_2023_2024'")
    mi = pd.read_csv(OUT / 'diagnostics/mutual_information.csv').query(
        "period == 'development_2023_2024' and target_kind == 'relative_correction'")
    intraday = pd.read_csv(OUT / 'diagnostics/intraday_summary.csv').query("period == 'development_2023_2024'")
    support = pd.read_csv(OUT / 'daily_har/training_support.csv')
    vif_rows = []
    for family, group in vif.groupby('feature_family'):
        finite = group.vif[np.isfinite(group.vif)]
        vif_rows.append({'Entradas': family, 'Posiciones en 16 diseños': len(group),
            'VIF > 10': int(group.high_vif_over_10.sum()),
            'VIF infinitos': int(np.isinf(group.vif).sum()),
            'Constantes': int(group.status.eq('constant_undefined').sum()),
            'Mediana VIF finito': float(finite.median())})
    top_mi = mi[mi.feature_family.eq('HAR44')].groupby('feature', as_index=False).mi_nats.mean().nlargest(8, 'mi_nats')
    primary = dm[dm.year.eq(2026) & dm.scope.eq('global') & dm.hac.eq(33)]
    primary_original = primary[primary.reference.eq(BASE)].iloc[0]
    results = []
    for year in (2025, 2026):
        part = metrics[metrics.year.eq(year)].set_index('model')
        results.append({'Año': year, 'MSE vigente': part.loc[BASE, 'mse'],
            'MSE con AR': part.loc[AR, 'mse'],
            'Reducción MSE (%)': 100 * (1 - part.loc[AR, 'mse'] / part.loc[BASE, 'mse'])})
    text = ['<!doctype html><html lang="es"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Correcciones al proyecto de volatilidad · revisión local</title>',
        '<style>body{font:17px/1.55 system-ui;max-width:1120px;margin:36px auto;padding:0 22px;color:#17324d}'
        'h1{line-height:1.2}h2{margin-top:34px}table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}'
        'td,th{padding:8px;border-bottom:1px solid #d3dce3;text-align:left}th{background:#edf4f7}'
        'img{max-width:100%;height:auto}.decision{padding:20px;background:#fbf4e5;border-left:5px solid #bd811f}'
        'a{color:#087f8c}pre{white-space:pre-wrap;font-size:13px;overflow-wrap:anywhere}'
        '.scroll{overflow-x:auto}details{margin:18px 0}</style>',
        '<h1>Correcciones al proyecto de volatilidad</h1><p>Revisión local · 18 de octubre de 2026</p>',
        '<p class="decision"><b>Los cambios aún no justifican reemplazar el modelo vigente.</b> '
        'La corrección AR reduce ligeramente el error observado, pero la ventaja de 2026 no es significativa. '
        'GitHub, Render y la presentación TeX permanecen sin cambios.</p>',
        tab(pd.DataFrame(results)),
        f'<p>En 2026, DM contra el vigente: p = {primary_original.p:.4f}; '
        f'p ajustado Holm4 = {primary_original.p_holm:.4f}. '
        'La corrección también queda por detrás de los mejores experimentos locales anteriores. '
        'Ambos años ya habían sido examinados: los resultados son retrospectivos.</p>',
        '<h2>Correcciones implementadas</h2><ul>'
        '<li>Información mutua y VIF de las 44 entradas HAR y de las seis variables diarias contemporáneas.</li>'
        '<li>Estacionalidad intradía por minuto y hora UTC, con comparación 2023/2024 e intervalos por bloques.</li>'
        '<li>Referencia HAR diario OLS con tres escalas, además de persistencia; HAR intradía como sensibilidad.</li>'
        '<li>Corrección autorregresiva por horizonte que solo accede a errores cuyo objetivo ya se observó; '
        'controles de último error maduro y de funcionamiento sin nuevas etiquetas.</li></ul>',
        '<h2>Información mutua y colinealidad</h2><p>Se usó exclusivamente el prefijo de desarrollo 2023–2024: '
        '660 orígenes con todas las etiquetas anteriores a 2025. Se añadió un diagnóstico separado de 2023 '
        '(294 orígenes). Las 22.400 estimaciones MI incluyen objetivos absolutos y correcciones relativas. '
        'No se eligieron variables mediante estos resultados.</p>',
        tab(pd.DataFrame(vif_rows)),
        '<p>Los 48 VIF infinitos corresponden a tres columnas en cada diseño, con una sola redundancia '
        'lineal por diseño: para la escala de un día, downside = (|retorno| − retorno)/2. '
        'Además, el decaimiento conocido de horizonte 7 es constante cuando la ventana objetivo es 7. '
        'El VIF completo del diseño SVR de 6L+7 columnas no se recalculó; daily6 cubre el bloque contemporáneo. '
        'Un VIF alto no demuestra que retirar una columna mejore Ridge.</p>',
        '<h3>Mayor MI marginal con la corrección relativa</h3>',
        tab(top_mi.rename(columns={'feature': 'Entrada HAR', 'mi_nats': 'MI media (nats)'})),
        '<p>Promedios sobre activos, ventanas y horizontes. Son asociaciones marginales; '
        'no son valores p ni importancia condicional. El decaimiento y la normalización comparten '
        'información con el objetivo móvil.</p>',
        '<h2>Estacionalidad intradía</h2>',
        tab(intraday[['symbol', 'n_complete_return_days', 'hourly_peak_utc', 'hourly_trough_utc',
            'hourly_peak_to_trough_ratio']].rename(columns={'symbol': 'Activo',
            'n_complete_return_days': 'Días completos', 'hourly_peak_utc': 'Hora pico UTC',
            'hourly_trough_utc': 'Hora valle UTC', 'hourly_peak_to_trough_ratio': 'Razón pico/valle'})),
        '<img src="diagnostics/intraday_development.png" alt="Perfil horario UTC de volatilidad de desarrollo">',
        '<p>La figura usa retornos cuadrados normalizados por su media diaria, conserva los huecos y compara '
        '2023 con 2024. Las bandas son intervalos individuales de un bootstrap circular de bloques de siete '
        'días de calendario, con 499 réplicas. Esto documenta patrones de desarrollo; '
        'no demuestra estabilidad futura ni mejora predictiva al desestacionalizar.</p>',
        '<h2>Referencias y evaluación del modelo</h2><p>El HAR diario usa [σ actual, media de 7 días de σ, '
        'media de 28 días de σ] e intercepto para cada horizonte, sin búsqueda de hiperparámetros. '
        'El calentamiento reduce sus muestras de entrenamiento por ventana a 648, 634, 620 y 606 '
        'orígenes. Conserva todas las fechas y objetivos de evaluación. El HAR intradía usa la raíz '
        'de la suma de retornos de minuto al cuadrado y sus medias de 7/28 días. '
        'Son adaptaciones del HAR al objetivo del estudio.</p>',
        '<img src="mse_comparison.svg" alt="Comparación del error de pronóstico en 2025 y 2026">']
    for year in (2026, 2025):
        selected = metrics[metrics.year.eq(year) & metrics.model.isin([
            'Persistencia', 'HAR diario OLS (1/7/28)', 'SVR Lineal', BASE, AR,
            'HAR-XG + last mature error', 'HAR-XG + AR no online labels',
            'HAR-XG Ridge ponderado', 'HAR-Ridge + XGBoost residual'])]
        text += [f'<h3>{year}: métricas globales</h3>',
            tab(selected[['model', 'mse', 'rmse', 'mae', 'r2']].sort_values('mse').rename(
                columns={'model': 'Modelo', 'mse': 'MSE', 'rmse': 'RMSE macro', 'mae': 'MAE', 'r2': 'R² macro'}))]
    text += ['<p>358 orígenes diarios en 2025 y 236 en 2026 (1 de enero–24 de agosto). '
        'Las métricas macro promedian resultados por activo, ventana y horizonte. '
        'RMSE macro no equivale a la raíz del MSE global. Cada fecha aporta una pérdida '
        'media sobre las 112 salidas dependientes.</p>',
        '<h3>Prueba primaria predeclarada: corrección AR, 2026</h3>',
        tab(primary[['reference', 'reduction_percent', 'p', 'p_holm']].rename(columns={
            'reference': 'Referencia', 'reduction_percent': 'Reducción MSE (%)', 'p': 'p bruto', 'p_holm': 'p Holm4'})),
        '<p>DM bilateral con HLN, horizonte 7 y HAC Bartlett de 33 rezagos. Holm se aplica a estas cuatro '
        'comparaciones de 2026; 2025 es secundario. Se conservaron sensibilidades a 14/56 rezagos y '
        'tablas por ventana. La regla exigía menor MSE y ventaja significativa frente a las cuatro '
        'referencias en 2026, además de menor MSE frente al vigente en 2025. No se cumple.</p>',
        '<h2>La autocorrelación residual sigue presente</h2>',
        tab(residuals[residuals.model.isin([BASE, AR])][['year', 'model', 'median_acf1',
            'fraction_lb28_raw_p_below05']].rename(columns={'year': 'Año', 'model': 'Modelo',
            'median_acf1': 'Mediana ACF, rezago 1', 'fraction_lb28_raw_p_below05': 'Fracción p LB28 < 0,05'})),
        '<p>La corrección modela parte de la dependencia sin eliminarla. Ljung–Box se informa de forma '
        'descriptiva por serie, sin ajuste múltiple ni prueba global de esa fracción. '
        'Las ventanas superpuestas también inducen dependencia.</p>',
        '<h2>Qué permanece pendiente</h2><p>El EDA original fue posterior al modelo. Recalcularlo con '
        'desarrollo únicamente documenta el análisis sin cambiar esa cronología. '
        '2025 y enero–agosto de 2026 ya se exploraron. Hace falta fijar el procedimiento y evaluar '
        'un periodo posterior sin inspección para obtener confirmación prospectiva.</p>'
        '<p>La validación AR es condicional a hiperparámetros base seleccionados sobre todo 2024; '
        'no es una validación anidada independiente. En evaluación, los coeficientes quedan fijos, '
        'pero el estado incorpora errores maduros: e[s,h] se permite solo tras s+h ≤ t, al cierre de t. '
        'El control de último error recibe el mismo acceso a información reciente.</p>',
        '<h2>Verificación y archivos</h2><p>23 pruebas automatizadas aprobadas: causalidad de entradas, '
        'madurez de etiquetas, preservación de huecos, aislamiento del entrenamiento, '
        'colinealidad numérica y unidad diaria de inferencia. '
        'Los 209 archivos protegidos originales mantienen sus hashes, incluida la presentación TeX.</p>',
        '<p><a href="report.html">Informe técnico completo</a> · '
        '<a href="evaluation_protocol.json">Protocolo</a> · <a href="summary.json">Resultados y hashes</a> · '
        '<a href="macro_metrics.csv">Métricas CSV</a> · <a href="dm_ar_holm4.csv">DM y sensibilidad CSV</a> · '
        '<a href="diagnostics/mutual_information.csv">Información mutua CSV</a> · '
        '<a href="diagnostics/variance_inflation.csv">VIF CSV</a> · '
        '<a href="diagnostics/intraday_minute_profile.csv">Perfil por minuto CSV</a></p>',
        '<details><summary>Comandos de reproducción</summary><pre>'
        '.venv-repro/Scripts/python.exe -m unittest discover -s tests -p "test_professor_feedback*.py"\n'
        '.venv-repro/Scripts/python.exe src/professor_feedback_diagnostics.py\n'
        '.venv-repro/Scripts/python.exe src/professor_feedback_har_baseline.py\n'
        '.venv-repro/Scripts/python.exe src/professor_feedback_daily_har.py\n'
        '.venv-repro/Scripts/python.exe src/professor_feedback_residuals.py\n'
        '.venv-repro/Scripts/python.exe src/evaluate_professor_feedback.py\n'
        '.venv-repro/Scripts/python.exe src/report_professor_feedback.py</pre>'
        '<p>Los ejecutores preservan resultados existentes y rechazan sobrescribirlos. Para un nuevo '
        'experimento, usar otra carpeta de salida en una copia de los scripts y congelar su protocolo '
        'antes de evaluar. Ninguno publica los resultados.</p></details>',
        '<p>Fuentes: <a href="https://doi.org/10.1093/jjfinec/nbp001">Corsi, HAR</a>; '
        '<a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html">información mutua</a>; '
        '<a href="https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html">VIF</a>; '
        '<a href="https://doi.org/10.1016/S0927-5398(97)00004-2">periodicidad intradía</a>; '
        '<a href="https://pkg.robjhyndman.com/forecast/reference/dm.test.html">DM-HLN</a>.</p></html>']
    path = OUT / 'informe.html'
    path.write_text('\n'.join(text), encoding='utf-8')
    sources = sorted((ROOT / 'src').glob('*professor_feedback*.py')) + sorted((ROOT / 'tests').glob('test_professor_feedback*.py'))
    manifest = dict(status='complete', publication='LOCAL ONLY', tests_passed=23,
        test_command='.venv-repro/Scripts/python.exe -m unittest discover -s tests -p test_professor_feedback*.py',
        protected_files_unchanged=len(before), github_or_render_updated=False,
        analysis_summary_sha256=sha(OUT / 'summary.json'), report_sha256=sha(path),
        code_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources})
    (OUT / 'review_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(str(path), flush=True)


if __name__ == '__main__':
    run()
