"""A saved, retrospective review alongside the unchanged original comparison."""
import logging

from dash import Input, Output, dcc, html
import pandas as pd
import plotly.graph_objects as go

from .. import final_review_data as data
from ..utils import graph_card, table


LOGGER = logging.getLogger(__name__)


def _format_p(frame, columns):
    """Keep small p-values visible instead of rounding them to zero."""
    for column in columns:
        frame[column] = frame[column].map(lambda value: f'{value:.6g}')
    return frame


def _link(label, relative, class_name=None):
    kwargs = {'href': data.public_url(relative), 'target': '_blank', 'rel': 'noopener'}
    if class_name:
        kwargs['className'] = class_name
    return html.A(label, **kwargs)


def _downloads():
    return html.Div([
        _link('Informe de la revisión', f'{data.RESULTS}/report.html', 'button'),
        _link('Entrega navegable', f'{data.DELIVERY}/index.html', 'button'),
        _link('Notebook reproducible', f'{data.DELIVERY}/notebooks/revision_resultados.ipynb', 'button'),
        _link('Presentación · 10 minutos', f'{data.DELIVERY}/pdf/presentacion_final_10min.pdf', 'button'),
    ], className='final-review-links')


def layout():
    """Render controls immediately; no repository audit or model is loaded."""
    return html.Div([
        html.H2('Revisión final · solo regresión'),
        html.P('Siete familias × cuatro métodos de optimización: 28 benchmarks. '
               'Esta vista amplía la evidencia del proyecto y conserva el modelo '
               'HAR-Ridge + XGBoost original como referencia.'),
        html.P('Los resultados de 2025 y 2026 son retrospectivos. Los candidatos '
               'HAR son experimentales: todavía no se ha demostrado una mejora '
               'estadística que justifique sustituir el modelo original.',
               className='notice final-review-status'),
        _downloads(),
        html.Div([
            html.Div([html.Label('Año de evaluación'), dcc.Dropdown(
                [{'label': '2025 · 358 orígenes', 'value': 2025},
                 {'label': '2026 · 236 orígenes', 'value': 2026}],
                2025, id='final-review-year', clearable=False)]),
            html.Div([html.Label('Método de optimización'), dcc.Dropdown(
                [{'label': 'Todos · 28 benchmarks', 'value': 'all'}]
                + [{'label': label, 'value': method} for method, label in data.OPTIMIZERS.items()],
                'all', id='final-review-optimizer', clearable=False)]),
            html.Div([html.Label('Ventana del objetivo (días)'), dcc.Dropdown(
                list(data.WINDOWS), 7, id='final-review-window', clearable=False)]),
        ], className='filters'),
        dcc.Loading(html.Div(html.P('Cargando tablas verificadas de la revisión…',
                                  className='notice'), id='final-review-content'), type='circle'),
    ], className='final-review')


def render(year, optimizer, window):
    try:
        return _render(int(year), optimizer, int(window))
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        LOGGER.exception('No se pudo cargar el paquete de revisión final.')
        return html.Div([
            html.H3('La revisión final no está disponible'),
            html.P('No se pudo leer el paquete de resultados de esta selección. '
                   'La comparación original continúa disponible en su pestaña.'),
        ], className='notice')


def _metrics_section(frame, year, window):
    reference = frame.loc[frame.model.eq(data.REFERENCE)].iloc[0]
    n = 358 if year == 2025 else 236
    best = frame.iloc[0]
    cards = html.Div([
        html.Div([html.Strong(str(n)), html.Span('Orígenes diarios comunes')], className='stat-card'),
        html.Div([html.Strong(str(len(frame) - 1)), html.Span('Benchmarks en este filtro')], className='stat-card'),
        html.Div([html.Strong(f'{reference.mse:.5f}'), html.Span('MSE del HAR original')], className='stat-card'),
    ], className='cards')
    # A readable preview, followed by the complete table, without hiding rivals.
    preview = frame.head(8)
    if data.REFERENCE not in preview.model.values:
        preview = pd.concat([preview, frame.loc[frame.model.eq(data.REFERENCE)]])
    preview = preview.sort_values('mse', ascending=False)
    figure = go.Figure(go.Bar(
        x=preview.mse, y=preview.model, orientation='h',
        marker_color=['#b13e75' if model == data.REFERENCE else '#087f8c' for model in preview.model],
        hovertemplate='%{y}<br>MSE: %{x:.6f}<extra></extra>'))
    figure.update_layout(height=360, xaxis_title='MSE (puntos porcentuales²)',
                         yaxis_title=None, yaxis_automargin=True, showlegend=False)
    display = frame.rename(columns={'model': 'Modelo', 'mse': 'MSE ↓', 'rmse': 'RMSE ↓',
                                    'mae': 'MAE ↓', 'r2': 'R² ↑'})
    return html.Section([
        html.H3(f'Errores observados · {year} · ventana {window} días'),
        html.P('Cuatro activos y siete horizontes sobre las mismas fechas. '
               'Las métricas se calculan por serie y después se promedian. '
               'RMSE medio no equivale a la raíz del MSE medio; R² no es una tasa de acierto.'),
        cards,
        graph_card('Vista de los ocho menores MSE y la referencia', figure,
                   f'{best.model} presenta el menor MSE observado en este filtro. '
                   'Esta ordenación descriptiva no demuestra superioridad estadística.'),
        table(display, page_size=10),
        html.P('RMSE y MAE: puntos porcentuales de volatilidad sin anualizar. '
               'MSE: puntos porcentuales al cuadrado. Cada ventana define un objetivo distinto.',
               className='interpretation'),
    ], className='panel')


def _dm_section(year, optimizer, window):
    if year != 2025:
        return html.Section([
            html.H3('Comparaciones dirigidas por ventana'),
            html.P('La tabla de 112 contrastes por ventana corresponde exclusivamente a 2025. '
                   'No se reutilizan sus valores p para 2026. Más abajo se muestran '
                   'los cuatro contrastes globales HAR de 2026, con su propia familia Holm4.',
                   className='notice'),
            _link('Consultar la evaluación estadística de 2026', f'{data.RESULTS}/evaluation/report.html'),
        ], className='panel')
    dm = data.dm_2025(optimizer, window)
    display = dm[['method', 'family', 'mse_a', 'mse_b', 'diferencia_mse', 'dm', 'p',
                  'p_holm7', 'p_holm28', 'p_holm112']].copy()
    for size in (7, 28, 112):
        display[f'Decisión Holm{size}'] = [data.dm_decision(row, size) for _, row in dm.iterrows()]
    display = display.rename(columns={'method': 'Optimizador', 'family': 'Benchmark',
        'mse_a': 'MSE original', 'mse_b': 'MSE benchmark', 'diferencia_mse': 'MSE original − benchmark',
        'dm': 'DM-HLN', 'p': 'p bruto', 'p_holm7': 'p Holm7',
        'p_holm28': 'p Holm28', 'p_holm112': 'p Holm112'})
    display = _format_p(display, ['p bruto', 'p Holm7', 'p Holm28', 'p Holm112'])
    counts = {size: sum(data.dm_decision(row, size) == 'Favorece al original'
                       for _, row in dm.iterrows()) for size in (7, 28, 112)}
    return html.Section([
        html.H3(f'Diebold–Mariano · 2025 · ventana {window} días'),
        html.P('Modelo A: HAR-Ridge + XGBoost original. Modelo B: benchmark. '
               'Una diferencia de MSE o un DM negativo favorece al original. '
               'Contraste bilateral con corrección HLN y varianza HAC; nivel del 5 %.'),
        html.Ul([
            html.Li('Holm7: siete familias dentro de cada optimizador y ventana.'),
            html.Li('Holm28: los 28 benchmarks de una misma ventana.'),
            html.Li('Holm112: los 28 benchmarks de las cuatro ventanas de 2025.'),
        ]),
        html.P('Los ajustes están calculados sobre sus familias completas. '
               'Cambiar el filtro solo muestra filas; no recalcula los valores p.', className='notice'),
        html.P(f'Filas mostradas que favorecen al original: Holm7 {counts[7]}/{len(dm)}, '
               f'Holm28 {counts[28]}/{len(dm)}, Holm112 {counts[112]}/{len(dm)}.'),
        table(display, page_size=7),
        html.P('No rechazar igualdad de pérdida esperada no demuestra equivalencia. '
               'Estos contrastes pertenecen a una revisión retrospectiva; no son una validación prospectiva.',
               className='interpretation'),
        _link('Informe completo de las 112 comparaciones', f'{data.WINDOW_REVIEW}/report.html'),
        html.Br(),
        _link('Descargar valores p completos (CSV)', f'{data.WINDOW_REVIEW}/dm_all_112.csv'),
    ], className='panel')


def _windows_section(year):
    if year != 2025:
        rows = []
        for window in data.WINDOWS:
            frame = data.metrics(year, 'all', window)
            original = frame.loc[frame.model.eq(data.REFERENCE)].iloc[0]
            rivals = frame.loc[~frame.model.eq(data.REFERENCE)]
            rows.append({'Ventana (días)': window, 'MSE original': original.mse,
                         'RMSE original': original.rmse, 'R² original': original.r2,
                         'Menor MSE benchmark': rivals.mse.min()})
        return html.Section([
            html.H3('Cuatro objetivos · resumen descriptivo de 2026'),
            html.P('Incluye los 28 benchmarks por ventana, con independencia del optimizador '
                   'seleccionado. Esta tabla no contiene contrastes DM por ventana.'),
            table(pd.DataFrame(rows), page_size=4),
        ], className='panel')
    source = data.window_summary()
    display = source[['volatility_window', 'reference_rmse', 'reference_r2', 'original_lower_mse']].copy()
    display = display.rename(columns={'volatility_window': 'Ventana (días)',
        'reference_rmse': 'RMSE original', 'reference_r2': 'R² original',
        'original_lower_mse': 'Original con menor MSE (de 28)'})
    for size in (7, 28, 112):
        display[f'Favorece al original Holm{size} (de 28)'] = source[f'original_wins_holm{size}']
    return html.Section([
        html.H3('Cuatro objetivos · resumen completo de 2025'),
        html.P('Siempre resume los 28 benchmarks por ventana, aunque el filtro superior '
               'muestre un solo optimizador. Los conteos Holm7 suman cuatro familias '
               'separadas de siete; no representan un ajuste conjunto de 28.'),
        table(display, page_size=4),
        html.P('Ninguna ventana favorece al original frente a una mayoría de los 28 '
               'benchmarks con Holm112. La ventana inicial de siete días es una elección '
               'expositiva retrospectiva. No se debe elegir el objetivo por el valor p más favorable.',
               className='interpretation'),
    ], className='panel')


def _candidate_section(year):
    frame = data.candidate_global(year)
    display = frame[['model', 'mse_original', 'mse', 'mse_reduction_pct', 'p', 'p_holm',
                     'decision_holm4']].rename(columns={'model': 'Candidato experimental',
        'mse_original': 'MSE original global', 'mse': 'MSE candidato global',
        'mse_reduction_pct': 'Reducción MSE (%)', 'p': 'p bruto',
        'p_holm': 'p Holm4', 'decision_holm4': 'Decisión al 5 %'})
    display = _format_p(display, ['p bruto', 'p Holm4'])
    return html.Section([
        html.H3(f'Candidatos HAR experimentales · global {year}'),
        html.P('Familia distinta: cuatro candidatos HAR-XG frente al HAR original. '
               'Se agregan las cuatro ventanas, cuatro activos y siete horizontes '
               '(112 errores por origen). Esta tabla no cambia con los filtros de ventana '
               'u optimizador y no es la tabla Holm7/28/112.'),
        table(display, page_size=4),
        html.P('Ninguno de los cuatro candidatos demuestra una mejora con el ajuste Holm4 '
               'en 2025 ni en 2026. El menor error puntual de algunos candidatos '
               'permite continuar la investigación; el modelo original permanece como referencia.',
               className='notice'),
        _link('MCS, SPA, intervalos estacionarios y contrastes dirigidos', f'{data.RESULTS}/evaluation/report.html'),
    ], className='panel')


def _outer_section(optimizer):
    frame = data.outer_metrics(optimizer)
    children = [html.H3('Optimización y validación temporal anidada'),
        html.P('Grid, Random, Bayesiana y Genética se comparan dentro de pliegues '
               'temporales internos. El bucle externo de 2024 estima el desempeño '
               'del procedimiento y guía la selección, antes de evaluar 2025/2026.')]
    if not frame.empty:
        columns = ['model', 'rmse_mean', 'rmse_sd', 'mae_mean', 'mae_sd', 'mse_mean']
        children.append(html.Details([html.Summary('Ver medias y desviaciones entre pliegues externos'),
            html.P('Agregación sobre activos y ventanas. Las desviaciones describen '
                   'la variación entre pliegues; los pliegues temporales no son independientes.'),
            table(frame[columns].rename(columns={'model': 'Procedimiento',
                'rmse_mean': 'RMSE medio externo', 'rmse_sd': 'DE entre pliegues RMSE',
                'mae_mean': 'MAE medio externo', 'mae_sd': 'DE entre pliegues MAE',
                'mse_mean': 'MSE medio externo'}), page_size=7)]))
    children.extend([
        _link('Curvas train/validación/test de los 32 procedimientos',
              f'{data.RESULTS}/plots/train_validation_test_all/report.html'),
        html.Br(),
        _link('Convergencia por número de evaluaciones internas',
              f'{data.RESULTS}/plots/anytime_evaluations.png'),
        html.Br(),
        _link('Convergencia por tiempo de búsqueda',
              f'{data.RESULTS}/plots/anytime_seconds.png'),
        html.Br(),
        _link('Estabilidad entre semillas y eficiencia de los cuatro optimizadores',
              f'{data.RESULTS}/sensitivity/report.html'),
        html.Br(),
        _link('Auditoría de convergencia Lasso', f'{data.RESULTS}/lasso_convergence/report.html'),
        html.P('Se conservaron las corridas Lasso que alcanzaron el límite de iteraciones; '
               'sus advertencias están documentadas y no se supone convergencia de todos los ajustes.',
               className='interpretation'),
    ])
    return html.Section(children, className='panel')


def _saved_image(relative, description):
    if not data.available(relative):
        return html.P('La figura no está incluida en este paquete. Consulta el informe enlazado.',
                      className='interpretation')
    return html.A(html.Img(src=data.public_url(relative), alt=description,
                          className='final-review-preview'),
                  href=data.public_url(relative), target='_blank', rel='noopener')


def _diagnostic_sections():
    eda = html.Section([
        html.H3('EDA ampliado · desarrollo 2023–2024'),
        html.P('Se incorporaron información mutua, VIF y estacionalidad intradía. '
               'El diagnóstico usa exclusivamente datos de desarrollo; no restaura '
               'el carácter inédito de los periodos de prueba ya explorados.'),
        _saved_image(f'{data.FEEDBACK}/diagnostics/intraday_development.png',
                     'Perfil intradía UTC de retornos cuadrados normalizados, desarrollo 2023–2024'),
        _link('Abrir diagnóstico EDA: MI, VIF y estacionalidad', f'{data.FEEDBACK}/informe.html'),
    ], className='panel')
    shap = html.Section([
        html.H3('Interpretabilidad · SHAP y LIME'),
        html.P('El procedimiento seleccionado con el MSE externo de 2024 se explica '
               'en las 16 configuraciones y siete horizontes de 2026. Las atribuciones '
               'son aproximadas y asociativas; las entradas correlacionadas y el '
               'enmascaramiento fuera del soporte limitan su interpretación causal.'),
        _saved_image(f'{data.RESULTS}/explain_best/plots/global_bar.png',
                     'Importancia SHAP global del procedimiento seleccionado por validación externa de 2024'),
        _link('SHAP global y dos explicaciones locales', f'{data.RESULTS}/explain_best/report.html'),
        html.Br(),
        _link('HAR: 16 configuraciones y comprobación de aditividad', f'{data.RESULTS}/explain_global/report.html'),
        html.Br(),
        _link('Comparación local SHAP–LIME y fidelidad', f'{data.RESULTS}/explain/report.html'),
    ], className='panel')
    efficiency = data.efficiency()
    efficiency_children = [html.H3('Costo computacional y residuos'),
        html.P('El benchmark compara variantes estándar y optimizada en los mismos '
               'datos de desarrollo, con tres repeticiones. Incluye tiempo de ajuste '
               'e inferencia, memoria RSS muestreada, memoria Python y complejidad '
               'teórica. Es un diagnóstico de costo y error, no evidencia adicional '
               'de generalización.')]
    if not efficiency.empty:
        cols = ['model', 'train_rows', 'fit_speedup_mean', 'inference_speedup_mean',
                'diagnostic_rmse_change_mean']
        efficiency_children.append(html.Details([
            html.Summary('Ver resumen con el mayor tamaño de entrenamiento'),
            table(efficiency[cols].rename(columns={'model': 'Familia', 'train_rows': 'Filas de ajuste',
                'fit_speedup_mean': 'Tiempo estándar / optimizado (ajuste)',
                'inference_speedup_mean': 'Tiempo estándar / optimizado (inferencia)',
                'diagnostic_rmse_change_mean': 'Cambio RMSE diagnóstico'}), page_size=8),
            html.P('Razones mayores de uno indican menor tiempo de la variante optimizada. '
                   'Cambio RMSE = optimizado − estándar. La memoria RSS incluye el '
                   'proceso y sus bibliotecas; el muestreo puede omitir picos breves.',
                   className='interpretation')]))
    efficiency_children.extend([
        _link('Eficiencia, memoria y complejidad', f'{data.RESULTS}/efficiency/report.html'),
        html.Br(),
        _link('White, BDS, ACF, Ljung–Box e histogramas de residuos', f'{data.RESULTS}/evaluation/report.html'),
        html.P('Los objetivos móviles y horizontes solapados inducen dependencia. '
               'Una corrección HAC o un bootstrap por bloques ajusta la inferencia; '
               'no elimina por sí sola la estructura remanente de los residuos.',
               className='interpretation')])
    return [html.Div([eda, shap], className='grid-two'),
            html.Section(efficiency_children, className='panel')]


def _render(year, optimizer, window):
    metrics = data.metrics(year, optimizer, window)
    return html.Div([
        _metrics_section(metrics, year, window),
        _dm_section(year, optimizer, window),
        _windows_section(year),
        _candidate_section(year),
        _outer_section(optimizer),
        *_diagnostic_sections(),
    ])


def register(app):
    app.callback(Output('final-review-content', 'children'),
                 Input('final-review-year', 'value'),
                 Input('final-review-optimizer', 'value'),
                 Input('final-review-window', 'value'))(render)
