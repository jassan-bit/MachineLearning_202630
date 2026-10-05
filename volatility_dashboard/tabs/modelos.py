import json
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from ..data_loader import ROOT, FAMILIES, repository, dataset, artifact_for, hyperparameters, importance, tuning, read_json
from ..metrics import aggregate
from ..diebold_mariano import comparisons
from ..utils import COLORS, METRICS, graph_card, table, acf_figure

LOGGER = logging.getLogger(__name__)


def layout():
    available = list(FAMILIES)
    assets = list(dataset().columns)
    return html.Div([
        html.H2('Comparación de modelos'),html.P('Test común: 1 enero–24 diciembre de 2025. RMSE y MAE en puntos porcentuales de volatilidad. R² sin ajustar. No se combinan métricas en un score.'),
        html.Div([
            html.Div([html.Label('Activo del ranking'),dcc.Dropdown(['TODOS']+assets,'TODOS',id='models-symbol',clearable=False)]),
            html.Div([html.Label('Ventana del objetivo (días)'),dcc.Dropdown([7,14,21,28],7,id='models-window',clearable=False)]),
            html.Div([html.Label('Horizonte del ranking'),dcc.Dropdown([{'label':'Promedio de los 7 horizontes','value':'TODOS'}]+[{'label':f'Día {h}','value':h} for h in range(1,8)],'TODOS',id='models-horizon',clearable=False)]),
            html.Div([html.Label('Modelos comparados'),dcc.Dropdown(available,available,multi=True,id='models-selected')]),
            html.Div([html.Label('Métrica para ordenar'),dcc.Dropdown([{'label':v,'value':k} for k,v in METRICS.items()],'rmse',id='models-metric',clearable=False)]),
            html.Div([html.Label('Modelo para diagnóstico'),dcc.Dropdown(available,'XGBoost' if 'XGBoost' in available else available[0],id='models-focus',clearable=False)]),
            html.Div([html.Label('Activo para diagnóstico individual'),dcc.Dropdown(assets,'BTCUSDT',id='models-asset',clearable=False)]),
            html.Div([html.Label('Horizonte para diagnóstico individual'),dcc.Dropdown(list(range(1,8)),1,id='models-detail-h',clearable=False)])],className='filters'),
        dcc.Loading(html.Div(html.P('Cargando resultados de los modelos…',className='notice'),id='models-content'),type='circle')])


def numeric_note(frame,metric):
    ordered = frame.sort_values(metric,ascending=metric!='r2').dropna(subset=[metric])
    if ordered.empty:
        return 'Sin resultados para esta selección.'
    first = ordered.iloc[0]
    text = f'{first.model}: {METRICS[metric]} = {first[metric]:.5f}, primer lugar descriptivo en esta métrica y filtro.'
    if len(ordered)>1:
        second = ordered.iloc[1]
        text += f' {second.model}: {second[metric]:.5f}; diferencia absoluta {abs(second[metric]-first[metric]):.5f}.'
    return text+' La ordenación no demuestra significancia estadística.'


def render(symbol,window,horizon,models,metric,focus,asset,detail_h):
    try:
        return _render(symbol,window,horizon,models,metric,focus,asset,detail_h)
    except (OSError,ValueError,KeyError,IndexError,TypeError):
        LOGGER.exception('No se pudo cargar la comparación de modelos.')
        return html.Div([
            html.H3('No se pudo cargar la comparación'),
            html.P('Vuelve a seleccionar la pestaña o cambia los filtros para intentarlo de nuevo.')],className='notice')


def _render(symbol,window,horizon,models,metric,focus,asset,detail_h):
    predictions, all_metrics, audit = repository()
    if not models:
        return html.P('Selecciona al menos un modelo.',className='notice')
    focused_asset = symbol if symbol != 'TODOS' else asset
    focused_h = int(horizon) if horizon != 'TODOS' else int(detail_h)
    focus = focus if focus in predictions.model.unique() else models[0]
    view = aggregate(all_metrics,symbol,int(window),horizon,models)
    global_view = view.groupby('model',as_index=False)[['mae','rmse','mse','r2']].mean()
    ranking = global_view.sort_values(metric,ascending=metric!='r2').reset_index(drop=True)
    ranking.insert(0,'posición',range(1,len(ranking)+1))
    dm_results = comparisons(predictions,symbol,window,horizon,models)
    figures = [graph_card(f'Comparación · {METRICS[m]}',px.bar(global_view,x='model',y=m,color='model',color_discrete_map=COLORS),numeric_note(global_view,m)) for m in ['rmse','mae','r2']]
    all_assets = aggregate(all_metrics,'TODOS',int(window),horizon,models)
    matrix = all_assets.pivot(index='model',columns='symbol',values=metric)
    heat = px.imshow(matrix,color_continuous_scale='Viridis',aspect='auto',text_auto='.4f')
    pivot = matrix.copy()
    pivot['Promedio'] = pivot.mean(axis=1)
    selected = all_metrics[all_metrics.model.isin(models)&all_metrics.volatility_window.eq(int(window))]
    if symbol!='TODOS':
        selected = selected[selected.symbol.eq(symbol)]
    horizons = selected.groupby(['model','horizon'],as_index=False)[['mae','rmse','r2']].mean()
    horizon_fig = px.line(horizons,x='horizon',y=metric,color='model',markers=True,color_discrete_map=COLORS)
    firstlast = horizons[horizons.horizon.isin([1,7])].pivot(index='model',columns='horizon',values=metric)
    hnote = '; '.join(f'{m}: día 1 {row[1]:.4f} → día 7 {row[7]:.4f}' for m,row in firstlast.iterrows())
    actual = predictions[(predictions.model==focus)&(predictions.symbol==focused_asset)&(predictions.volatility_window==int(window))&(predictions.horizon==focused_h)].sort_values('origin').copy()
    actual['residuo'] = actual.actual-actual.forecast
    trend = go.Figure([go.Scatter(x=actual.origin,y=actual.actual,name='Volatilidad real',line={'color':'#0891b2'}),
                      go.Scatter(x=actual.origin,y=actual.forecast,name=focus,line={'color':COLORS[focus]})])
    scatter = px.scatter(actual,x='actual',y='forecast',opacity=.5)
    lower,upper = float(actual[['actual','forecast']].min().min()),float(actual[['actual','forecast']].max().max())
    scatter.add_trace(go.Scatter(x=[lower,upper],y=[lower,upper],mode='lines',name='Ideal y = x',line={'dash':'dash','color':'#dc2626'}))
    selected_score = all_metrics.query('model == @focus and symbol == @focused_asset and volatility_window == @window and horizon == @focused_h').iloc[0]
    note = f'{focus} · {focused_asset}, ventana {window}, horizonte {focused_h}: {len(actual)} fechas; RMSE {selected_score.rmse:.5f}, MAE {selected_score.mae:.5f}, R² {selected_score.r2:.5f}.'
    under = 100*float((actual.forecast<actual.actual).mean())
    mean,std = actual.residuo.mean(),actual.residuo.std()
    residual_time = px.line(actual,x='origin',y='residuo',labels={'residuo':'Real − predicho'})
    residual_time.add_hline(y=0,line_dash='dash')
    residual_hist = px.histogram(actual,x='residuo',nbins=45)
    residual_scatter = px.scatter(actual,x='forecast',y='residuo',opacity=.5)
    residual_scatter.add_hline(y=0,line_dash='dash')
    residual_acf,residual_acf_note = acf_figure(actual.residuo,'ACF de residuos de test')
    params = []
    for model in predictions.model.unique():
        p = hyperparameters(model,focused_asset,int(window))
        params.append(dict(Modelo=model,Escalamiento=p.pop('escalamiento'),Tipo='Regresión',Parametros=json.dumps(p,ensure_ascii=False,default=str)))
    focus_params = hyperparameters(focus,focused_asset,int(window))
    importance_frame,method = importance(focus,focused_asset,int(window),focused_h)
    imp_fig = px.bar(importance_frame.sort_values('importance'),x='importance',y='feature',orientation='h',color_discrete_sequence=[COLORS[focus]])
    imp_fig.update_layout(height=550)
    leading = importance_frame.iloc[0]
    interpretation = graph_card('Importancia · '+focus,imp_fig,f'{method}. Mayor magnitud: {leading.feature}, valor {leading.importance:.5f}. No se comparan escalas de métodos distintos; variables correlacionadas pueden compartir o diluir la importancia.')
    regularization = []
    for model in ['Ridge','Lasso']:
        if model not in predictions.model.unique():
            continue
        net = artifact_for(model,focused_asset,int(window))['model'].named_steps[model.lower()]
        coef = net.coef_[focused_h-1]
        result = all_metrics.query('model == @model and symbol == @focused_asset and volatility_window == @window and horizon == @focused_h').iloc[0]
        regularization.append(dict(model=model,alpha=net.alpha,n_coef=len(coef),exactamente_cero=int(np.sum(coef==0)),cerca_cero_abs_1e_6=int(np.sum(np.abs(coef)<=1e-6)),RMSE=result.rmse,MAE=result.mae,R2=result.r2))
    ridge_figures = []
    for model in ['Ridge','Lasso']:
        if model in predictions.model.unique():
            coeff,desc = importance(model,focused_asset,int(window),focused_h)
            ridge_figures.append(graph_card('Coeficientes · '+model,px.bar(coeff.sort_values('importance'),x='importance',y='feature',orientation='h'),f'{desc}. Se muestran {len(coeff)} coeficientes de mayor magnitud para horizonte {focused_h}.'))
    specific = []
    for model,axis in [('k-NN','n_neighbors'),('Random Forest','min_samples_leaf'),('XGBoost','learning_rate')]:
        if model not in predictions.model.unique():
            continue
        search = tuning(model,focused_asset,int(window))
        if not search.empty and axis in search:
            options = search.groupby([axis,'input_window'],as_index=False).validation_rmse.min()
            options['input_window'] = options.input_window.astype(str)
            fig = px.line(options,x=axis,y='validation_rmse',color='input_window',markers=True)
            minimum = options.loc[options.validation_rmse.idxmin()]
            specific.append(graph_card('Tuning de '+model,fig,f'{len(search)} combinaciones guardadas en 2024. Mejor RMSE de validación {minimum.validation_rmse:.5f}, {axis}={minimum[axis]}, input={minimum.input_window}. Cada punto conserva el mínimo entre los demás parámetros; no son resultados de test.'))
            specific.append(html.Section([html.H3('Búsqueda completa · '+model),
                html.P('Filtra por profundidad, vecinos, regularización o cualquier parámetro guardado. Las filas corresponden a validación 2024.'),
                table(search)],className='panel'))
    times = []
    for model,family in FAMILIES.items():
        for filename in ['status.json','verification.json']:
            path = ROOT/'results'/family/filename
            if path.exists():
                saved = read_json(path)
                if 'seconds' in saved:
                    times.append(dict(model=model,segundos_guardados=saved['seconds'],alcance='Ejecución completa; no separa entrenamiento/predicción',fuente=str(path.relative_to(ROOT))))
                    break
    return html.Div([
        html.Section([html.H3('Tabla general · mismo test'),html.P(f'Activo: {symbol}; ventana del objetivo: {window}; horizonte: {horizon}. Cuando se elige TODOS, las métricas son promedios de métricas por horizonte y activo, no métricas sobre objetivos concatenados.'),table(view),html.H3('Ranking por '+METRICS[metric]),table(ranking),html.P(numeric_note(global_view,metric),className='interpretation')],className='panel'),
        html.Section([html.H3('Prueba de Diebold–Mariano'),
            html.P('H₀: igual pérdida cuadrática esperada. Prueba bilateral con corrección Harvey–Leybourne–Newbold, varianza HAC con pesos Bartlett y ajuste Holm entre los pares de esta selección (α = 0.05). DM negativo favorece al modelo A. Se compara MSE, incluso si el ranking se ordena por R², RMSE o MAE.'),
            html.P('TODOS promedia las pérdidas de activos y horizontes dentro de cada origen diario; no los trata como observaciones independientes. Los rezagos HAC cubren ventana + horizonte − 2, con un mínimo automático. La evaluación de 2025 es retrospectiva; la prueba no elimina el sesgo por selección previa ni implica equivalencia cuando no se rechaza H₀.'),
            table(dm_results) if not dm_results.empty else html.P('Selecciona al menos dos modelos.'),
            html.A('Referencia metodológica',href='https://pkg.robjhyndman.com/forecast/reference/dm.test.html',target='_blank')],className='panel'),
        html.Div(figures,className='grid-three'),
        graph_card('Mapa de rendimiento por activo',heat,f'{matrix.shape[0]} modelos y {matrix.shape[1]} activos disponibles; ventana {window}, horizonte {horizon}. SOL no tiene resultados y no participa en el promedio.'),
        html.Section([html.H3('Comparación global por activo'),table(pivot.reset_index())],className='panel'),
        graph_card('Error por horizonte',horizon_fig,hnote+'. Promedios homogéneos para la ventana seleccionada.'),
        html.H3(f'Diagnóstico individual · {focus} · {focused_asset} · h={focused_h}'),
        html.Div([graph_card('Real vs predicho · test',trend,note),graph_card('Dispersión real vs predicho',scatter,f'{under:.1f} % de fechas con subestimación; sesgo medio real − predicho {mean:.5f}. La diagonal representa igualdad, no un ajuste estadístico.')],className='grid-two'),
        html.Details([html.Summary('Residuos y diagnósticos'),html.Div([
            graph_card('Residuos a través del tiempo',residual_time,f'{len(actual)} residuos; media {mean:.5f}, desviación muestral {std:.5f}. Positivo significa subestimación.'),
            graph_card('Distribución de residuos',residual_hist,f'Mediana {actual.residuo.median():.5f}; Q1 {actual.residuo.quantile(.25):.5f}, Q3 {actual.residuo.quantile(.75):.5f}.'),
            graph_card('Residuos vs predicción',residual_scatter,f'Predicciones entre {actual.forecast.min():.5f} y {actual.forecast.max():.5f}; residuos entre {actual.residuo.min():.5f} y {actual.residuo.max():.5f}.'),
            graph_card('Autocorrelación residual',residual_acf,residual_acf_note)],className='grid-two'),
            html.P('No hay resultados Jarque–Bera, Breusch–Pagan o BDS guardados y alineados para este diagnóstico. No se atribuyen pruebas de modelos históricos a los modelos actuales.')],className='panel'),
        html.Details([html.Summary('Complejidad e hiperparámetros reales'),table(pd.DataFrame(params)),html.H4('Modelo de diagnóstico'),html.Pre(json.dumps(focus_params,indent=2,ensure_ascii=False,default=str))],className='panel'),
        html.Details([html.Summary('Interpretabilidad del modelo seleccionado'),interpretation],className='panel'),
        html.Details([html.Summary('Ridge vs Lasso · L2 y L1'),html.P('Ridge reduce magnitudes con L2; Lasso puede anular coeficientes con L1. Los conteos corresponden al horizonte seleccionado y a las respectivas ventanas de entrada.'),table(pd.DataFrame(regularization)),html.Div(ridge_figures,className='grid-two')],className='panel'),
        html.Details([html.Summary('k-NN, Random Forest y XGBoost · tuning guardado'),html.Div(specific,className='grid-two')],className='panel'),
        html.Details([html.Summary('Tiempo computacional registrado'),table(pd.DataFrame(times)),html.P('No se dispone de mediciones separadas y comparables de entrenamiento y predicción. Los tiempos de búsquedas con distintos presupuestos no forman un ranking de velocidad.')],className='panel'),
        html.Details([html.Summary('Auditoría interna'),table(audit)],className='panel')])


def register(app):
    app.callback(Output('models-content','children'),Input('models-symbol','value'),Input('models-window','value'),Input('models-horizon','value'),Input('models-selected','value'),Input('models-metric','value'),Input('models-focus','value'),Input('models-asset','value'),Input('models-detail-h','value'))(render)
