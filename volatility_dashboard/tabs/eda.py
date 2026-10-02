import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from ..data_loader import dataset, eda_frame, FEATURES
from ..utils import graph_card, table, acf_figure, empty


def layout():
    panel = dataset()
    return html.Div([
        html.H2('EDA orientado a volatilidad'),html.P('La vista inicia en development. Explorar test es un análisis retrospectivo; las gráficas no modifican modelos ni características.'),
        html.Div([
            html.Div([html.Label('Activo'),dcc.Dropdown(list(panel.columns),'BTCUSDT',id='eda-symbol',clearable=False)]),
            html.Div([html.Label('Ventana de volatilidad'),dcc.Dropdown([7,14,21,28],7,id='eda-window',clearable=False)]),
            html.Div([html.Label('Horizonte futuro (días)'),dcc.Dropdown(list(range(1,8)),1,id='eda-horizon',clearable=False)]),
            html.Div([html.Label('Feature para relación y distribución'),dcc.Dropdown(FEATURES+['decaimiento_conocido'],FEATURES[0],id='eda-feature',clearable=False)]),
            html.Div([html.Label('Correlación'),dcc.Dropdown(['Pearson','Spearman'],'Spearman',id='eda-correlation',clearable=False)]),
            html.Div([html.Label('Periodo de orígenes'),dcc.DatePickerRange(id='eda-period',min_date_allowed=str(panel.index.min().date()),max_date_allowed=str(panel.index.max().date()),start_date='2023-01-01',end_date='2024-12-24',display_format='DD/MM/YYYY')]),
            html.Div([html.Label('Activos para comparar'),dcc.Dropdown(list(panel.columns),list(panel.columns),multi=True,id='eda-assets')])],className='filters'),
        dcc.Loading(html.Div(id='eda-content'),type='circle')])


def render(symbol,window,horizon,feature,correlation,start,end,assets):
    if not start or not end:
        return html.P('Selecciona un periodo completo.')
    lower, upper = pd.Timestamp(start,tz='UTC'),pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=1)
    frame = eda_frame(symbol,int(window),int(horizon))
    # Also contain the future label in the selected calendar period.
    selected = frame[(frame.index>=lower)&(frame.index<upper)].copy()
    selected.loc[selected.index+pd.Timedelta(days=int(horizon))>=upper,'volatilidad_futura'] = np.nan
    if selected.close.notna().sum()<2:
        return html.P('No hay suficientes observaciones en este periodo.',className='notice')
    timeline = make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=['Cierre diario UTC','Retorno logarítmico (%)','Volatilidad histórica y futura'],vertical_spacing=.09)
    for row,column in [(1,'close'),(2,'retorno_log_pct'),(3,'volatilidad_historica'),(3,'volatilidad_futura')]:
        timeline.add_trace(go.Scatter(x=selected.index,y=selected[column],name=column,connectgaps=False,mode='lines'),row=row,col=1)
    timeline.update_layout(height=650)
    returns = selected.retorno_log_pct.dropna()
    return_stats = pd.DataFrame([dict(n=len(returns),media=returns.mean(),std=returns.std(),skewness=returns.skew(),kurtosis_exceso=returns.kurt())])
    histogram = px.histogram(selected,x='retorno_log_pct',nbins=55,color_discrete_sequence=['#2563eb'],labels={'retorno_log_pct':'Retorno logarítmico (%)'})
    rows, statistics = [], []
    for asset in assets or [symbol]:
        f = eda_frame(asset,int(window),int(horizon)).copy()
        f = f[(f.index>=lower)&(f.index<upper)]
        f.loc[f.index+pd.Timedelta(days=int(horizon))>=upper,'volatilidad_futura'] = np.nan
        q = f.volatilidad_futura.dropna()
        if q.empty:
            continue
        statistics.append(dict(activo=asset,n=len(q),media=q.mean(),mediana=q.median(),std=q.std(),min=q.min(),max=q.max(),Q1=q.quantile(.25),Q3=q.quantile(.75)))
        f['symbol'] = asset
        rows.append(f)
    compared = pd.concat(rows) if rows else pd.DataFrame()
    target_hist = px.histogram(compared,x='volatilidad_futura',color='symbol',barmode='overlay',opacity=.55,nbins=45) if not compared.empty else empty('Target','Sin etiquetas completas en este periodo.')
    relevant = selected[FEATURES+['decaimiento_conocido','volatilidad_historica','volatilidad_futura']]
    corr = relevant.corr(method=correlation.lower())
    corr_fig = px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale='RdBu_r',text_auto='.2f',aspect='auto')
    pairs = selected[[feature,'volatilidad_futura']].dropna()
    scatter = px.scatter(pairs,x=feature,y='volatilidad_futura',opacity=.45)
    rho = pairs.corr(method=correlation.lower()).iloc[0,1] if len(pairs)>1 else np.nan
    return_acf,return_note = acf_figure(selected.retorno_log_pct,'ACF de retornos diarios')
    vol_acf,vol_note = acf_figure(selected.volatilidad_historica,'ACF de volatilidad histórica')
    squared_acf,squared_note = acf_figure(selected.retorno_log_pct**2,'ACF de retornos al cuadrado')
    variable_hist = px.histogram(selected,x=feature,nbins=45,color_discrete_sequence=['#0891b2'])
    boxes = []
    if not compared.empty:
        for column in ['retorno_log_pct','volatilidad_futura']:
            means = compared.groupby('symbol')[column].median().dropna()
            note = f'Medianas observadas: '+ '; '.join(f'{a}: {v:.3f}' for a,v in means.items())+'. Comparación descriptiva del mismo periodo.'
            boxes.append(graph_card(column.replace('_',' ').capitalize()+' entre activos',px.box(compared,x='symbol',y=column,color='symbol',points=False),note))
    return html.Div([
        graph_card('Evolución temporal',timeline,f'{symbol}: {selected.close.notna().sum()} cierres disponibles; precio entre {selected.close.min():,.2f} y {selected.close.max():,.2f}. Ventana {window} días y horizonte {horizon}; las etiquetas deben terminar dentro del periodo seleccionado.'),
        html.Div([graph_card('Distribución de retornos',histogram,f'Media {returns.mean():.4f} %, desviación {returns.std():.4f} %, asimetría {returns.skew():.3f} y exceso de curtosis {returns.kurt():.3f}.'),
                  graph_card('Distribución de la feature',variable_hist,f'{feature}: {selected[feature].notna().sum()} valores; mediana {selected[feature].median():.4f}. La feature conserva la normalización utilizada por los modelos.')],className='grid-two'),
        html.Section([html.H3('Estadísticas de retornos'),table(return_stats),html.P('No se atribuyen pruebas Jarque–Bera de otros periodos a este filtro. No hay un resultado guardado y alineado para esta selección.')],className='panel'),
        graph_card('Distribución de volatilidad futura',target_hist,f'{len(statistics)} activos con etiquetas completas para horizonte {horizon} y ventana {window}. Las unidades son puntos porcentuales, sin anualizar.'),
        html.Section([html.H3('Estadísticas del target'),table(pd.DataFrame(statistics))],className='panel'),
        html.Div([graph_card('Correlaciones '+correlation,corr_fig,f'{len(pairs)} pares completos para la feature seleccionada; correlación con target = {rho:.3f}. El heatmap excluye identificadores y fechas; correlación no implica causalidad.'),
                  graph_card('Feature frente a target',scatter,f'{symbol}: {len(pairs)} pares completos; {correlation} = {rho:.3f}. Se muestran todas las observaciones de este dataset diario.')],className='grid-two'),
        html.Div([graph_card('Autocorrelación de retornos',return_acf,return_note),graph_card('Autocorrelación de volatilidad',vol_acf,vol_note),graph_card('Autocorrelación de cuadrados',squared_acf,squared_note)]+boxes,className='grid-two'),
        html.P('Volumen relativo vs volatilidad: no disponible. El dataset procesado compartido por los modelos solo conserva cierres y características de retornos.',className='notice')])


def register(app):
    app.callback(Output('eda-content','children'),Input('eda-symbol','value'),Input('eda-window','value'),Input('eda-horizon','value'),Input('eda-feature','value'),Input('eda-correlation','value'),Input('eda-period','start_date'),Input('eda-period','end_date'),Input('eda-assets','value'))(render)
