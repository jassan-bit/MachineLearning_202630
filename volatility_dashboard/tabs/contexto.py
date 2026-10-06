import pandas as pd
import plotly.express as px
from dash import dcc, html
from ..data_loader import ROOT, dataset, temporal_calendar, repository, FEATURES, FAMILIES
from ..utils import graph_card, table

DESCRIPTIONS = {'k-NN':'Promedia vecinos históricos según la distancia entre características.',
    'Ridge':'Regresión lineal con penalización L2 para controlar el tamaño de los coeficientes.',
    'Lasso':'Regresión lineal con penalización L1; puede anular coeficientes.',
    'Random Forest':'Combina árboles de regresión para aprender relaciones no lineales.',
    'XGBoost':'Añade árboles sucesivos para corregir los errores del conjunto.',
    'SVR Lineal':'Regresión con margen epsilon y regularización.'}


def layout():
    panel = dataset()
    predictions, _, audit = repository()
    folds, eligible, _ = temporal_calendar()
    boundary = pd.Timestamp('2025-01-01',tz='UTC')
    development = panel.index[panel.index < boundary]
    test = pd.DatetimeIndex(sorted(predictions.origin.unique()))
    final_train = panel.index[eligible & (panel.index+pd.Timedelta(days=7)<boundary)]
    counts = len(development)+len(test)
    timeline = pd.DataFrame([
        dict(Periodo='Development',Inicio=development.min(),Fin=development.max()+pd.Timedelta(days=1),Observaciones=len(development),Porcentaje=100*len(development)/counts),
        dict(Periodo='Validation · dentro de development',Inicio=panel.index[folds[0][1].min()],Fin=panel.index[folds[-1][1].max()]+pd.Timedelta(days=1),Observaciones=sum(len(va) for _,va in folds),Porcentaje=None),
        dict(Periodo='Test · orígenes completos',Inicio=test.min(),Fin=test.max()+pd.Timedelta(days=1),Observaciones=len(test),Porcentaje=100*len(test)/counts)])
    fig = px.timeline(timeline,x_start='Inicio',x_end='Fin',y='Periodo',color='Periodo',hover_data=['Observaciones','Porcentaje'])
    fig.update_yaxes(autorange='reversed')
    fold_rows = [dict(fold=i,train_n=len(tr),train_inicio=str(panel.index[tr.min()].date()),train_fin=str(panel.index[tr.max()].date()),
        ultima_etiqueta_train=str(panel.index[tr.max()+7].date()),validation_n=len(va),validation_inicio=str(panel.index[va.min()].date()),validation_fin=str(panel.index[va.max()].date())) for i,(tr,va) in enumerate(folds,1)]
    cards = [html.Div([html.Strong(symbol.replace('USDT','')),html.P('Disponible · fuente Binance 1 min')],className='stat-card') for symbol in panel.columns]
    return html.Div([
        html.Div([html.Span('REGRESIÓN · ESTUDIO 2023–2025',className='eyebrow'),html.H2('Estimar el riesgo futuro con información histórica'),
            html.P('Se estiman siete volatilidades diarias futuras usando cierres y retornos observados hasta cada origen. Los resultados son retrospectivos: 2025 ya se había examinado.')],className='hero'),
        html.Div(cards,className='cards'),
        html.Section([html.H3('Datos y características reales'),html.P(f'{len(panel):,} fechas diarias entre {panel.index.min().date()} y {panel.index.max().date()}, derivadas de cierres por minuto. Los huecos se conservan; no se imputan precios.'),
            html.P('El dataset procesado principal contiene date y '+', '.join(panel.columns)+'. Las características del modelo se reconstruyen con las funciones originales:'),
            html.Ul([html.Li(name.replace('_',' ')) for name in FEATURES]),
            html.P('Se añaden siete variables de salida conocida de la ventana. Se usan lags de 7, 14, 21 o 28 días: 6L + 7 entradas (49, 91, 133 o 175). No hay volumen ni rango OHLC en este dataset procesado.')],className='panel'),
        html.Section([html.H3('Objetivo y horizonte'),dcc.Markdown(r'$$r_t=100\ln(P_t/P_{t-1}),\qquad \sigma_t^{(w)}=\sqrt{\frac1w\sum_{i=0}^{w-1}(r_{t-i}-\bar r_t)^2}$$'+ '\n\n'+r'$$y_{t,h}=\sigma_{t+h}^{(w)},\qquad h=1,\ldots,7;\quad w\in\{7,14,21,28\}$$',mathjax=True),
            html.P('Desviación estándar poblacional de retornos diarios (ddof=0), incluyendo el retorno del día t. Unidad: puntos porcentuales de volatilidad no anualizada. Los modelos aprenden correcciones relativas respecto a la volatilidad actual; se reconstruyen las predicciones en las unidades originales.')],className='panel'),
        graph_card('Calendario temporal',fig,f'{len(development)} fechas de development y {len(test)} orígenes completos de test. Los porcentajes excluyen los siete últimos días de 2025 sin target completo; validation está dentro de development y no se suma como partición independiente.'),
        html.Section([html.H3('Cortes crecientes de validación'),table(pd.DataFrame(fold_rows)),html.P(f'Ajuste final: {len(final_train)} orígenes elegibles; etiquetas terminan antes del 1 de enero de 2025. Separación comprobada: última etiqueta train anterior al primer origen validation.')],className='panel'),
        html.Section([html.H3('Metodología'),html.Div([html.Div(step,className='pipeline-step') for step in ['Cierres Binance 1 min','Características causales','Split temporal','Escalado solo train, cuando aplica','Entrenamiento','Predicción de 7 salidas','Evaluación alineada','Comparación por métrica']],className='pipeline')],className='panel'),
        html.Div([html.Div([html.H4(model),html.P(description)],className='stat-card') for model,description in DESCRIPTIONS.items()],className='cards'),
        html.Section([html.H3('Mejora propuesta: HAR-Ridge + XGBoost'),
            html.P('Se conservan los seis modelos anteriores. XGBoost se presenta como modelo individual; HAR-Ridge + XGBoost es un nuevo modelo propuesto que combina árboles con una regresión regularizada de resúmenes de volatilidad.'),
            table(pd.read_csv(ROOT/'results/improved_classical_2023_2025/macro_metrics.csv')
                  .query("symbol == 'GLOBAL_MACRO'")[['model','r2','rmse','mae']]
                  .replace({'model': {'HAR_Ridge_XGBoost': 'HAR-Ridge + XGBoost (mejora propuesta)',
                                      'XGBoost_reference': 'XGBoost (modelo individual)',
                                      'Persistence': 'Persistencia'}})),
            html.P('Promedios macro de 2025 sobre BTC, ETH, BNB y XRP y cuatro ventanas. El RMSE mejora un 3.73 % frente a XGBoost individual. La mejora no ocurre en todas las métricas: XRP tiene un MAE ligeramente mayor. Evaluación retrospectiva, sin prueba de significancia estadística La pestaña Comparación de modelos incluye los siete métodos con filtros comunes.')],className='panel'),
        html.Details([html.Summary('Auditoría de comparabilidad y archivos fuente'),table(audit),html.P('Solo se comparan resultados con las mismas claves activo/ventana/origen/horizonte y el mismo y real. Selecciones verificadas por hash; se comprueban fechas de ajuste y los objetivos contra el dataset.'),
            html.P('Se conservan seis modelos clásicos y se añade HAR-Ridge + XGBoost como mejora propuesta, sin redes neuronales. El ranking es descriptivo, no una prueba de superioridad estadística.')],className='panel'),
        html.A('Leer capítulo del Jupyter Book (Markdown)',href='/book-report',className='button')])
