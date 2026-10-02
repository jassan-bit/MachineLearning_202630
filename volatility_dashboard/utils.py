import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, dash_table
from statsmodels.tsa.stattools import acf

COLORS = {'k-NN':'#2563eb','Ridge':'#0891b2','Lasso':'#7c3aed','Random Forest':'#059669','XGBoost':'#d97706','SVR Lineal':'#dc2626','MLP':'#0f172a'}
METRICS = {'rmse':'RMSE ↓','mae':'MAE ↓','r2':'R² ↑'}


def polish(fig):
    fig.update_layout(template='plotly_white', font={'family':'Arial','color':'#243247'},
                      margin={'l':45,'r':20,'t':55,'b':45}, paper_bgcolor='white',
                      legend={'orientation':'h','y':-0.22}, hovermode='closest')
    return fig


def empty(title,message):
    fig = go.Figure()
    fig.add_annotation(text=message,x=.5,y=.5,xref='paper',yref='paper',showarrow=False)
    fig.update_layout(title=title)
    return polish(fig)


def graph_card(title,figure,note):
    return html.Section([html.H3(title),dcc.Graph(figure=polish(figure),config={'displaylogo':False}),html.P(note,className='interpretation')],className='panel')


def table(frame, page_size=12):
    clean = frame.copy()
    for col in clean:
        if pd.api.types.is_float_dtype(clean[col]):
            clean[col] = clean[col].round(6)
        elif clean[col].dtype == object:
            clean[col] = clean[col].map(lambda v: str(v) if isinstance(v,(list,dict,tuple)) else v)
    clean = clean.astype(object).where(pd.notna(clean),None)
    return dash_table.DataTable(data=clean.to_dict('records'),columns=[{'name':str(c),'id':str(c)} for c in clean.columns],
        sort_action='native',filter_action='native',page_size=page_size,export_format='csv',
        style_table={'overflowX':'auto'},style_cell={'fontFamily':'Arial','padding':'10px','textAlign':'left','minWidth':'95px','maxWidth':'320px','whiteSpace':'normal'},
        style_header={'backgroundColor':'#eef3f9','fontWeight':'bold'},style_data_conditional=[{'if':{'row_index':'odd'},'backgroundColor':'#f8fafc'}],
        tooltip_header={str(c):str(c) for c in clean.columns},tooltip_duration=None)


def acf_figure(values,title):
    values = np.asarray(values,float)
    finite = np.isfinite(values)
    if finite.sum() < 5 or np.nanstd(values)==0:
        return empty(title,'Observaciones insuficientes o serie constante.'), f'{finite.sum()} valores finitos; ACF no estimable.'
    nlags = min(30,len(values)-2,max(1,int(finite.sum())//3))
    coefficients = acf(values,nlags=nlags,fft=False,missing='conservative')
    fig = go.Figure(go.Bar(x=list(range(len(coefficients))),y=coefficients,marker_color='#2563eb'))
    fig.update_layout(title=title,xaxis_title='Lag (días)',yaxis_title='Autocorrelación')
    return fig,f'{finite.sum()} observaciones finitas; ACF lag 1 = {coefficients[1]:.3f}. Los huecos se conservan; esta dependencia no demuestra causalidad.'
