"""Informe y notebook de cambios de media y eventos documentados."""
from pathlib import Path
import hashlib,json,runpy,re
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tables'
table=runpy.run_path(str(ROOT/'src/09_render_univariate_report.py'))['table']


def render(write_notebook=True):
    cfg=json.loads((OUT/'change_events_protocol.json').read_text(encoding='utf-8'))
    meta=json.loads((OUT/'change_events_metadata.json').read_text(encoding='utf-8'))
    assert meta['development_sha256']==hashlib.sha256((ROOT/'data/splits/development_80.csv').read_bytes()).hexdigest()
    assert meta['protocol_sha256']==hashlib.sha256((OUT/'change_events_protocol.json').read_bytes()).hexdigest()
    changes=pd.read_csv(OUT/'change_points.csv')
    sensitivity=pd.read_csv(OUT/'change_points_sensitivity.csv')
    events=pd.read_csv(OUT/'event_windows.csv')
    main=changes.loc[changes.block_days.eq(14)]
    date=lambda x:str(x)[:10]
    text=['### 2.6.11 Puntos de cambio y cronología de eventos',
        '**Alcance retrospectivo.** Se estudia un cambio dominante de media, no se asume que la serie tenga exactamente dos regímenes ni se utiliza este análisis para modificar el SVR, sus variables o TEST. El protocolo de este diagnóstico está fijado en `change_events_protocol.json` antes de ejecutar sus cálculos. Es una ampliación posterior al EDA inicial, no un estudio confirmatorio preregistrado.',
        '#### Serie diaria sin solapamiento de retornos',
        'Se toma la etiqueta horaria de cada ancla de las 23:00 UTC y se asigna al día siguiente, su fecha nominal de emisión. Esa etiqueta contiene los retornos de las 00:00 a las 23:00 de ese día: conserva la misma desviación estándar centrada, divisor 24 y escala porcentual. No es una media de etiquetas ni volatilidad acumulada de un retorno diario. Días consecutivos no comparten retornos, aunque comparten el cierre de frontera y pueden seguir siendo dependientes.',
        'Solo se admiten días con los 25 cierres consecutivos y convencionales. Los puntos de cambio se calculan en el tramo diario completo más largo: 829 días, del 25 de marzo de 2023 al 30 de junio de 2025, común a los cinco activos. No se concatenan días separados por huecos. La cronología de eventos usa los días válidos de todo DEVELOPMENT; por eso también incluye 2022.',
        '#### Método y elección de fecha candidata',
        'Se ajustan dos medias por mínimos cuadrados y se examinan todas las divisiones con al menos 90 días a cada lado. La fecha candidata es el primer día del segundo segmento en la división que minimiza la suma de errores cuadrados. El estadístico es la reducción de esa suma respecto de una sola media, dividida por la suma total de cuadrados. Se contrasta su sensibilidad a mínimos de 60 y 180 días. Es un diagnóstico de media, sensible a extremos y cambios graduales; no detecta necesariamente cambios de varianza, todos los regímenes ni deriva condicional.',
        'El procedimiento siempre propone una división si la serie no es constante. Por ello una fecha candidata, por sí sola, **no demuestra una ruptura**. Para evaluar la mejora bajo una aproximación de media constante se remuestrea la serie centrada mediante bloques circulares de 7, 14 y 28 días, con 499 réplicas y semilla 42. En cada réplica se vuelve a buscar la mejor división. El valor p usa (1 + réplicas con mejora al menos tan grande)/(499 + 1), evitando ceros.',
        'Se aplica Holm a los cinco activos, por separado en cada longitud de bloque. El bootstrap supone dependencia local y estabilidad suficiente bajo la hipótesis nula; la muestra y los cambios graduales pueden incumplir esa aproximación. Los valores p son exploratorios. No se selecciona la longitud que produzca el menor p ni se interpreta la sensibilidad como tres confirmaciones independientes.',
        table(['Activo','Fecha candidata','Media antes (%)','Media después (%)','Diferencia (pp)','Reducción SSE (%)','p Holm, bloque 14d'],
              [[r.symbol,date(r.candidate_day),f'{r.mean_before:.4f}',f'{r.mean_after:.4f}',f'{r.delta:.4f}',f'{100*r.gain_ratio:.2f}',f'{r.p_holm:.3f}'] for r in main.itertuples()]),
        table(['Activo','p Holm 7d','p Holm 14d','p Holm 28d','Fechas candidatas al variar mínimo 60/90/180d'],
              [[s]+[f'{g.loc[g.block_days.eq(b),"p_holm"].iloc[0]:.3f}' for b in [7,14,28]]+
               [' / '.join(date(v) for v in sensitivity.loc[sensitivity.symbol.eq(s),'candidate_day'])]
               for s,g in changes.groupby('symbol')]),
        '#### Incertidumbre de la localización',
        'Condicionando al modelo de dos medias estimado, se remuestrean los residuos en bloques circulares **por separado dentro de cada segmento**, se añaden sus respectivas medias y se vuelve a estimar la división. Los percentiles 2,5 y 97,5 de 499 localizaciones dan el intervalo exploratorio mostrado. No son intervalos simultáneos ni contemplan la incertidumbre entre cero, uno o varios cambios; tampoco garantizan cobertura nominal si la forma de dos medias es inadecuada. Se calculan para las tres longitudes y se presenta 14 días como referencia fijada.',
        table(['Activo','Localización: percentil 2,5','Localización: percentil 97,5'],
              [[r.symbol,date(r.conditional_low),date(r.conditional_high)] for r in main.itertuples()])]
    unstable=[s for s,g in changes.groupby('symbol') if g.p_holm.lt(.05).nunique()>1]
    stable=[s for s,g in changes.groupby('symbol') if g.p_holm.lt(.05).all()]
    text += ['Con umbral exploratorio 0,05, mantienen rechazo en las tres longitudes: '+(', '.join(stable) or 'ningún activo')+
             '. La decisión cambia con la longitud en: '+(', '.join(unstable) or 'ningún activo')+
             '. Los intervalos amplios y esta sensibilidad impiden presentar todas las fechas como rupturas precisas o estables. Las dos medias describen el contraste del tramo; no autorizan a atribuirlo a un evento concreto.',
        '[Resultados por longitud de bloque](../../outputs/tables/change_points.csv) y [sensibilidad al tamaño mínimo del segmento](../../outputs/tables/change_points_sensitivity.csv).',
        '#### Cronología contrastada con fuentes primarias',
        'Se eligen tres eventos ilustrativos de tecnología, proveedor y regulación a partir de fuentes primarias. La lista no es exhaustiva y se incorpora retrospectivamente; no se seleccionan nuevos eventos buscando coincidencias con los máximos de estos gráficos. Las fuentes acreditan fecha y hecho, no su efecto en la volatilidad.',
        table(['Fecha documentada','Evento','Fuente primaria'],[[e['date'],e['name'],f"[Fuente]({e['source']})"] for e in cfg['events']]),
        'El comunicado de la SEC de 2023 se describe como anuncio de cargos, no como una sentencia ni como descripción del estado actual del litigio. La fecha de enero de 2024 corresponde a la aprobación de cotización de ETP spot de bitcoin, no a una aprobación general de las criptomonedas.',
        'Se comparan los 14 días calendario anteriores y los 14 posteriores a cada fecha, excluyendo el día del evento de las medias. Se utiliza una convención de días UTC, sin inventar una hora exacta de anuncio. Cada etiqueta está contenida en su día y no atraviesa de la ventana anterior a la posterior. Los gráficos sí muestran el día cero como contexto. Se conservan los conteos de días válidos; no se imputan días faltantes.',
        table(['Evento','Activo','n antes/después','Media antes (%)','Media después (%)','Diferencia (pp)'],
              [[r.event,r.symbol,f'{r.n_before}/{r.n_after}',f'{r.mean_before:.4f}',f'{r.mean_after:.4f}',f'{r.delta:.4f}'] for r in events.itertuples()]),
        'Las diferencias tienen signos y magnitudes distintos según evento y activo. Son asociaciones descriptivas alrededor de una fecha: pueden intervenir anticipación, otros anuncios, tendencia, condiciones de mercado y dependencia entre activos. No hay grupo de control ni identificación causal; no se estiman efectos causales ni se añaden p-valores iid a estas ventanas cortas. Los eventos y las fechas candidatas de cambio se investigan por separado y no se emparejan automáticamente.',
        '[Ventanas y fuentes](../../outputs/tables/event_windows.csv); [serie diaria y faltantes](../../outputs/tables/daily_nonoverlapping_volatility.csv).']
    text += [f'```{{figure}} ../_static/figures/change_events_{s}.png\n:alt: Cambio de media candidato y ventanas de eventos para {s}.\n\n{s}: tramo continuo y ventanas de calendario; relaciones descriptivas, no causales.\n```' for s in main.symbol]
    text += ['#### Reproducción',
        'Ejecutar `python src/22_change_events.py` y `python src/23_render_change_events.py`. Las pruebas verifican una ruptura conocida, una serie constante, invariancia de fecha ante cambios de escala y rechazo de faltantes. Los metadatos guardan las huellas del protocolo y DEVELOPMENT. [Notebook de cambios y eventos](../../notebooks/22_change_events.ipynb). TEST no se lee y los modelos predictivos no se modifican.']
    section='\n\n'.join(text)
    path=ROOT/'book/sections/02_eda.md'
    before,after=path.read_text(encoding='utf-8').split('## 2.7 Componente espacial',1)
    before=before.split('### 2.6.11 ',1)[0].rstrip()
    before=before.replace('Los gráficos permiten localizar periodos candidatos para estudiar cambios de régimen; no se han estimado puntos de cambio con incertidumbre ni atribuido movimientos a noticias o eventos externos. Esa atribución sigue sin evidencia y no se inventa.',
                          'La sección 2.6.11 amplía este diagnóstico con fechas candidatas de cambio e incertidumbre condicional y una cronología documentada de eventos. No se atribuyen causalmente los movimientos a esos eventos.')
    path.write_text(before+'\n\n'+section+'\n\n## 2.7 Componente espacial'+after,encoding='utf-8')
    if write_notebook:
        code='''from pathlib import Path
import runpy
import pandas as pd
from IPython.display import display, Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/22_change_events.py').is_file())
RECALCULAR=False
if RECALCULAR:
    runpy.run_path(str(ROOT/'src/22_change_events.py'),run_name='__main__')
runpy.run_path(str(ROOT/'src/23_render_change_events.py'))['render'](False)
display(pd.read_csv(ROOT/'outputs/tables/change_points.csv'))
display(pd.read_csv(ROOT/'outputs/tables/event_windows.csv'))
'''
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Cambios de media y eventos: diagnóstico retrospectivo'),nbformat.v4.new_code_cell(code),nbformat.v4.new_markdown_cell(re.sub(r'```\{figure\}.*?```','',section,flags=re.S).replace('../../outputs/','../outputs/').replace('../../notebooks/','./'))])
        for s in main.symbol:nb.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(ROOT/'book/_static/figures/change_events_{s}.png')))"))
        nb.metadata.kernelspec=dict(name='python3',display_name='Python 3',language='python')
        nbformat.write(nb,ROOT/'notebooks/22_change_events.ipynb')
    print('Cambios y eventos integrados.')


if __name__=='__main__':render()
