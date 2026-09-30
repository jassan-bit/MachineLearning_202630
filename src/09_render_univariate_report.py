"""Incorpora todas las variables originales al EDA, sin ampliar el modelo."""
from pathlib import Path
import csv
import hashlib
import json
import re
import runpy
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/tables'


def table(headers, rows):
    headers = [str(v).replace('|', '\\|') for v in headers]
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v).replace('|', '\\|') for v in row)+' |' for row in rows])


def render(write_notebook=True):
    meta = json.loads((OUT/'univariate_metadata.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/meta['source']).read_bytes()).hexdigest() == meta['sha256']
    stats = pd.read_csv(OUT/'univariate_summary.csv')
    types = pd.read_csv(OUT/'univariate_types.csv')
    times = pd.read_csv(OUT/'univariate_temporal.csv')
    cats = pd.read_csv(OUT/'univariate_categories.csv')
    labels = runpy.run_path(str(ROOT/'src/09_univariate_development.py'))['LABELS']
    assert len(stats) == 45 and len(types) == 12
    extra = ['### 2.2.8 Cobertura de las doce columnas originales',
        'El EDA univariado cubre las nueve columnas numéricas, las dos marcas temporales y `symbol`. La inclusión de una variable en el EDA no implica incorporarla al SVR. Todos los resúmenes corresponden a DEVELOPMENT antes de filtrar ventanas; no se leen valores de TEST.',
        table(['Variable', 'Tipo', 'Cardinalidad global', 'Nulos'],
              [[r.variable, r.type, r.cardinality, r.missing] for r in types.itertuples()]),
        'Los resúmenes numéricos se separan por activo: tienen 42.833 observaciones cada uno. Se usa desviación estándar descriptiva con `ddof=1`, percentiles interpolados linealmente y exceso de curtosis de Fisher (referencia normal cero). El objetivo conserva `ddof=0`. Los histogramas tienen eje de frecuencias logarítmico; no se transforman las observaciones. Los boxplots usan 1,5 IQR. Los umbrales globales de DEVELOPMENT son descriptivos y no se reutilizan para limpiar los folds.',
        '[Tabla completa con mínimos, máximos, cuartiles, percentiles 1/5/95/99, ceros y límites IQR](../../outputs/tables/univariate_summary.csv). No se aplican pruebas marginales de normalidad: no son un supuesto del SVR y la dependencia temporal impide interpretar sus p-valores iid de manera convencional.']
    for j, (field, (label, unit)) in enumerate(labels.items(), 9):
        if field == 'close':
            continue
        g = stats.loc[stats.variable.eq(field)].sort_values('symbol')
        extra += [f'#### {label}: `{field}`', f'Unidad: {unit}.',
            table(['Activo', 'Media', 'Mediana', 'DE', 'P5', 'P95'],
                  [[r.symbol]+[f'{getattr(r,k):.5g}' for k in ['mean','median','std','p05','p95']] for r in g.itertuples()]),
            table(['Activo', 'Asimetría', 'Exceso curtosis', 'Señaladas IQR', '% IQR'],
                  [[r.symbol,f'{r.skew:.3f}',f'{r.excess_kurtosis:.3f}',r.flagged,f'{r.percent:.2f}'] for r in g.itertuples()])]
        top = g.loc[g.percent.idxmax()]
        comment = (f'La mayor proporción señalada por IQR es {top.percent:.2f} % en {top.symbol}. '
                   'Estas marcas describen colas respecto al rango intercuartílico; no demuestran errores ni justifican eliminar registros. ')
        if field in ['open','high','low']:
            comment += 'Los niveles de precio mezclan periodos y regímenes; su dispersión no equivale a la volatilidad de retornos. Comparar precios absolutos entre activos no mide cuál tiene mayor riesgo.'
        elif field in ['volume','taker_buy_base_asset_volume']:
            comment += 'Las cantidades están en unidades de cada criptomoneda y no son directamente comparables entre activos. Los extremos pueden reflejar actividad concentrada; no se atribuyen a eventos sin evidencia adicional.'
        elif field == 'number_of_trades':
            comment += 'Es un conteo horario discreto de operaciones, no de personas ni de participantes independientes. Su dispersión describe actividad, sin demostrar capacidad predictiva de volatilidad futura.'
        else:
            comment += 'La unidad USDT facilita comparar cantidades cotizadas; los cambios de precio y de actividad siguen afectando la distribución. El volumen comprador taker, cuando corresponde, es un subconjunto del volumen y no representa todo el flujo comprador.'
        extra += [comment,
            f'```{{figure}} ../_static/figures/univariate_{field}.png\n:alt: Histogramas y boxplots de {field} por activo.\n\n{label}: cinco activos, sin mezclar sus escalas.\n```']
    extra += ['### 2.2.9 Identificador y marcas temporales',
        table(['Activo', 'Frecuencia', '%'], [[r.symbol,r.frequency,f'{r.percent:.1f}'] for r in cats.itertuples()]),
        'Las cinco categorías tienen la misma frecuencia y no hay categorías raras. `symbol` identifica series y no es una clase objetivo. Se mantiene como clave de agrupación; no necesita codificación en los modelos separados por activo.',
        '```{figure} ../_static/figures/univariate_symbol.png\n:alt: Frecuencia de cada activo en DEVELOPMENT.\n\nDistribución de symbol.\n```',
        table(['Activo','Campo','Cardinalidad','Inicio UTC','Fin UTC'],
              [[r.symbol,r.variable,r.cardinality,r.min,r.max] for r in times.itertuples()]),
        'Las fechas se interpretan como coordenadas temporales, no como magnitudes con media o normalidad marginal. La distribución mensual registra exposición: los meses extremos son parciales y los meses tienen distinta duración; sus conteos no demuestran estacionalidad del mercado. El análisis temporal audita los huecos y los cierres no convencionales. Estos resúmenes incluyen registros originales, mientras que el modelado excluye las ventanas afectadas.',
        '```{figure} ../_static/figures/univariate_time_coverage.png\n:alt: Número de velas observadas por mes y activo.\n\nCobertura mensual; los conteos coinciden entre los cinco activos.\n```',
        '### 2.2.10 Reproducción del análisis completo',
        'Ejecutar `python src/09_univariate_development.py` y `python src/09_render_univariate_report.py`. Los metadatos registran la huella del archivo, versiones y convenciones estadísticas. El informe comprueba la huella antes de incorporar los resultados. [Notebook de las doce columnas](../../notebooks/09_univariate_development.ipynb). No se imputan ni eliminan valores por este análisis.']
    added = '\n\n'.join(extra)
    path = ROOT/'book/sections/02_eda.md'
    text = path.read_text(encoding='utf-8')
    text = text.replace('## 2.2 Análisis unidimensional de close', '## 2.2 Análisis unidimensional de las variables originales')
    before, after = text.split('## 2.3 Análisis bidimensional', 1)
    before = before.split('### 2.2.8 ', 1)[0].rstrip()
    before = before.replace('Las demás columnas no se analizan en esta sección; las fechas y los símbolos se utilizan únicamente para identificar las observaciones y separar los activos.',
        'Los apartados 2.2.1–2.2.7 detallan close; los apartados siguientes completan el análisis de las demás columnas, incluidas las fechas y los símbolos.')
    before = before.replace('Los archivos del análisis más amplio se conservan como antecedentes; no forman parte de la presentación de esta sección.',
        'Los resultados de las restantes columnas se incorporan a continuación como parte del EDA completo.')
    path.write_text(before+'\n\n'+added+'\n\n## 2.3 Análisis bidimensional'+after, encoding='utf-8')
    checklist_path = OUT/'entregable1_checklist.csv'
    with checklist_path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f); fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        if row['requisito'].startswith('2.2.'):
            row.update(estado='RESPONDIDO', evidencia='Sección 2.2: doce columnas originales, 45 resúmenes numéricos por activo, gráficos e interpretación.',
                       archivo_resultado='outputs/tables/univariate_summary.csv',
                       observaciones='Normalidad marginal no requerida; no se aplican pruebas iid. EDA descriptivo, sin selección ni limpieza basada en TEST.')
    with checklist_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    if write_notebook:
        code = '''from pathlib import Path
import runpy
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/09_univariate_development.py').is_file())
RECALCULAR = False
if RECALCULAR:
    runpy.run_path(str(ROOT/'src/09_univariate_development.py'), run_name='__main__')
runpy.run_path(str(ROOT/'src/09_render_univariate_report.py'))['render'](False)
display(pd.read_csv(ROOT/'outputs/tables/univariate_types.csv'))
display(pd.read_csv(ROOT/'outputs/tables/univariate_summary.csv'))
'''
        description = re.sub(r'```\{figure\}[^\n]*\n.*?```', '', added, flags=re.S).replace('../../outputs/', '../outputs/').replace('../../notebooks/', './')
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Análisis univariado completo de DEVELOPMENT'),
            nbformat.v4.new_code_cell(code), nbformat.v4.new_markdown_cell(description)])
        for field in [*labels, 'symbol', 'time_coverage']:
            nb.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(ROOT/'book/_static/figures/univariate_{field}.png')))"))
        nb.metadata.kernelspec = dict(name='python3', display_name='Python 3', language='python')
        nbformat.write(nb, ROOT/'notebooks/09_univariate_development.ipynb')
    print('EDA univariado completo integrado.')


if __name__ == '__main__':
    render()
