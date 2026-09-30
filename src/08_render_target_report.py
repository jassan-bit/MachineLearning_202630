"""Actualiza el informe y notebook 2.1 desde las tablas ya calculadas.

Ejecutar después de 08_eda_target_development.py. No lee datos de TEST.
"""
from pathlib import Path
import csv
import re
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'outputs/tables'


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def render(write_notebook=True):
    summary = pd.read_csv(TABLES/'eda_target_summary.csv').set_index('symbol')
    corr = pd.read_csv(TABLES/'eda_target_correlations.csv')
    extremes = pd.read_csv(TABLES/'eda_target_extremes.csv').set_index('symbol')
    sensitivity = pd.read_csv(TABLES/'eda_target_sensitivity.csv')
    path = ROOT/'book/sections/02_eda.md'
    text = path.read_text(encoding='utf-8')
    first, rest = re.split(r'## 2\.2 Análisis unidimensional (?:de close|de las variables originales)', text, maxsplit=1)

    def table(header, rows):
        nonlocal first
        headers = header.strip('| ').split(' | ')
        pattern = re.escape(header) + r'\n(?:\|[^\n]*\n?)+'
        first, count = re.subn(pattern, lambda _: markdown_table(headers, rows)+'\n', first, count=1)
        assert count == 1, header

    def paragraph(start, replacement):
        nonlocal first
        first, count = re.subn(re.escape(start)+r'[^\n]*', lambda _: replacement, first, count=1)
        assert count == 1, start

    definition = r'''### 2.1.1 Definición y disponibilidad temporal

La variable objetivo es la **desviación estándar de los retornos horarios de las próximas 24 horas**, una magnitud continua. Se adopta la fórmula de volatilidad del profesor, adaptando la frecuencia de días a horas y fijando una ventana de 24 retornos. No es la volatilidad acumulada del retorno de un día.

Sea $C_{i,s}$ el cierre de la vela del activo $i$ cuya apertura UTC es $s$. Se define:

$$
r_{i,s}=\ln\left(\frac{C_{i,s}}{C_{i,s-1}}\right),\qquad
\bar r^+_{i,s}=\frac1{24}\sum_{j=1}^{24}r_{i,s+j},
$$

$$
\boxed{y_{i,s}=100\sqrt{\frac1{24}\sum_{j=1}^{24}(r_{i,s+j}-\bar r^+_{i,s})^2}}.
$$

El divisor es **24**, no 23: se utiliza `ddof=0`. El factor 100 expresa el resultado en porcentaje; no se anualiza ni se multiplica por $\sqrt{24}$. Esta es la misma definición y escala del objetivo de las secciones 2.3 y 3.

La predicción se emite **después del cierre de la vela ancla $s$ y una vez disponible ese cierre**, alrededor de $s+1$ hora. El objetivo utiliza los retornos $r_{i,s+1},\ldots,r_{i,s+24}$, construidos a partir de los 25 cierres $C_{i,s},\ldots,C_{i,s+24}$. Solo se conoce después del cierre de la última vela del horizonte, alrededor de $s+25$ horas. Por ejemplo, para el ancla de apertura 10:00 se predice alrededor de las 11:00 y se evalúan los 24 retornos posteriores. No se predice a la apertura de la vela ancla usando su cierre futuro.

En la notación del profesor, $\sigma_t$ usa los retornos $r_{t-24},\ldots,r_{t-1}$. Con $t=s+1$, la referencia histórica disponible es $\sigma_{s+1}$ y el objetivo aquí almacenado es $y_{i,s}=100\sigma_{i,s+25}$. Así se reconcilia la notación de la fórmula con el índice `open_time` del código sin desplazar las observaciones una hora por error.

Las frecuencias y el desbalance de clases no aplican: el problema es de regresión temporal. Las variables explicativas utilizan únicamente la vela ya cerrada o información anterior.

'''
    start = first.index('### 2.1.1')
    stop = first.index('### 2.1.2')
    first = first[:start] + definition + first[stop:]
    table('| Activo | Mínimo (%) | Mediana (%) | Media (%) | P95 (%) | P99 (%) | Máximo (%) |',
          [[s]+[f'{r[k]:.4f}' for k in ['min','median','mean','p95','p99','max']] for s,r in summary.iterrows()])
    table('| Activo | Asimetría | Exceso de curtosis | Señalados por IQR | Porcentaje |',
          [[s,f'{r["skew"]:.3f}',f'{r.excess_kurtosis:.3f}',f'{int(r.iqr_flagged):,}',f'{r.iqr_percent:.2f}%'] for s,r in summary.iterrows()])
    table('| Activo | Asimetría original | Asimetría de ln(y) | Exceso de curtosis original | Exceso de curtosis de ln(y) |',
          [[s]+[f'{r[k]:.3f}' for k in ['skew','log_skew','excess_kurtosis','log_excess_kurtosis']] for s,r in summary.iterrows()])
    table('| Activo | Apertura de referencia (UTC) | Máximo (%) |',
          [[s,pd.Timestamp(r.anchor_open_time).strftime('%Y-%m-%d %H:%M'),f'{r["max"]:.4f}'] for s,r in extremes.iterrows()])
    table('| Activo | Correlación del objetivo a 1h | Correlación del objetivo a 24h |',
          [[s,f'{r.acf_1h:.3f}',f'{r.acf_24h:.3f}'] for s,r in summary.iterrows()])
    delta = (sensitivity.strict_mean-sensitivity.relaxed_mean).abs().max()
    paragraph('Como sensibilidad,', f'Como sensibilidad, aceptar cierres irregulares manteniendo las demás condiciones produciría 42,569 objetivos por activo. Añadiría 30 ventanas en BTC, ETH y BNB, y 5 en XRP y SOL. La diferencia absoluta máxima entre las medias estricta y relajada es {delta:.6f} puntos porcentuales; los máximos no cambian. Se mantiene la regla conservadora por consistencia temporal, no porque esta sensibilidad demuestre la validez de los cierres irregulares. Véase la [tabla de sensibilidad](../../outputs/tables/eda_target_sensitivity.csv).')
    assert summary.zero.sum() == 0
    paragraph('No se observan objetivos iguales a cero.', f'No se observan objetivos iguales a cero. SOL presenta la mayor mediana ({summary.loc["SOLUSDT","median"]:.4f}%) y BTC la menor ({summary.loc["BTCUSDT","median"]:.4f}%). XRP alcanza el máximo más alto ({summary.loc["XRPUSDT","max"]:.4f}%), aunque su mediana es inferior a la de SOL. El nivel habitual y la intensidad de los episodios extremos deben distinguirse. Los cuartiles y la desviación estándar están en el [resumen completo](../../outputs/tables/eda_target_summary.csv). Las pequeñas diferencias de XRP y SOL respecto de 2.3.4 obedecen a que aquí se usan todas las etiquetas válidas de cada activo y allí la intersección de timestamps de los cinco.')
    paragraph('La asimetría es positiva', f'La asimetría es positiva en todos los activos y la media supera la mediana. El exceso de curtosis va de {summary.excess_kurtosis.min():.3f} a {summary.excess_kurtosis.max():.3f}, compatible con colas empíricas pronunciadas respecto de una referencia normal. No demuestra una ley de colas ni independencia. BNB y XRP presentan los mayores excesos de curtosis; XRP tiene la mayor proporción señalada por IQR.')
    paragraph('Como exploración inicial', r'Como exploración inicial se compara el objetivo con las nueve variables numéricas de la vela cerrada, el retorno de esa hora y la volatilidad pasada $100\,\operatorname{std}(r_{i,s-23},\ldots,r_{i,s};\mathrm{ddof}=0)$. Esta última centra los retornos en su media, utiliza divisor 24 y requiere 25 cierres consecutivos convencionales. Es la referencia disponible de Persistence en la sección 3; aquí solo se estudia su asociación, sin evaluar pronósticos.')
    past = corr.loc[corr.predictor.eq('rv_past_24h_pct')]
    paragraph('La volatilidad pasada de 24 horas muestra', f'La volatilidad pasada de 24 horas muestra correlaciones de Pearson entre {past.pearson.min():.3f} y {past.pearson.max():.3f}, y de Spearman entre {past.spearman.min():.3f} y {past.spearman.max():.3f}. Sus retornos no se solapan con los futuros, aunque comparten el cierre de frontera. La asociación motiva la referencia Persistence, cuyo desempeño se evalúa separadamente en la sección 3.')
    volume = corr.loc[corr.predictor.eq('quote_asset_volume')].set_index('symbol')
    trades = corr.loc[corr.predictor.eq('number_of_trades')]
    values = ', '.join(f'{s}: {volume.loc[s,"spearman"]:.3f}' for s in summary.index)
    paragraph('El volumen en USDT presenta', f'El volumen en USDT presenta asociaciones Spearman distintas por activo ({values}). Para el número de operaciones, Spearman va de {trades.spearman.min():.3f} a {trades.spearman.max():.3f}. Estas diferencias no establecen utilidad predictiva fuera de muestra; las asociaciones de todos los campos de volumen se conservan en la tabla completa.')
    returns = corr.loc[corr.predictor.eq('return_1h_pct')]
    paragraph('Los precios OHLC tienen' if 'Los precios OHLC tienen' in first else 'Los cuatro precios OHLC muestran', f'Los precios OHLC tienen asociaciones descriptivas que pueden reflejar tendencias y regímenes; no se interpretan como causalidad. El retorno horario con signo presenta Pearson entre {returns.pearson.min():.3f} y {returns.pearson.max():.3f}. Su asociación lineal pequeña no descarta relaciones no lineales ni relaciones con su magnitud absoluta. Los diagramas muestran dispersión y extremos que un coeficiente aislado no resume.')
    paragraph('Las métricas comunes son' if 'Las métricas comunes son' in first else 'Se propone reportar MAE y RMSE', 'Las métricas comunes son RMSE, MAPE y $R^2$; MAE es complementaria. RMSE y MAE se expresan en puntos porcentuales de volatilidad, MAPE en porcentaje y $R^2$ es adimensional. Se reportan por activo y fold antes del promedio con pesos iguales. MAPE es indefinido con objetivos cero y sensible a valores cercanos a cero; no se añaden denominadores artificiales. $R^2$ requiere variabilidad del objetivo. Esta sección describe el objetivo; las métricas predictivas se presentan en la sección 3.')
    paragraph('El cálculo está en', 'El cálculo está en `src/08_eda_target_development.py`; `src/08_render_target_report.py` sincroniza estas tablas e interpretaciones con los resultados guardados. Ejecutar ambos scripts en ese orden desde el entorno de `requirements.txt`. Los resultados y metadatos se guardan en `outputs/tables/eda_target_*`; la tabla derivada permanece separada de los datos originales. Cada objetivo válido se contrasta con el cálculo directo centrado de sus 24 retornos, con divisor 24; se verifica el borde de DEVELOPMENT y su SHA-256. El notebook ejecuta el cálculo y muestra las cuatro figuras. No se consulta TEST ni se reentrenan modelos para esta corrección.')
    rest = rest.replace('Estos resultados son nuevos y no reutilizan las cifras de la definición anterior de 2.1.', 'La definición coincide con la sección 2.1; las muestras descriptivas se especifican en cada comparación.')
    path.write_text(first+'## 2.2 Análisis unidimensional de las variables originales'+rest, encoding='utf-8')

    base_path = ROOT/'book/sections/01_base_datos.md'
    base = base_path.read_text(encoding='utf-8')
    a = base.index('#### Rango de la variable objetivo')
    b = base.index('### 1.6.3', a)
    ranges = markdown_table(['Activo','Mínimo (%)','Máximo (%)','Objetivos válidos'],
                            [[s,f'{r["min"]:.4f}',f'{r["max"]:.4f}',f'{int(r.n):,}'] for s,r in summary.iterrows()])
    base = base[:a] + '#### Rango de la variable objetivo\n\nCalculado únicamente sobre DEVELOPMENT con la fórmula del profesor: desviación estándar centrada de 24 retornos horarios futuros, divisor 24 (`ddof=0`), expresada en porcentaje. La construcción y distribución se documentan en la sección 2.1.\n\n'+ranges+'\n\n'+base[b:]
    base_path.write_text(base,encoding='utf-8')

    master_path = ROOT/'book/entregable1_master.md'
    master = master_path.read_text(encoding='utf-8')
    master = re.split(r'### (?:Alcance de esta decisión|Definición aplicada) y resultados existentes', master)[0] + '''### Definición aplicada y resultados existentes

El objetivo de la sección 2.1, sus tablas, figuras y notebook se recalcularon con la fórmula del profesor: desviación estándar de 24 retornos horarios futuros centrados en su media, divisor 24 (`ddof=0`) y escala porcentual. Esta definición coincide con el objetivo de las secciones 2.3 y 3 y con la referencia pasada de Persistence.

Se sustituye el cálculo anterior basado en la raíz de la suma de retornos al cuadrado. Esa magnitud difiere en el centrado y el divisor; sus cifras no se conservan como resultados de la definición vigente. La actualización utiliza exclusivamente DEVELOPMENT, conserva los datos originales y no requiere consultar TEST ni repetir el ajuste del SVR. Las diferencias de cobertura entre EDA y modelado se deben a los históricos adicionales y a la intersección de timestamps elegibles, no a otra definición del objetivo.
'''
    master_path.write_text(master,encoding='utf-8')
    model_path = ROOT/'book/sections/03_modelo_base.md'
    model_text = model_path.read_text(encoding='utf-8').replace('Las cifras antiguas de 2.1 no se reutilizan para construir el objetivo.', 'La sección 2.1 utiliza la misma definición del objetivo; cada experimento conserva sus propias reglas de cobertura e historial disponible.')
    model_path.write_text(model_text,encoding='utf-8')

    checklist_path = TABLES/'entregable1_checklist.csv'
    with checklist_path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        columns, checklist = reader.fieldnames, list(reader)
    for row in checklist:
        if row['requisito'].startswith('1.16 ') or (row['seccion'].startswith('2.1 ') and row['estado'] != 'NO APLICA'):
            row.update(estado='RESPONDIDO', evidencia='Secciones 1.6.2 y 2.1: fórmula del profesor; resultados recalculados con ddof=0.',
                       archivo_resultado='outputs/tables/eda_target_summary.csv',
                       observaciones='Solo DEVELOPMENT. Distribución, log diagnóstico, extremos, correlaciones y tiempo actualizados; Box-Cox discutido sin ajuste global. La corrección no modifica los folds del modelo.')
        if row['requisito'] == 'Criterio común de comparabilidad entre modelos':
            row['observaciones'] = 'EDA 2.1 recalculado con la misma definición del objetivo de SVR/Persistence; TEST no consultado.'
    with checklist_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(checklist)

    if write_notebook:
        description = re.sub(r'```\{figure\}[^\n]*\n.*?```', '', first, flags=re.S)
        description = description.replace('../../outputs/', '../outputs/')
        intro = '# EDA del objetivo: fórmula del profesor\n\nEjecutar todas las celdas recalcula el EDA únicamente sobre DEVELOPMENT. El código fuente reside en `src/08_eda_target_development.py`; el informe se sincroniza desde las tablas mediante `src/08_render_target_report.py`. La interpretación guardada corresponde al snapshot de datos documentado.\n'
        code = '''from pathlib import Path
import runpy
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p/'src/08_eda_target_development.py').is_file())
runpy.run_path(str(ROOT/'src/08_eda_target_development.py'), run_name='__main__')
report = runpy.run_path(str(ROOT/'src/08_render_target_report.py'))
report['render'](write_notebook=False)
'''
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro),
             nbformat.v4.new_code_cell(code), nbformat.v4.new_markdown_cell(description)])
        for name in ['distribution','time','correlations','relations']:
            nb.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(ROOT/'book/_static/figures/eda_target_{name}.png')))"))
        nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
        nbformat.write(nb,ROOT/'notebooks/08_eda_target_development.ipynb')
    print('Informe 2.1, rango del objetivo y presentación sincronizados.')


if __name__ == '__main__':
    render()
