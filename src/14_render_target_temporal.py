"""Integra el análisis temporal horario del objetivo y sus límites inferenciales."""
from pathlib import Path
import hashlib
import json
import re
import runpy
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/tables'
table = runpy.run_path(str(ROOT/'src/09_render_univariate_report.py'))['table']


def render(write_notebook=True):
    meta = json.loads((OUT/'temporal_target_metadata.json').read_text(encoding='utf-8'))
    assert meta['development_sha256'] == hashlib.sha256((ROOT/'data/splits/development_80.csv').read_bytes()).hexdigest()
    coverage = pd.read_csv(OUT/'temporal_target_coverage.csv')
    tests = pd.read_csv(OUT/'temporal_target_stationarity.csv')
    corr = pd.read_csv(OUT/'temporal_target_correlations.csv')
    quarter = pd.read_csv(OUT/'temporal_target_quarters.csv')
    cycles = pd.read_csv(OUT/'temporal_target_cycles.csv')
    txt = ['### 2.6.10 Análisis horario del objetivo definitivo',
        'El análisis anterior del precio se complementa con el objetivo realmente pronosticado: desviación estándar centrada de los 24 retornos logarítmicos horarios futuros, multiplicada por 100, con divisor 24 y sin anualización. Cada etiqueta exige 25 cierres consecutivos y convencionales. Se conserva la rejilla horaria, se invalidan las ventanas afectadas por huecos y se excluyen las últimas 24 anclas de DEVELOPMENT. TEST no se abre.',
        '**Este análisis es retrospectivo.** El objetivo futuro, sus momentos móviles y su descomposición STL no son entradas disponibles al emitir la predicción; no se incorporan al pipeline. Se incluyen para diagnosticar la serie. Los patrones de calendario se agrupan por el instante nominal de emisión (apertura del ancla + 1 hora, UTC).',
        'La serie completa, los momentos móviles y los ciclos utilizan todas las etiquetas válidas de DEVELOPMENT. ACF/PACF, ADF, KPSS y STL requieren una secuencia regular: se usa exclusivamente el tramo válido consecutivo más largo, elegido por cobertura, sin interpolar. Los resultados de ese tramo no representan necesariamente los periodos excluidos.',
        table(['Activo','Etiquetas válidas','n del tramo','Inicio ancla UTC','Fin ancla UTC'],
              [[r.symbol,r.total_valid,r.segment_n,r.segment_start,r.segment_end] for r in coverage.itertuples()]),
        '#### Dependencia horaria y efecto del solapamiento',
        'Se calculan ACF y PACF hasta 168 horas; PACF utiliza Levinson–Durbin sin ajuste de sesgo (`ldbiased`). Los primeros rezagos se resumen a continuación. No se presentan bandas iid como evidencia inferencial. Los objetivos consecutivos comparten 23 retornos: una ACF elevada no demuestra capacidad predictiva ni independencia de las filas. Como contraste descriptivo se calcula la ACF de los retornos absolutos y al cuadrado, que no son objetivos de 24 horas superpuestos.',
        table(['Activo','Rezago h','ACF y','PACF y','ACF |r|','ACF r²'],
              [[r.symbol,r.lag_hours]+[f'{getattr(r,k):.4f}' for k in ['target_acf','target_pacf','abs_return_acf','squared_return_acf']]
              for r in corr.loc[corr.lag_hours.isin([1,24,168])].itertuples()]),
        f'La ACF del objetivo a una hora va de {corr.loc[corr.lag_hours.eq(1),"target_acf"].min():.4f} a {corr.loc[corr.lag_hours.eq(1),"target_acf"].max():.4f} entre activos; a 168 horas va de {corr.loc[corr.lag_hours.eq(168),"target_acf"].min():.4f} a {corr.loc[corr.lag_hours.eq(168),"target_acf"].max():.4f}. La dependencia no se limita al primer rezago. Su interpretación requiere considerar tanto el solapamiento como la evolución del proceso.',
        'La ACF de retornos transformados se calcula en su propio tramo consecutivo más largo, cuyos límites se guardan en el CSV; no se supone que coincida exactamente con el del objetivo. La relación con los rezagos de close continúa documentada en 2.6.5 y no se interpreta como causalidad.',
        '#### Estacionariedad: hipótesis y límites',
        'ADF se ejecuta con constante, máximo 48 rezagos horarios y selección AIC dentro de ese máximo; su hipótesis nula es raíz unitaria. KPSS se ejecuta con constante y rezagos automáticos; su hipótesis nula es estacionariedad en nivel. Se reportan ambas salidas; no rechazar una hipótesis no equivale a demostrarla. El horizonte solapado, la selección de tramo y los cambios de distribución limitan la extrapolación. Son diez contrastes exploratorios, no un criterio de selección de variables ni de hiperparámetros; no se declara significación conjunta.',
        table(['Activo','ADF estadístico','ADF p','Rezagos ADF','KPSS estadístico','KPSS p tabulado','Rezagos KPSS'],
              [[r.symbol,f'{r.adf_stat:.4f}',f'{r.adf_p:.5g}',r.adf_lags,f'{r.kpss_stat:.4f}',
                '≤0.01' if r.kpss_p==.01 else ('≥0.10' if r.kpss_p==.1 else f'{r.kpss_p:.5f}'),r.kpss_lags] for r in tests.itertuples()])]
    interpretations=[]
    for r in tests.itertuples():
        interpretations.append(f'{r.symbol}: ADF '+('rechaza' if r.adf_p<.05 else 'no rechaza')+
            ' raíz unitaria y KPSS '+('rechaza' if r.kpss_p<.05 else 'no rechaza')+
            ' estacionariedad en nivel, usando 0,05 solo como referencia descriptiva.')
    txt += [' '.join(interpretations),
        'Los p-valores KPSS pueden ser límites de la tabla (0,01 o 0,10), no valores exactos. Las advertencias originales se conservan en `temporal_target_stationarity.csv`. Rechazar ambas hipótesis puede indicar que ninguna simplificación describe bien el tramo; no se resuelve declarando la serie estacionaria por una sola prueba.',
        '#### Calendario, momentos móviles y evolución trimestral',
        'Los paneles incluyen la media y la varianza móviles sobre 168 horas consecutivas con `min_periods=168`; no se calculan a través de faltantes. Las agregaciones diaria, semanal y mensual son promedios de etiquetas horarias válidas, no volatilidades recalculadas a otra frecuencia; la cobertura puede variar. Los boxplots muestran hora, día de semana y mes de emisión, ocultando únicamente los puntos extremos para facilitar la lectura, sin eliminarlos de las estadísticas.',
        'La STL usa periodo diario de 24 horas, ajuste robusto e interpolación de los suavizadores cada tres puntos (`seasonal_jump=trend_jump=low_pass_jump=3`). Es una descomposición retrospectiva del tramo, no un filtro causal ni una prueba de estacionalidad estable. Se muestran tendencia, componente estacional y residuo. La descomposición semanal del precio analizada previamente responde a otra serie y frecuencia.',
        table(['Activo','Mediana mínima–máxima por hora (%)','Día semanal de menor mediana','Día semanal de mayor mediana'],
              [[s,f'{g.loc[g.cycle.eq("hour"),"median"].min():.4f} a {g.loc[g.cycle.eq("hour"),"median"].max():.4f}',
                ['lunes','martes','miércoles','jueves','viernes','sábado','domingo'][int(g.loc[g.loc[g.cycle.eq('weekday'),'median'].idxmin(),'group'])],
                ['lunes','martes','miércoles','jueves','viernes','sábado','domingo'][int(g.loc[g.loc[g.cycle.eq('weekday'),'median'].idxmax(),'group'])]]
               for s,g in cycles.groupby('symbol')]),
        'El rango entre medianas por hora permite valorar la magnitud del ciclo horario; cada objetivo cubre un día completo, lo que suaviza diferencias intradiarias. Los días con menor y mayor mediana son resúmenes del periodo observado, no efectos causales ni una regla garantizada para el futuro.',
        table(['Activo','Mínima mediana trimestral (%)','Máxima mediana trimestral (%)','Rango de DE trimestral (pp)'],
              [[s,f'{g["median"].min():.4f}',f'{g["median"].max():.4f}',f'{g["std"].min():.4f} a {g["std"].max():.4f}'] for s,g in quarter.groupby('symbol')]),
        'Los resúmenes trimestrales comparan distribuciones del objetivo y complementan la deriva de close. Los trimestres extremos son parciales. Variación de la distribución marginal no prueba un cambio de la relación condicional entre predictores y objetivo (concept drift). La sección 2.6.11 amplía este diagnóstico con fechas candidatas de cambio e incertidumbre condicional y una cronología documentada de eventos. No se atribuyen causalmente los movimientos a esos eventos.',
        'Las diferencias por calendario mezclan años y regímenes: no demuestran efectos horarios permanentes. Los paneles por activo permiten revisar heterogeneidad sin mezclar niveles. Estas comprobaciones sustentan la validación cronológica, la evaluación por fold y la incertidumbre por bloques; no modifican retrospectivamente la configuración del modelo ni consultan TEST.']
    txt += [f'```{{figure}} ../_static/figures/temporal_target_{s}.png\n:alt: EDA temporal horario del objetivo y retornos de {s}.\n\n{s}: objetivo completo, agregaciones, calendario, dependencia y STL del tramo continuo.\n```' for s in coverage.symbol]
    txt += ['#### Reproducción',
        'Ejecutar `python src/14_temporal_target.py` y `python src/14_render_target_temporal.py`. Las tablas `outputs/tables/temporal_target_*.csv` conservan cobertura, ciclos, trimestres, componentes y diagnósticos. Los metadatos fijan convenciones y huella de DEVELOPMENT. [Notebook temporal del objetivo](../../notebooks/14_temporal_target.ipynb).']
    section='\n\n'.join(txt)
    path=ROOT/'book/sections/02_eda.md'
    before,after=path.read_text(encoding='utf-8').split('## 2.7 Componente espacial',1)
    addition='\n\n### 2.6.11 '+before.split('### 2.6.11 ',1)[1].rstrip() if '### 2.6.11 ' in before else ''
    before=before.split('### 2.6.10 ',1)[0].rstrip()
    path.write_text(before+'\n\n'+section+addition+'\n\n## 2.7 Componente espacial'+after,encoding='utf-8')
    if write_notebook:
        code='''from pathlib import Path
import runpy
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/14_temporal_target.py').is_file())
RECALCULAR = False
if RECALCULAR:
    runpy.run_path(str(ROOT/'src/14_temporal_target.py'), run_name='__main__')
runpy.run_path(str(ROOT/'src/14_render_target_temporal.py'))['render'](False)
display(pd.read_csv(ROOT/'outputs/tables/temporal_target_stationarity.csv'))
display(pd.read_csv(ROOT/'outputs/tables/temporal_target_coverage.csv'))
'''
        description=re.sub(r'```\{figure\}[^\n]*\n.*?```','',section,flags=re.S).replace('../../notebooks/','./')
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# EDA horario de la volatilidad futura'), nbformat.v4.new_code_cell(code),nbformat.v4.new_markdown_cell(description)])
        for s in coverage.symbol:
            nb.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(ROOT/'book/_static/figures/temporal_target_{s}.png')))"))
        nb.metadata.kernelspec=dict(name='python3',display_name='Python 3',language='python')
        nbformat.write(nb,ROOT/'notebooks/14_temporal_target.ipynb')
    print('Temporal horario del objetivo integrado.')


if __name__=='__main__':
    render()
