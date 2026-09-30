"""Construye un único notebook con informe, figuras y cálculo completo ejecutable."""
from pathlib import Path
import base64
import json
import re
import textwrap
import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'notebooks/Entregable_1_Completo.ipynb'
STAGES = [
    '04_missingness_development.cjs', '05_duplicates_development.cjs',
    '06_outliers_development.py', '07_consistency_development.py',
    '08_eda_target_development.py', '09_univariate_development.py',
    '11_bivariate_close.py', '14_temporal_close.py', '18_base_model.py',
    '12_multivariate_close.py', '13_feature_diagnostics.py',
    '14_temporal_target.py', '17_validate_pipeline.py',
    '19_extended_diagnostics.py', '22_change_events.py',
]
COMPARISONS = {
    'base_selection.csv':['mean_rmse'],
    'base_metrics_by_fold.csv':['rmse','mape','r2'],
    'base_confidence_intervals.csv':['estimate','low','high'],
    'base_learning_curve.csv':['train_rmse','validation_rmse'],
    'eda_target_summary.csv':['mean','median','std'],
    'univariate_summary.csv':['mean','median','std'],
    'multivariate_summary.csv':['pc1_ratio','effective_rank'],
    'leakage_all_features_scores.csv':['rmse','r2'],
    'temporal_target_stationarity.csv':['adf_stat','kpss_stat'],
    'base_residual_diagnostics_all.csv':['mean_residual','acf_1h'],
    'base_bootstrap_sensitivity.csv':['estimate','low','high'],
    'change_points.csv':['gain_ratio','p_raw','p_holm'],
    'event_windows.csv':['mean_before','mean_after'],
}


def report_cells(path):
    source=path.read_text(encoding='utf-8')
    source=re.sub(r'^---\n.*?\n---\n', '', source, count=1, flags=re.S)
    # Convertir enlaces del Book a enlaces relativos al notebook.
    def link(match):
        label,url=match.groups()
        if re.match(r'\w+://|#|attachment:',url): return match.group(0)
        target=(path.parent/url).resolve()
        try: relative=target.relative_to(ROOT).as_posix()
        except ValueError: return match.group(0)
        if relative=='notebooks/Entregable_1_Completo.ipynb': return label
        return f'[{label}](../{relative})'
    source=re.sub(r'\[([^\]]*)\]\(([^)]+)\)',link,source)
    def admonition(match):
        title,body=match.groups()
        lines=[line for line in body.strip().splitlines() if not line.startswith(':')]
        return '> **'+title+'**\n>\n'+'\n'.join('> '+line for line in lines)
    source=re.sub(r'```\{admonition\} ([^\n]+)\n(.*?)```',admonition,source,flags=re.S)
    cells=[]; last=0
    for match in re.finditer(r'```\{figure\} ([^\n]+)\n(.*?)```',source,re.S):
        preceding=source[last:match.start()].strip()
        if preceding: cells.append(nbformat.v4.new_markdown_cell(preceding))
        image_path=(path.parent/match.group(1).strip()).resolve()
        body=match.group(2)
        alt=re.search(r'^:alt: (.*)$',body,re.M)
        caption='\n'.join(line for line in body.strip().splitlines() if not line.startswith(':')).strip()
        mime='image/svg+xml' if image_path.suffix=='.svg' else 'image/png'
        data=image_path.read_text(encoding='utf-8') if mime.endswith('svg+xml') else base64.b64encode(image_path.read_bytes()).decode('ascii')
        cell=nbformat.v4.new_markdown_cell(f'![{alt.group(1) if alt else image_path.stem}](attachment:{image_path.name})\n\n{caption}')
        cell.attachments={image_path.name:{mime:data}}
        cells.append(cell); last=match.end()
    if source[last:].strip(): cells.append(nbformat.v4.new_markdown_cell(source[last:].strip()))
    return cells


def build():
    cells=[nbformat.v4.new_markdown_cell('''# Entregable 1 — Pronóstico de volatilidad en criptomonedas

**Estudiantes:** Jassan Arteta - Mateo Bernal  
**Profesor:** Lihki Rubio  
**Asignatura:** Machine Learning  
**Universidad del Norte — 2026**

## Lectura y ejecución

Este notebook reúne el informe completo, sus tablas, figuras e interpretaciones, y el código que reproduce los cálculos. Las figuras están incorporadas al archivo y se pueden consultar sin ejecutar las celdas.

Para **ejecutar todo**, descargar el repositorio completo, conservar este archivo en `notebooks/` e instalar las dependencias de `requirements-lock-windows-py310.txt` en Python 3.10. Se necesita Node para las dos comprobaciones iniciales de calidad. Seleccionar ese entorno como kernel y ejecutar **Restart Kernel and Run All Cells**. El archivo no instala paquetes ni descarga datos automáticamente.

La sección de código, después de las conclusiones, recalcula por defecto todos los análisis y los 170 ajustes predictivos. Crea una carpeta nueva bajo `outputs/`, utiliza únicamente el snapshot DEVELOPMENT y contrasta los resultados con las tablas del informe. La descarga y la partición originales se documentan como procedencia y no se repiten; TEST permanece reservado.

La ejecución puede tardar varios minutos según el equipo. Las salidas guardadas al final muestran el resultado de una ejecución completa de este mismo notebook.
''')]
    for name in ['entregable1_master.md','sections/01_base_datos.md','sections/02_eda.md','sections/03_modelo_base.md','sections/05_conclusiones.md']:
        cells.extend(report_cells(ROOT/'book'/name))
    cells.append(nbformat.v4.new_markdown_cell('''# 5. Código reproducible y resultados de ejecución

Las etapas se ejecutan en orden de dependencia computacional. Por eso la construcción de los folds del SVR precede al diagnóstico multivariado sobre TRAIN. Cada celda Python contiene el código completo de su etapa dentro de una función para mantener separadas sus variables. Las dos etapas JavaScript se muestran y se ejecutan con Node. Los módulos auxiliares y protocolos proceden de `src/` y `outputs/tables/` del repositorio.
'''))
    setup='''from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, time
import numpy as np
import pandas as pd
from IPython.display import display, Image

PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                     if (p/'data/splits/development_80.csv').is_file()
                     and (p/'src/18_base_model.py').is_file()), None)
assert PROJECT_ROOT is not None, 'Abrir el notebook dentro del proyecto descargado.'
NODE = shutil.which('node')
assert NODE is not None, 'Se necesita Node para las dos etapas JavaScript.'
SOURCE_DATA = PROJECT_ROOT/'data/splits/development_80.csv'
SOURCE_HASH = hashlib.sha256(SOURCE_DATA.read_bytes()).hexdigest()
assert SOURCE_HASH == '5fbbc54a12e976551b35652b72c989be8d1a120ef41cd7c6a4b50dcaf5fa87a2'
RUN_ROOT = PROJECT_ROOT/'outputs'/('compiled_notebook_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
RUN_ROOT.mkdir(exist_ok=False)
for folder in ['src','tests','data/splits','outputs/tables','outputs/models','book/sections','book/_static/figures','notebooks']:
    (RUN_ROOT/folder).mkdir(parents=True,exist_ok=True)
for folder,patterns in [('src',['*.py','*.cjs']),('tests',['*.py']),('book/sections',['*.md'])]:
    for pattern in patterns:
        for source in (PROJECT_ROOT/folder).glob(pattern):
            shutil.copy2(source,RUN_ROOT/folder/source.name)
for name in ['base_model_protocol.json','multivariate_protocol.json','change_events_protocol.json',
             'base_report_template.md','entregable1_checklist.csv']:
    shutil.copy2(PROJECT_ROOT/'outputs/tables'/name,RUN_ROOT/'outputs/tables'/name)
shutil.copy2(SOURCE_DATA,RUN_ROOT/'data/splits/development_80.csv')
assert sorted(p.name for p in (RUN_ROOT/'data/splits').iterdir()) == ['development_80.csv']
EXECUTION = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
                 development_sha256=SOURCE_HASH,test_read=False,test_copied=False,stages=[])
def record_stage(name, started):
    EXECUTION['stages'].append(dict(stage=name,seconds=round(time.perf_counter()-started,3),status='passed'))
    (RUN_ROOT/'execution.json').write_text(json.dumps(EXECUTION,indent=2),encoding='utf-8')
    print('Etapa completada:',name)
print('Python:',sys.version.split()[0])
print('Carpeta independiente:',RUN_ROOT)
print('Filas DEVELOPMENT:',len(pd.read_csv(SOURCE_DATA)))
print('SHA-256:',SOURCE_HASH)
'''
    cells.append(nbformat.v4.new_code_cell(setup))
    cells.append(nbformat.v4.new_markdown_cell('## 5.1 Dependencias y controles temporales'))
    cells.append(nbformat.v4.new_code_cell('''started=time.perf_counter()
for arguments in [['-m','pip','check'],['-m','unittest','discover','-s','tests','-v']]:
    result=subprocess.run([sys.executable,*arguments],cwd=RUN_ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    print(result.stdout); print(result.stderr)
    assert result.returncode==0, 'La comprobación debe completarse antes del análisis.'
record_stage('dependencias_y_12_pruebas',started)
'''))
    for i,name in enumerate(STAGES,2):
        source=(ROOT/'src'/name).read_text(encoding='utf-8-sig')
        cells.append(nbformat.v4.new_markdown_cell(f'## 5.{i} {name}'))
        if name.endswith('.cjs'):
            cells.append(nbformat.v4.new_markdown_cell('```javascript\n'+source+'\n```'))
            code=f'''started=time.perf_counter()
result=subprocess.run([NODE,str(RUN_ROOT/'src/{name}')],cwd=RUN_ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
print(result.stdout)
if result.stderr: print(result.stderr)
assert result.returncode==0
record_stage({name!r},started)
'''
        else:
            # Código visible y ejecutable; __file__ apunta a la copia independiente.
            code=f"started=time.perf_counter()\ndef ejecutar_etapa():\n    __file__=str(RUN_ROOT/'src/{name}')\n"+textwrap.indent(source,'    ')+f'\n\nejecutar_etapa()\nrecord_stage({name!r},started)\n'
        cells.append(nbformat.v4.new_code_cell(code))
    cells.append(nbformat.v4.new_markdown_cell('''## 5.17 Concordancia del informe con la ejecución

Se contrastan trece tablas numéricas, con `rtol=1e-5` y `atol=1e-8`. Las figuras del informe son las de la ejecución documentada; abajo se muestran además figuras generadas en esta ejecución y las métricas recalculadas. Se verifica que los 170 ajustes y las predicciones pertenecen a DEVELOPMENT y que el archivo original conserva su huella.
'''))
    final='COMPARISONS = '+repr(COMPARISONS)+'''
comparison_rows=[]
for name,columns in COMPARISONS.items():
    expected=pd.read_csv(PROJECT_ROOT/'outputs/tables'/name)[columns]
    actual=pd.read_csv(RUN_ROOT/'outputs/tables'/name)[columns]
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-8,equal_nan=True,err_msg=name)
    comparison_rows.append(dict(table=name,rows=len(actual),max_absolute_difference=float(np.nanmax(np.abs(actual.to_numpy()-expected.to_numpy())))))
display(pd.DataFrame(comparison_rows))
metadata=json.loads((RUN_ROOT/'outputs/tables/base_metadata.json').read_text())
assert metadata['fits']==170 and metadata['test_read'] is False
assert SOURCE_HASH==hashlib.sha256(SOURCE_DATA.read_bytes()).hexdigest()
assert len(EXECUTION['stages'])==16
metrics=pd.read_csv(RUN_ROOT/'outputs/tables/base_metrics_by_fold.csv')
display(metrics.groupby('model')[['rmse','mape','r2']].mean())
display(pd.read_csv(RUN_ROOT/'outputs/tables/base_confidence_intervals.csv'))
display(Image(filename=str(RUN_ROOT/'book/_static/figures/eda_target_distribution.png')))
display(Image(filename=str(RUN_ROOT/'book/_static/figures/base_BTCUSDT.png')))
EXECUTION.update(status='passed',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 comparisons=comparison_rows,model_fits=metadata['fits'])
(RUN_ROOT/'execution.json').write_text(json.dumps(EXECUTION,indent=2),encoding='utf-8')
print('Ejecución completa: 170 ajustes, 13 tablas concordantes y TEST reservado.')
'''
    cells.append(nbformat.v4.new_code_cell(final))
    nb=nbformat.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec=dict(name='python3',display_name='Python 3',language='python')
    nb.metadata.language_info=dict(name='python',version='3.10.21')
    nb.metadata['project']=dict(title='Entregable 1 completo',test_reserved=True,default_execution='full_recomputation')
    nbformat.validate(nb)
    nbformat.write(nb,TARGET)
    print(json.dumps(dict(notebook=str(TARGET),cells=len(cells),code_cells=sum(c.cell_type=='code' for c in cells),embedded_figures=sum(bool(c.get('attachments')) for c in cells)),indent=2))


if __name__=='__main__': build()
