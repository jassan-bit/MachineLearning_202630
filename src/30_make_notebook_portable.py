"""Incorpora entradas al IPYNB y convierte figuras en salidas estándar de Jupyter."""
from pathlib import Path
import base64, hashlib, io, json, re, runpy, zipfile
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def make_portable(path):
    nb=nbformat.read(path,as_version=4)
    if nb.metadata.get('portable_inputs'): return
    comparison=runpy.run_path(str(ROOT/'src/28_build_compiled_notebook.py'))['COMPARISONS']
    names=['data/splits/development_80.csv','requirements-lock-windows-py310.txt']
    names += [f'outputs/tables/{name}' for name in comparison]
    names += ['outputs/tables/'+name for name in ['base_model_protocol.json','multivariate_protocol.json',
              'change_events_protocol.json','base_report_template.md','entregable1_checklist.csv']]
    for folder,patterns in [('src',['*.py','*.cjs']),('tests',['*.py']),('book/sections',['*.md'])]:
        for pattern in patterns:
            names += [p.relative_to(ROOT).as_posix() for p in (ROOT/folder).glob(pattern)]
    names=sorted(set(names))
    assert not any('test_20.csv' in p or 'master_1h.csv' in p for p in names)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in names: archive.write(ROOT/name,name)
    payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
    encoded=base64.b64encode(payload).decode('ascii')
    bootstrap='''# Preparación autónoma: las entradas están incorporadas en este archivo.
from pathlib import Path
import base64, hashlib, io, tempfile, zipfile
'''+f'INPUTS_BASE64 = {encoded!r}\nINPUTS_SHA256 = {digest!r}\n'+'''
_payload=base64.b64decode(INPUTS_BASE64)
assert hashlib.sha256(_payload).hexdigest()==INPUTS_SHA256
PORTABLE_ROOT=Path(tempfile.mkdtemp(prefix='entregable1_'))
PROJECT_ROOT=PORTABLE_ROOT/'inputs'
PROJECT_ROOT.mkdir()
with zipfile.ZipFile(io.BytesIO(_payload)) as _archive:
    for _name in _archive.namelist():
        assert (PROJECT_ROOT/_name).resolve().is_relative_to(PROJECT_ROOT.resolve())
    _archive.extractall(PROJECT_ROOT)
del _payload, INPUTS_BASE64
print('Entradas recuperadas del propio notebook. No se requiere la carpeta del proyecto.')
'''
    rebuilt=[]
    for cell in nb.cells:
        if cell.get('attachments'):
            for name,bundle in cell.attachments.items():
                data=dict(bundle)
                if 'image/svg+xml' in data:
                    svg=data['image/svg+xml'];data['image/svg+xml']=''.join(svg) if isinstance(svg,list) else svg
                code='from IPython.display import display\ndisplay('+repr(data)+', raw=True)'
                figure=nbformat.v4.new_code_cell(code)
                figure.metadata.update(embedded_figure=name,jupyter={'source_hidden':True})
                figure.outputs=[nbformat.v4.new_output('display_data',data=data,metadata={})]
                rebuilt.append(figure)
            caption=re.sub(r'!\[[^\]]*\]\(attachment:[^)]+\)','',cell.source).strip()
            if caption: rebuilt.append(nbformat.v4.new_markdown_cell(caption))
            continue
        if cell.cell_type=='code' and 'PROJECT_ROOT = next(' in cell.source:
            initial=nbformat.v4.new_code_cell(bootstrap)
            initial.metadata.update(portable_bootstrap=True,jupyter={'source_hidden':True})
            rebuilt.append(initial)
            a=cell.source.index('PROJECT_ROOT = next('); b=cell.source.index('NODE = shutil.which',a)
            cell.source=cell.source[:a]+cell.source[b:]
            cell.source=cell.source.replace("assert NODE is not None, 'Se necesita Node para las dos etapas JavaScript.'", "assert NODE is not None, 'Se necesita Node.js para las dos comprobaciones JavaScript; los datos ya están dentro del notebook.'")
        if cell.cell_type=='code' and "record_stage('" in cell.source:
            plots={
                '08_eda_target_development.py':['eda_target_distribution.png'],
                '09_univariate_development.py':['univariate_close.png'],
                '18_base_model.py':['base_BTCUSDT.png'],
                '12_multivariate_close.py':['multivariate_BTCUSDT.png'],
                '14_temporal_target.py':['temporal_target_BTCUSDT.png'],
                '19_extended_diagnostics.py':['residuals_all_BTCUSDT.png'],
                '22_change_events.py':['change_events_BTCUSDT.png'],
            }
            for stage,figures in plots.items():
                if f'record_stage({stage!r},started)' in cell.source:
                    cell.source+='\n# Figura generada por esta etapa.\n'
                    for figure in figures:
                        cell.source+=f"display(Image(filename=str(RUN_ROOT/'book/_static/figures/{figure}')))\n"
        rebuilt.append(cell)
    rebuilt[0].source='''# Entregable 1 — Pronóstico de volatilidad en criptomonedas

**Estudiantes:** Jassan Arteta - Mateo Bernal  
**Profesor:** Lihki Rubio  
**Asignatura:** Machine Learning  
**Universidad del Norte — 2026**

## Lectura y ejecución

Este único archivo incorpora el informe, código, datos DEVELOPMENT, protocolos, módulos auxiliares, tablas de referencia y gráficas. Puede abrirse desde Descargas u otra carpeta. No necesita descargar la carpeta del proyecto, claves de Binance ni consultar TEST.

Las figuras se guardan como salidas estándar de Jupyter. Si el editor oculta las salidas, desplegarlas. Para recalcular, seleccionar el kernel **Entregable 1 (Python 3.10)** en este computador y usar **Restart Kernel and Run All Cells**. En otro computador se necesita Python con las dependencias del proyecto y Node.js para dos comprobaciones; el archivo no instala programas automáticamente.

La sección 5 recupera las entradas incorporadas en una carpeta temporal y repite todos los análisis y los 170 ajustes. Las celdas de preparación y figuras tienen el código contraído para facilitar la lectura; el código de los cálculos está visible. La ejecución tarda varios minutos. TEST no está incluido y permanece reservado.
'''
    for cell in rebuilt:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('Las figuras están incorporadas al archivo y se pueden consultar sin ejecutar las celdas.', 'Las figuras son salidas guardadas en el propio archivo.')
            cell.source=cell.source.replace('Los módulos auxiliares y protocolos proceden de `src/` y `outputs/tables/` del repositorio.', 'Los módulos auxiliares y protocolos están incorporados al archivo y se recuperan en la preparación automática.')
    nb.cells=rebuilt
    nb.metadata.kernelspec=dict(name='entregable1',display_name='Entregable 1 (Python 3.10)',language='python')
    nb.metadata['portable_inputs']=dict(sha256=digest,files=len(names),bytes=len(payload),test_included=False)
    nb.metadata['project']['standalone_data']=True
    nbformat.validate(nb);nbformat.write(nb,path)
    print(json.dumps(dict(cells=len(nb.cells),code_cells=sum(c.cell_type=='code' for c in nb.cells),payload_MB=len(payload)/1e6,notebook_MB=path.stat().st_size/1e6),indent=2))


if __name__=='__main__': make_portable(ROOT/'notebooks/Entregable_1_Completo.ipynb')
