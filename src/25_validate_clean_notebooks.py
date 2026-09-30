"""Ejecuta los notebooks corregidos con el kernel del entorno nuevo y datos aislados."""
from pathlib import Path
import asyncio,hashlib,json,os,shutil,sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert sys.prefix!=sys.base_prefix
    manifest_path=ROOT/'outputs/tables/clean_reproduction_metadata.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['status']=='passed'
    destination=Path(manifest.get('workspace',ROOT/'outputs/clean_reproduction')).resolve()
    assert destination.is_relative_to((ROOT/'outputs').resolve())
    manifest['workspace']=str(destination)
    # Los cálculos ya se compararon; incorporar las correcciones editoriales finales.
    renderers=['09_render_univariate_report.py','13_render_feature_diagnostics.py',
               '14_render_target_temporal.py','18_render_base_report.py','23_render_change_events.py']
    manifest['report_source_hashes']={}
    for name in renderers:
        source=ROOT/'src'/name
        shutil.copy2(source,destination/'src'/name)
        manifest['report_source_hashes'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
    os.environ['IPYTHONDIR']=str(destination/'outputs/.ipython')
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    names=['09_univariate_development','13_feature_diagnostics','14_temporal_target',
           '17_preprocessing_close','18_base_model','22_change_events']
    results=[]
    for name in names:
        p=destination/'notebooks'/f'{name}.ipynb'
        shutil.copy2(ROOT/'notebooks'/f'{name}.ipynb',p)
        nb=nbformat.read(p,as_version=4)
        nb.cells.insert(0,nbformat.v4.new_code_cell(
            'from pathlib import Path\nimport sys\nassert Path(sys.prefix).resolve() == Path('+repr(sys.prefix)+').resolve()\nprint("Kernel del entorno limpio:", sys.executable)'))
        km=KernelManager(kernel_name='python3')
        km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        NotebookClient(nb,km=km,timeout=300,extra_arguments=['--HistoryManager.enabled=False'],
            resources={'metadata':{'path':str(destination)}}).execute(cleanup_kc=True)
        for cell in nb.cells:
            if cell.cell_type=='code':
                assert cell.execution_count is not None
                assert not any(o.output_type=='error' for o in cell.outputs)
        nbformat.write(nb,p)
        results.append(dict(notebook=name,code_cells=sum(c.cell_type=='code' for c in nb.cells),kernel=sys.executable))
        manifest['clean_notebooks']=results
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('Notebook limpio:',name,flush=True)


if __name__=='__main__':main()
