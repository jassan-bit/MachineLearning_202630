"""Ejecuta y guarda todas las celdas del notebook completo con el intérprete actual."""
from pathlib import Path
import asyncio
import hashlib
import json
import os
import sys
import time
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]


def main():
    if sys.platform=='win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.environ['IPYTHONDIR']=str(ROOT/'outputs/.ipython')
    path=ROOT/'notebooks/Entregable_1_Completo.ipynb'
    nb=nbformat.read(path,as_version=4)
    execution_directory=Path(tempfile.mkdtemp(prefix='notebook_validation_')) if nb.metadata.get('portable_inputs') else ROOT
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    def begin(cell,cell_index,**kwargs):
        if cell.cell_type=='code': print('Ejecutando celda',cell_index,cell.source.splitlines()[0],flush=True)
    def finished(cell,cell_index,**kwargs):
        if cell.cell_type=='code':
            if not cell.metadata.get('embedded_figure'):
                nbformat.write(nb,path)
            print('Completada celda',cell_index,flush=True)
    started=time.perf_counter()
    NotebookClient(nb,km=km,timeout=1800,allow_errors=False,on_cell_start=begin,on_cell_executed=finished,
                   extra_arguments=['--HistoryManager.enabled=False'],
                   resources={'metadata':{'path':str(execution_directory)}}).execute(cleanup_kc=True)
    codes=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in codes)
    assert not any(o.output_type=='error' for c in codes for o in c.outputs)
    nbformat.validate(nb); nbformat.write(nb,path)
    result=dict(notebook=path.relative_to(ROOT).as_posix(),status='passed',executed_code_cells=len(codes),
                embedded_figures=sum(bool(c.get('attachments')) or bool(c.metadata.get('embedded_figure')) for c in nb.cells),
                independent_working_directory=str(execution_directory),standalone_inputs=bool(nb.metadata.get('portable_inputs')),
                seconds=time.perf_counter()-started,python=sys.version,interpreter=sys.executable,
                notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),test_read=False)
    (ROOT/'outputs/tables/compiled_notebook_execution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
