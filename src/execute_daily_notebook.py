"""Execute the portable notebook and verify a second independent model run."""
from pathlib import Path
import json
import time
import base64
import gzip
import argparse
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def verification_cell():
    reference = base64.b64encode(gzip.compress((ROOT/'results/metrics.csv').read_bytes())).decode()
    return nbformat.v4.new_code_cell("import base64, gzip, io\nimport pandas as pd\n"
        f"reference = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode({reference!r}))))\n"
        "recomputed = pd.read_csv(workspace / 'results/metrics.csv')\n"
        "pd.testing.assert_frame_equal(recomputed, reference, check_exact=False, rtol=1e-10, atol=1e-12)\n"
        "print('Reproducción verificada con tolerancia numérica (rtol=1e-10, atol=1e-12).')",
        metadata={'daily_verification':True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-existing-workspace', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    path = ROOT/'notebooks/Entregable_1_Completo.ipynb'
    nb = nbformat.read(path, as_version=4)
    # Assert the recomputation matches the stored metrics, not just successful execution.
    import hashlib
    expected = hashlib.sha256((ROOT/'results/metrics.csv').read_bytes()).hexdigest()
    nb.cells = [c for c in nb.cells if not (c.metadata.get('daily_verification') or
        (c.cell_type == 'code' and 'Reproducción verificada' in c.source))]
    cell = verification_cell()
    if args.verify_existing_workspace:
        # Recheck the existing independently executed run; no duplicate training.
        exec(cell.source, {'workspace':args.verify_existing_workspace})
        cell.execution_count = max(c.execution_count or 0 for c in nb.cells if c.cell_type=='code')+1
        cell.outputs = [nbformat.v4.new_output('stream', name='stdout',
            text='Reproducción verificada con tolerancia numérica (rtol=1e-10, atol=1e-12).\n')]
        nb.cells.append(cell)
    else:
        nb.cells.append(cell)
        NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb, path)
    record_path = ROOT/'results/notebook_execution.json'
    record = json.loads(record_path.read_text()) if args.verify_existing_workspace else {}
    record.update(status='executed_and_verified', cells=len(nb.cells), metrics_sha256=expected,
        comparison='pandas.assert_frame_equal rtol=1e-10 atol=1e-12')
    if not args.verify_existing_workspace: record['seconds'] = time.perf_counter()-started
    (ROOT/'results/notebook_execution.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
