"""Empaqueta los archivos reproducibles con una lista explícita; nunca abre TEST."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def files():
    selected = []
    for directory in ['src', 'tests', 'notebooks', 'outputs/tables', 'outputs/models', 'book/sections', 'book/_static']:
        selected.extend(p for p in (ROOT / directory).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts)
    selected.extend(ROOT / name for name in ['README.md', 'requirements.txt',
        'requirements-lock-windows-py310.txt', 'book/myst.yml', 'book/entregable1_master.md',
        'data/splits/development_80.csv'])
    # Registros de la reproducción ejecutada, sin copiar sus datos o su entorno.
    selected.extend((ROOT / 'outputs/clean_reproduction/logs').glob('*.txt'))
    selected.extend(p for p in (ROOT / 'book/_build/html').rglob('*') if p.is_file())
    return sorted(set(selected))


def main():
    selected = files()
    assert all(p.is_file() for p in selected)
    assert all('test_20.csv' not in str(p) and 'master_1h.csv' not in str(p) for p in selected)
    manifest = {'test_included': False, 'files': {p.relative_to(ROOT).as_posix():
        {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in selected}}
    destination = ROOT / 'delivery'
    destination.mkdir(exist_ok=True)
    path = destination / 'Entregable1_corregido.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in selected:
            archive.write(p, p.relative_to(ROOT).as_posix())
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    (destination / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'archive': str(path), 'files': len(selected), 'bytes': path.stat().st_size,
                      'test_included': False}, indent=2))


if __name__ == '__main__':
    main()
