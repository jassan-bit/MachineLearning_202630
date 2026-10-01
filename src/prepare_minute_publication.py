"""Copy the verified minute delivery into the existing Git publication checkout."""
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    dest=(ROOT/'.publish-checkout').resolve()
    assert dest.parent==ROOT.resolve() and (dest/'.git').exists()
    manifest=json.loads((ROOT/'delivery/MINUTE_MANIFEST.json').read_text())
    names=set(manifest)|{'delivery/MINUTE_MANIFEST.json','delivery/Entregable1_minuto_2023_2025.zip',
                         'src/prepare_minute_publication.py','src/verify_minute_publication.py',
                         'book/myst.yml','book/entregable1_master.md'}
    names|={str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'book/sections').glob('*.md')}
    for name in sorted(names):
        target=(dest/name).resolve()
        assert target.is_relative_to(dest)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    myst=(ROOT/'book/myst.yml').read_text(encoding='utf-8').replace(
        'file: entregable1_master.md','file: book/entregable1_master.md').replace(
        'file: sections/','file: book/sections/').replace('style: _static/','style: book/_static/')
    (dest/'myst.yml').write_text(myst,encoding='utf-8')
    attributes=(dest/'.gitattributes').read_text(encoding='utf-8')
    for rule in ['data/processed/minute_2023_2025/** -text','book/** -text','*.py -text','*.md -text','*.json -text']:
        if rule not in attributes: attributes+='\n'+rule+'\n'
    (dest/'.gitattributes').write_text(attributes,encoding='utf-8')
    names|={'myst.yml','.gitattributes'}
    (ROOT/'outputs/minute_git_paths.txt').write_text('\n'.join(sorted(names))+'\n',encoding='utf-8')
    print(f'Prepared {len(names)} paths')


if __name__=='__main__': main()
