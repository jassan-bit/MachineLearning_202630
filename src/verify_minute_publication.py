"""Verify the deployed study and the exact published download bytes."""
import argparse
import hashlib
import json
from urllib.parse import urljoin
import requests
from verify_daily_publication import Links,ROOT,BASE


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--commit',required=True); args=parser.parse_args()
    response=requests.get('https://api.github.com/repos/jassan-bit/MachineLearning_202630/actions/runs',
        params={'head_sha':args.commit,'per_page':20},timeout=45)
    response.raise_for_status(); runs=response.json()['workflow_runs']
    outcomes=[{k:r[k] for k in ['name','status','conclusion','html_url']} for r in runs]
    if len(outcomes)<2 or any(r['conclusion']!='success' for r in outcomes):
        print(json.dumps(dict(status='pending',runs=outcomes),indent=2)); return
    home=requests.get(BASE,params={'v':args.commit},timeout=45); home.raise_for_status()
    parsed=Links(home.content.decode('utf-8'))
    assert '2023' in ' '.join(parsed.text) and '2025' in ' '.join(parsed.text)
    reports=[urljoin(BASE,l) for l in parsed.links if 'estudio-minuto' in l]
    assert reports,'Missing minute report'
    report=requests.get(reports[0],timeout=45); report.raise_for_status()
    assert '40.320' in report.content.decode('utf-8')
    downloads={}
    for name in ['notebooks/Entregable_1_Minuto.ipynb','delivery/Entregable1_minuto_2023_2025.zip']:
        stem=(ROOT/name).stem.lower()
        matches=[urljoin(BASE,l) for l in parsed.links if stem in l.lower()]
        assert matches, name
        data=requests.get(matches[0],timeout=45); data.raise_for_status()
        digest=hashlib.sha256(data.content).hexdigest()
        assert digest==hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name
        downloads[name]=dict(url=matches[0],sha256=digest)
    result=dict(status='verified',commit=args.commit,runs=outcomes,downloads=downloads)
    (ROOT/'outputs/minute_publication.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
