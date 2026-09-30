"""Verifica el despliegue y descargas públicas contra los archivos locales."""
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin
import argparse
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://jassan-bit.github.io/MachineLearning_202630/'


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.links = []
        self.words = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k, v in attrs if k == 'href')

    def handle_data(self, data):
        self.words.append(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    session = requests.Session()
    session.headers.update({'User-Agent': 'Entregable1-publication-audit'})
    response = session.get('https://api.github.com/repos/jassan-bit/MachineLearning_202630/actions/runs',
                           params={'head_sha': args.commit, 'per_page': 10}, timeout=60)
    response.raise_for_status()
    runs = response.json()['workflow_runs']
    assert any(r['head_sha'] == args.commit and r['conclusion'] == 'success' for r in runs), 'Deployment not yet successful'
    pages = {}
    for route in ['', 'base-datos/', 'eda/', 'modelo-base/', 'conclusiones/']:
        response = session.get(urljoin(BASE, route), timeout=60)
        response.raise_for_status()
        pages[route] = Page(response.content.decode('utf-8'))
    assert 'Organización del informe' in ' '.join(pages[''].words)
    assert not any('/auditoria' in link for link in pages[''].links)
    conclusions = ' '.join(pages['conclusiones/'].words)
    assert '0,623259' in conclusions and '0,389006' in conclusions
    assert '4.475' in ' '.join(pages['eda/'].words)
    downloads = []
    for filename, local in [
        ('development_80', 'data/splits/development_80.csv'),
        ('requirements-lock-wi', 'requirements-lock-windows-py310.txt'),
        ('18_base_model', 'notebooks/18_base_model.ipynb'),
    ]:
        links = [link for link in pages['conclusiones/'].links if filename in link]
        assert links, ('Missing download link', filename)
        url = urljoin(BASE, links[0])
        response = session.get(url, timeout=60)
        response.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()
        expected = (ROOT / local).read_bytes()
        mode = 'byte_for_byte'
        if local == 'requirements-lock-windows-py310.txt':
            assert response.content.replace(b'\r\n', b'\n') == expected.replace(b'\r\n', b'\n')
            mode = 'identical_text_allowing_CRLF_LF'
        else:
            assert digest == hashlib.sha256(expected).hexdigest(), ('Download mismatch', local)
        downloads.append({'url': url, 'file': local, 'bytes': len(response.content), 'sha256': digest, 'comparison': mode})
    result = dict(url=BASE, http_status=200, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        method='GET of five pages, content markers, download SHA-256, successful Actions deployment',
        commit_verified=args.commit, corrected_version_published=True, content_version_verified=True,
        pages_verified=[urljoin(BASE, p) for p in pages], downloads_verified=downloads)
    if args.record:
        (ROOT / 'outputs/tables/publication_access_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
