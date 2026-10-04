"""Check audit evidence, unchanged originals, local links and optional live release."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
import argparse
import hashlib
import json
import re
import zipfile

import numpy as np
import pandas as pd
import requests
from evaluate_frozen_holdout import verify_protocol
from volatility_experiment import targets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/current_delivery_audit'
BASE = 'https://jassan-bit.github.io/MachineLearning_202630/'
PAGES = {'': ROOT / 'book/entregable1_master.md',
         'base-datos/': ROOT / 'book/sections/11_base_datos.md',
         'eda/': ROOT / 'book/sections/12_eda.md',
         'modelo-base-svr/': ROOT / 'book/sections/13_modelo_base_svr.md',
         'evaluacion/': ROOT / 'book/sections/14_evaluacion.md',
         'reproducibilidad/': ROOT / 'book/sections/15_reproducibilidad.md'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.headings, self.links, self.text = [], [], []
        self.heading = None
        self.hidden = 0
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag in ['script', 'style']:
            self.hidden += 1
        if tag in ['h1', 'h2', 'h3', 'h4']:
            self.heading = []
        if tag in ['a', 'img']:
            field = 'href' if tag == 'a' else 'src'
            self.links.extend(value for key, value in attrs if key == field)

    def handle_endtag(self, tag):
        if tag in ['script', 'style']:
            self.hidden -= 1
        if tag in ['h1', 'h2', 'h3', 'h4'] and self.heading is not None:
            self.headings.append(''.join(self.heading).rstrip('¶').strip())
            self.heading = None

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)
            if self.heading is not None:
                self.heading.append(data)


def local():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    for relative, digest in manifest.items():
        assert sha(ROOT / relative) == digest, ('Audit artifact changed', relative)
    _, protocol_digest = verify_protocol()
    for relative, digest in json.loads((OUT / 'model/provenance.json').read_text(encoding='utf-8'))['original_files_sha256'].items():
        assert sha(ROOT / relative.replace('\\', '/')) == digest, ('Original changed', relative)
    for filename, digest in json.loads((OUT / 'quality/artifact_hashes.json').read_text(encoding='utf-8')).items():
        assert sha(OUT / 'quality' / filename) == digest, ('Quality evidence changed', filename)
    for family, metadata, key, script in [
        ('quality', 'methodology.json', 'script_sha256', 'audit_current_dataset.py'),
        ('eda', 'metadata.json', 'source_sha256', 'audit_current_eda.py'),
        ('model', 'provenance.json', 'source_sha256', 'audit_current_model.py')]:
        information = json.loads((OUT / family / metadata).read_text(encoding='utf-8'))
        assert information[key] == sha(ROOT / 'src' / script), ('Script/provenance differ', family)
    broken = []
    for source in PAGES.values():
        markdown = source.read_text(encoding='utf-8')
        assert '**Pendiente:**' not in markdown and '**Pendiente para' not in markdown
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', markdown):
            target = link.split('#')[0]
            if not target or re.match(r'^[a-z]+:', target):
                continue
            if not (source.parent / target).exists():
                broken.append((str(source.relative_to(ROOT)), target))
    assert not broken, ('Broken local links', broken)
    original = pd.read_csv(ROOT / 'data/processed/minute_2023_2025/daily_target_closes.csv', index_col=0)
    addition = pd.read_csv(OUT / 'holdout/daily_target_closes.csv', index_col=0)
    panel = pd.concat([original, addition])
    panel.index = pd.to_datetime(panel.index, utc=True)
    forecasts = pd.read_csv(OUT / 'holdout/predictions.csv.gz')
    origins = pd.to_datetime(forecasts.origin, utc=True)
    future = origins + pd.to_timedelta(forecasts.horizon, unit='D')
    assert (future == pd.to_datetime(forecasts.target_date, utc=True)).all()
    assert origins.min() == pd.Timestamp('2026-01-01', tz='UTC')
    assert origins.max() == pd.Timestamp('2026-08-24', tz='UTC')
    assert future.max() < pd.Timestamp('2026-09-01', tz='UTC')
    assert len(forecasts) == 236 * 16 * 7
    assert not forecasts.duplicated(['symbol', 'volatility_window', 'origin', 'horizon']).any()
    for (symbol, window), group in forecasts.groupby(['symbol', 'volatility_window']):
        volatility = targets(panel[symbol], int(window))
        origin = pd.to_datetime(group.origin, utc=True)
        label = pd.to_datetime(group.target_date, utc=True)
        np.testing.assert_allclose(group.actual, volatility.reindex(label), rtol=1e-12)
        np.testing.assert_allclose(group.persistence, volatility.reindex(origin), rtol=1e-12)
    with zipfile.ZipFile(ROOT / 'delivery/Entregable1_auditoria_actualizada.zip') as archive:
        assert archive.testzip() is None
        for relative, digest in manifest.items():
            assert hashlib.sha256(archive.read(relative)).hexdigest() == digest
    return dict(audit_files=len(manifest), pages_with_links_verified=len(PAGES),
                protocol_sha256=protocol_digest, original_files_unchanged=True,
                future_targets_verified=True, supplemental_package_verified=True)


def live(commit):
    session = requests.Session()
    session.headers.update({'Cache-Control': 'no-cache', 'User-Agent': 'current-delivery-verification'})
    response = session.get('https://api.github.com/repos/jassan-bit/MachineLearning_202630/actions/runs',
        params={'head_sha': commit, 'per_page': 10}, timeout=30)
    response.raise_for_status()
    deploy = [run for run in response.json()['workflow_runs']
              if run['head_sha'] == commit and run['path'] == '.github/workflows/deploy.yml']
    assert deploy and deploy[0]['conclusion'] == 'success', 'This commit has not deployed successfully'

    def get_page(item):
        route, source = item
        url = urljoin(BASE, route) + '?v=' + commit[:12]
        response = session.get(url, timeout=45)
        response.raise_for_status()
        page = Page(response.content.decode('utf-8'))
        expected = [re.sub(r'[*`]', '', title).strip() for title in
                    re.findall(r'^#{1,4} (.+)$', source.read_text(encoding='utf-8'), re.M)]
        assert all(title in page.headings for title in expected), ('Stale page headings', route)
        return dict(url=url, headings_verified=len(expected), status=response.status_code), page

    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(get_page, PAGES.items()))
    downloads = []
    references = {'Entregable1_auditor': ROOT / 'delivery/Entregable1_auditoria_actualizada.zip',
                  'current_holdout_met': ROOT / 'book/figures/current_holdout_metrics.png',
                  'metric_confidence_i': OUT / 'model/metric_confidence_intervals.csv'}
    for marker, path in references.items():
        matching = [urljoin(BASE, link) for _, page in checked for link in page.links if marker in link]
        assert matching, ('Download not exposed', marker)
        matched = False
        for url in dict.fromkeys(matching):
            response = session.get(url, timeout=60)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() == sha(path):
                downloads.append(dict(url=url, bytes=len(response.content), sha256=sha(path)))
                matched = True
                break
        assert matched, ('Published download differs', marker)
    return dict(commit=commit, workflow=deploy[0]['html_url'], pages=[row for row, _ in checked],
                verified_downloads=downloads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit')
    args = parser.parse_args()
    result = local()
    if args.commit:
        result['publication'] = live(args.commit)
    result['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    (ROOT / 'outputs/current_delivery_verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
