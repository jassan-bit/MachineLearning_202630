"""HTTP integration for the public, precomputed methodological review."""
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

module = importlib.import_module('volatility_dashboard.app')


class FinalReviewRoutesTests(unittest.TestCase):
    def setUp(self):
        self.client = module.app.server.test_client()

    def test_deep_link_selects_review_and_rejects_unknown_tabs(self):
        for search, expected in [('?tab=revision-final', 'revision-final'),
                                 ('?tab=bad', 'contexto'), ('', 'contexto')]:
            response = self.client.post('/_dash-update-component', json={
                'output': 'main-tabs.value',
                'outputs': {'id': 'main-tabs', 'property': 'value'},
                'inputs': [{'id': 'page-location', 'property': 'search', 'value': search}],
                'state': [], 'changedPropIds': ['page-location.search']})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['response']['main-tabs']['value'], expected)

    def test_public_assets_are_bounded_and_release_identifies_runtime(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / 'files' / 'delivery'
            assets.mkdir(parents=True)
            (assets / 'report.html').write_text('<h1>Review</h1>', encoding='utf-8')
            (assets / 'metrics.csv').write_text('mse\n0.1\n', encoding='utf-8')
            (assets / 'slides.pdf').write_bytes(b'%PDF-1.4\n')
            (assets / 'unlisted.txt').write_text('not public', encoding='utf-8')
            (root / 'bundle_manifest.json').write_text(json.dumps({'files': {
                'delivery/' + name: {} for name in ['report.html', 'metrics.csv', 'slides.pdf']}}), encoding='utf-8')
            (root / 'release.json').write_text(json.dumps({'release_id': 'test-review'}), encoding='utf-8')
            with patch.object(module, 'FINAL_REVIEW_BUNDLE', root), \
                    patch.dict(module.os.environ, {'RENDER_GIT_COMMIT': 'runtime-commit'}):
                for name, mime in [('report.html', 'text/html'), ('metrics.csv', 'text/csv'),
                                   ('slides.pdf', 'application/pdf')]:
                    with self.client.get('/final-review/delivery/' + name) as response:
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.mimetype, mime)
                for route in ['/final-review/../release.json', '/final-review/%2e%2e/release.json',
                              '/final-review/release.json', '/final-review/.env',
                              '/final-review/delivery/unlisted.txt',
                              '/final-review/delivery/missing.html']:
                    self.assertEqual(self.client.get(route).status_code, 404, route)
                response = self.client.get('/review-release.json')
                self.assertEqual(response.get_json(), {
                    'release_id': 'test-review', 'deployed_commit': 'runtime-commit'})

    def test_missing_bundle_returns_404_and_entry_redirect_is_relative(self):
        with TemporaryDirectory() as temp, patch.object(module, 'FINAL_REVIEW_BUNDLE', Path(temp)):
            self.assertEqual(self.client.get('/review-release.json').status_code, 404)
            self.assertEqual(self.client.get('/final-review/missing.html').status_code, 404)
        response = self.client.get('/final-review/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'],
                         '/final-review/delivery/final_regression_local_20261018/index.html')


if __name__ == '__main__':
    unittest.main()
