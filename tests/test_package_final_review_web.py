"""Publication checks use tiny dependency graphs and never train or publish."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from package_final_review_web import DESTINATION, ENTRYPOINT, STUDY, package_review


class PackageFinalReviewWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))
        return path

    def build(self, html=None, extra=()):
        if html is not None:
            self.write(ENTRYPOINT, html)
        return package_review(self.root, entrypoints=(ENTRYPOINT,), additional_inputs=extra)

    def test_realistic_relative_graph_preserves_sources_and_separates_public_hashes(self):
        original = ('<!doctype html><html><head><title>Revisión</title></head><body>'
                    '<p>GitHub, Render y el modelo vigente no se actualizan.</p>'
                    '<pre>GitHub y Render sin cambios.</pre>'
                    '<script>var history="GitHub y Render sin cambios.";</script>'
                    '<a href="../../results/final_regression_20261018/report.html#metrics">Informe</a>'
                    '</body></html>')
        self.write(ENTRYPOINT, original)
        report = '<html><h1 id="metrics">Métricas</h1><img src="plots/view.png"><a href="summary.json">Datos</a></html>'
        self.write(STUDY + "report.html", report)
        image = self.write(STUDY + "plots/view.png", b"\x89PNG\r\nfixture")
        self.write(STUDY + "summary.json", '{"publication":"LOCAL ONLY","mse":0.42}')
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = self.build()
        self.assertEqual(result["file_count"], 4)
        self.assertEqual(result["html_pages"], 2)
        self.assertEqual(result["broken_local_links"], [])
        self.assertFalse(result["publication_context"]["model_replaced"])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
            relative = path.relative_to(self.root).as_posix()
            copy = self.root / DESTINATION / "files" / relative
            item = result["files"][relative]
            self.assertEqual(item["source_sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(item["public_sha256"], hashlib.sha256(copy.read_bytes()).hexdigest())
        public_html = (self.root / DESTINATION / "files" / ENTRYPOINT).read_text(encoding="utf-8")
        self.assertIn('id="public-review-context"', public_html)
        self.assertIn("18/10/2026", public_html)
        self.assertIn("original continúa como modelo vigente", public_html)
        self.assertIn('<pre>GitHub y Render sin cambios.</pre>', public_html)
        self.assertIn('var history="GitHub y Render sin cambios.";', public_html)
        self.assertEqual((self.root / DESTINATION / "files" / image.relative_to(self.root)).read_bytes(), image.read_bytes())
        self.assertEqual(result["files"][STUDY + "summary.json"]["source_sha256"],
                         result["files"][STUDY + "summary.json"]["public_sha256"])
        saved = json.loads((self.root / DESTINATION / "bundle_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["public_hash"], result["public_hash"])

    def test_missing_file_or_fragment_fails_before_creating_output(self):
        for link in ("missing.csv", "report.html#does-not-exist"):
            with self.subTest(link=link):
                self.write(STUDY + "report.html", '<h1 id="real">Métricas</h1>')
                html = '<a href="../../results/final_regression_20261018/' + link + '">Datos</a>'
                with self.assertRaises(ValueError):
                    self.build(html)
                self.assertFalse((self.root / DESTINATION).exists())

    def test_private_paths_model_formats_and_extra_python_are_rejected(self):
        candidates = (STUDY + "models/model.joblib", STUDY + "weights.npz",
                      STUDY + "secrets.json", STUDY + "script.py", ".git/config",
                      "src/unreviewed.py")
        for name in candidates:
            with self.subTest(name=name):
                self.write(name, "not for publication")
                with self.assertRaises(ValueError):
                    self.build('<h1>Informe</h1>', extra=(name,))
                self.assertFalse((self.root / DESTINATION).exists())

    def test_link_escape_and_sensitive_values_are_rejected_without_echoing_secret(self):
        values = ('<a href="/outside-prefix">Escape</a>',
                  '<a href="../../../../outside.html">Escape</a>',
                  '<a href="https://username:private-value@example.com/">Private</a>',
                  '<a href="https://example.com/?token=private-value">Private</a>',
                  '<p>ghp_' + 'A' * 36 + '</p>')
        for value in values:
            with self.subTest(value=value[:28]):
                with self.assertRaises(ValueError) as error:
                    self.build(value)
                self.assertNotIn("private-value", str(error.exception))
                self.assertNotIn("A" * 36, str(error.exception))
                self.assertFalse((self.root / DESTINATION).exists())

    def test_rebuild_uses_sources_once_and_refuses_stale_public_files(self):
        first = self.build('<h1>Informe</h1><p>Los resultados siguen locales.</p>')
        second = self.build()
        self.assertEqual(first["public_hash"], second["public_hash"])
        copied = self.root / DESTINATION / "files" / ENTRYPOINT
        self.assertEqual(copied.read_text(encoding="utf-8").count('id="public-review-context"'), 1)
        stale = self.root / DESTINATION / "files" / "unmanifested.txt"
        stale.write_text("retain for review", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.build()
        self.assertEqual(stale.read_text(encoding="utf-8"), "retain for review")


if __name__ == "__main__":
    unittest.main()
