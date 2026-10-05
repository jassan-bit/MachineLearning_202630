"""The web path serves the exact offline-audited original forecasts."""
import unittest
from unittest.mock import patch

from volatility_dashboard import data_loader as loader
from volatility_dashboard import runtime_audit


class RuntimeAuditTests(unittest.TestCase):
    def tearDown(self):
        loader.repository.cache_clear()

    def test_web_load_checks_original_forecasts_without_replaying_models(self):
        loader.repository.cache_clear()
        with patch.object(loader, 'predict', side_effect=AssertionError('Web must not replay models')), \
                patch.object(loader, 'predict_bundle', side_effect=AssertionError('Web must not replay HAR')), \
                patch.object(loader, 'artifact_for', side_effect=AssertionError('Web must not load all artifacts')):
            predictions, _, audit = loader.repository()
        self.assertEqual(set(predictions.model), set(loader.FAMILIES))
        self.assertEqual(len(predictions), 7 * 40096)
        self.assertEqual(int(audit.estado_comparable.eq('Comparable').sum()), 7)

    def test_changed_fitted_artifact_invalidates_the_offline_audit(self):
        original_digest = runtime_audit.file_digest

        def changed(path):
            if str(path).endswith('BNBUSDT_v7.joblib'):
                return '0' * 64
            return original_digest(path)

        with patch.object(runtime_audit, 'file_digest', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'Archivo distinto del auditado'):
                runtime_audit.read_audit(loader.ROOT, loader.RUNTIME_AUDIT, loader.FAMILIES)


if __name__ == '__main__':
    unittest.main()
