"""Reproduce every saved forecast and audit selection/calendar integrity."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from improve_classical_forecast import OUT, SOURCE, DATA, ROOT, load_panel, minute_features, predict_bundle


def verify():
    status = json.loads((OUT / 'status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete':
        raise ValueError('Experiment is incomplete')
    assert status['selection_sha256'] == hashlib.sha256((OUT / 'selected_inputs.csv').read_bytes()).hexdigest()
    config = json.loads((OUT / 'configuration.json').read_text(encoding='utf-8'))
    assert config['source_choices_sha256'] == hashlib.sha256((SOURCE / 'selected_inputs.csv').read_bytes()).hexdigest()
    manifest = json.loads((ROOT / 'results/minute_2023_2025/data_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        digest = hashlib.sha256()
        with (DATA / name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        assert digest.hexdigest() == expected, f'Dataset changed: {name}'
    choices = pd.read_csv(OUT / 'selected_inputs.csv')
    search = pd.read_csv(OUT / 'search.csv')
    calendar = pd.read_csv(OUT / 'calendar.csv')
    for _, fold in calendar.groupby('fold'):
        last_label = pd.to_datetime(fold.loc[fold.split == 'train', 'target_end'], utc=True).max()
        first_validation = pd.to_datetime(fold.loc[fold.split == 'validation', 'origin'], utc=True).min()
        assert last_label < first_validation
        assert pd.to_datetime(fold.target_end, utc=True).max() < pd.Timestamp('2025-01-01', tz='UTC')
    predictions = pd.read_csv(OUT / 'predictions.csv.gz')
    keys = ['symbol', 'volatility_window', 'origin', 'horizon']
    assert not predictions.duplicated(keys).any()
    reference = pd.read_csv(SOURCE / 'predictions.csv.gz')
    joined = predictions.merge(reference, on=keys, suffixes=('', '_reference'), validate='one_to_one')
    assert len(joined) == len(reference) == len(predictions)
    np.testing.assert_allclose(joined.actual, joined.actual_reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(joined.xgboost, joined.forecast_reference, rtol=1e-6, atol=1e-7)
    panel = load_panel()
    maximum_error = 0.
    for symbol, group in predictions.groupby('symbol', sort=False):
        minute = minute_features(np.load(DATA / f'{symbol}.npy'))
        for window, frame in group.groupby('volatility_window'):
            saved = frame.pivot(index='origin', columns='horizon', values='forecast').sort_index()
            origins = panel.index.get_indexer(pd.to_datetime(saved.index, utc=True))
            assert (origins >= 0).all()
            bundle = joblib.load(OUT / 'models' / f'{symbol}_v{window}.joblib')
            assert pd.Timestamp(bundle['fitted_through']) < pd.Timestamp('2025-01-01', tz='UTC')
            actual = predict_bundle(bundle, panel[symbol], minute, origins)
            assert np.isfinite(actual).all() and (actual >= 0).all()
            np.testing.assert_allclose(actual, saved.to_numpy(), rtol=1e-12, atol=1e-12)
            maximum_error = max(maximum_error, float(np.max(np.abs(actual-saved.to_numpy()))))
            selected = choices.query('symbol == @symbol and volatility_window == @window').iloc[0]
            options = search.query('symbol == @symbol and volatility_window == @window')
            assert np.isclose(selected.validation_rmse, options.validation_rmse.min())
            assert selected.validation_rmse <= selected.xgboost_validation_rmse + 1e-12
    report = dict(status='passed', checked_predictions=len(predictions), models=len(choices),
                  serialization_max_absolute_error=maximum_error, temporal_folds=calendar.fold.nunique(),
                  xgboost_reference_reproduced=True, common_targets_verified=True)
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify()
