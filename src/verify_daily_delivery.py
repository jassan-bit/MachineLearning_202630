"""Independent artifact checks on actual outputs, API, notebook and splits."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
import joblib
from volatility_experiment import ROOT, configuration, load_daily, targets, arrays


def main():
    out = ROOT/'results'; cfg = configuration()
    status = json.loads((out/'run_status.json').read_text())
    assert status['status'] == 'complete'
    assert status['protocol']['configuration'] == cfg
    assert status['protocol']['daily_sha256'] == hashlib.sha256((ROOT/'data/processed/daily_2020_2025.csv').read_bytes()).hexdigest()
    metrics = pd.read_csv(out/'metrics.csv'); preds = pd.read_csv(out/'predictions.csv.gz')
    folds = pd.read_csv(out/'fold_calendar.csv')
    registry = pd.read_csv(out/'model_registry.csv')
    assert len(registry) == 16 and len(pd.read_csv(out/'selection.csv')) == 64
    assert set(preds.symbol) == set(cfg['symbols'])
    assert np.isfinite(preds[['actual','svr','persistence']].to_numpy()).all()
    assert not preds.duplicated(['symbol','volatility_window','input_window','fold','split','origin','horizon']).any()
    for fold, g in folds.groupby('fold'):
        tr = g[g.split == 'train']; va = g[g.split == 'val']; te = g[g.split == 'test']
        assert pd.to_datetime(tr.target_end, utc=True).max() < pd.to_datetime(va.anchor, utc=True).min()
        assert pd.to_datetime(va.target_end, utc=True).max() < pd.to_datetime(te.anchor, utc=True).min()
    groups = ['symbol','volatility_window','input_window','fold','split']
    for key, g in preds.groupby(groups):
        actual = g.pivot(index='origin',columns='horizon',values='actual').to_numpy()
        prediction = g.pivot(index='origin',columns='horizon',values='svr').to_numpy()
        rmse = np.sqrt(((actual-prediction)**2).mean(axis=0)).mean()
        selected = metrics
        for k,v in zip(groups,key): selected = selected[selected[k] == v]
        row = selected[(selected.model == 'SVR') & (selected.horizon == 0)].iloc[0]
        np.testing.assert_allclose(row.rmse, rmse, rtol=1e-10)
    panel = load_daily()
    # Artifact predictions match stored evaluation on actual folds.
    for symbol in cfg['symbols']:
        key = (symbol, 28, 28, 1)
        g = preds[(preds.symbol == symbol) & (preds.volatility_window == 28) &
            (preds.input_window == 28) & (preds.fold == 1) & (preds.split == 'test')]
        origin = sorted(g.origin.unique())[0]
        anchor = panel.index.get_loc(pd.Timestamp(origin))
        X, y, _ = arrays(panel[symbol], targets(panel[symbol],28), np.array([anchor]),28)
        model = joblib.load(out/f'models/{symbol}_v28_l28_f1.joblib')['model']
        np.testing.assert_allclose(model.predict(X)[0], g[g.origin == origin].sort_values('horizon').svr)
        np.testing.assert_allclose(y[0], g[g.origin == origin].sort_values('horizon').actual)
    import sys
    sys.path.insert(0,str(ROOT))
    from fastapi.testclient import TestClient
    from app.api import app
    client = TestClient(app)
    for row in registry.to_dict('records'):
        lags = panel[row['symbol']].iloc[-int(row['input_window']):].tolist()
        response = client.post('/predict',json=dict(symbol=row['symbol'],volatility_window=int(row['volatility_window']),lags=lags))
        assert response.status_code == 200, response.text
        assert len(response.json()['prediction']) == 7
    nbformat.validate(nbformat.read(ROOT/'notebooks/Entregable_1_Completo.ipynb', as_version=4))
    result = dict(status='passed', configurations=64, fold_models=320, deployed_models=16,
        metric_rows=len(metrics), prediction_rows=len(preds), fold_sizes=folds.groupby(['fold','split']).size().to_dict())
    result['fold_sizes'] = {str(k):int(v) for k,v in result['fold_sizes'].items()}
    (out/'verification.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
