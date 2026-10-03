"""Export DM comparisons from the dashboard's audited forecasts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from volatility_dashboard.data_loader import repository, ROOT
from volatility_dashboard.diebold_mariano import comparisons


def main():
    predictions, _, audit = repository()
    out = ROOT / 'results' / 'diebold_mariano_2025'
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / 'audit.csv', index=False)
    models = predictions.model.unique().tolist()
    rows = []
    for window in sorted(predictions.volatility_window.unique()):
        for symbol in ['TODOS'] + sorted(predictions.symbol.unique()):
            for horizon in ['TODOS'] + list(range(1, 8)):
                result = comparisons(predictions, symbol, window, horizon, models)
                result.insert(0, 'horizon', horizon)
                result.insert(0, 'volatility_window', window)
                result.insert(0, 'symbol', symbol)
                rows.append(result)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out / 'comparisons.csv', index=False)
    summary = combined.query('symbol == "TODOS" and volatility_window == 7 and horizon == "TODOS"')
    print(summary.to_string(index=False))
    print(f'\nResultados: {out / "comparisons.csv"}')


if __name__ == '__main__':
    main()
