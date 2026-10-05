"""Bounded causal HAR research using only data ending in 2024.

This script writes an isolated research directory and never evaluates 2025.
Run: .venv-repro/Scripts/python.exe src/research_har_candidates.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from improve_classical_forecast import har_features
from optimize_minute_svr import ROOT, DATA, LAGS, calendars, features, load_panel, minute_features, predict
from optimize_minute_xgboost import estimator

OUT = ROOT / "results/har_candidate_research_2024"
ALPHAS = [1., 10., 100., 1000., 10000.]
END = pd.Timestamp("2025-01-01", tz="UTC")


def dynamic_features(close, minute, window):
    """Unscaled causal volatility levels and known retained-window moments."""
    returns = 100 * np.log(close).diff()
    _, y, base = features(close, minute, 28, window)
    columns = [base, np.log(base)]
    for span in [1, 3, 7, 14, 28]:
        columns.extend([
            returns.rolling(span).mean().to_numpy(),
            np.sqrt((returns**2).rolling(span).mean()).to_numpy(),
            np.sqrt((returns.clip(upper=0)**2).rolling(span).mean()).to_numpy(),
        ])
        for j in range(minute.shape[1]):
            columns.append(pd.Series(minute[:, j]).rolling(span).mean().to_numpy())
    columns.append((returns**2).rolling(28).std(ddof=0).to_numpy())
    known, persistence_decay = [], []
    for horizon in range(1, 8):
        retained = window-horizon
        sums = returns.rolling(retained).sum().to_numpy() if retained else np.zeros(len(close))
        squares = (returns**2).rolling(retained).sum().to_numpy() if retained else np.zeros(len(close))
        # E[(sample mean)^2] includes variance of unseen future return sums.
        k = squares/window-(sums/window)**2
        c = horizon*(window-1)/window**2
        known.append(k)
        persistence_decay.append(np.sqrt(np.maximum(k+c*base**2, 0)))
        columns.extend([sums/window, np.sqrt(np.maximum(k, 0))])
    return np.column_stack(columns), y, base, np.column_stack(persistence_decay)


def continuous_blend(actual, boosted, ridge):
    """Closed-form MSE-minimizing convex weight for each horizon, on 2024 only."""
    difference = ridge-boosted
    weights = np.clip(np.sum((actual-boosted)*difference, axis=0)
                      / np.maximum(np.sum(difference**2, axis=0), 1e-20), 0, 1)
    return boosted+weights*difference, weights


def ridge_model(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="cholesky"))


def fit_candidates(folds, X, y, base, H, D, decay, candidate, xgb_variants=True, jobs=1):
    boost_forecasts = {"source_relative": []}
    if xgb_variants:
        boost_forecasts.update({key: [] for key in ["mse_relative", "mse_decay", "mse_shallow", "level_correction"]})
    ridge_forecasts = {(mode, alpha): [] for mode in ["relative", "mse_relative", "level", "decay"] for alpha in ALPHAS}
    for tr, va in folds:
        assert tr.max()+7 < va.min()
        weights = base[tr]**2
        weights /= weights.mean()
        relative = y[tr]/base[tr, None]-1
        for mode, alpha in ridge_forecasts:
            model = ridge_model(alpha)
            matrix = H if mode in ["relative", "mse_relative"] else D
            target = relative if mode in ["relative", "mse_relative"] else (y[tr] if mode == "level" else y[tr]-decay[tr])
            kwargs = {"ridge__sample_weight": weights} if mode == "mse_relative" else {}
            model.fit(matrix[tr], target, **kwargs)
            p = model.predict(matrix[va])
            if mode in ["relative", "mse_relative"]:
                p = base[va, None]*(1+p)
            elif mode == "decay":
                p = p+decay[va]
            ridge_forecasts[(mode, alpha)].append(np.maximum(p, 0))
        for variant in boost_forecasts:
            model = estimator(candidate)
            model.set_params(n_jobs=jobs)
            if variant == "mse_shallow":
                model.set_params(max_depth=2, learning_rate=.03, min_child_weight=10, reg_lambda=30)
            if variant in ["mse_decay", "mse_shallow"]:
                target = (y[tr]-decay[tr])/base[tr, None]
            elif variant == "level_correction":
                target = y[tr]-base[tr, None]
            else:
                target = relative
            kwargs = {"sample_weight": weights} if variant.startswith("mse") else {}
            model.fit(X[tr], target, **kwargs)
            p = model.predict(X[va])
            if variant in ["mse_decay", "mse_shallow"]:
                p = decay[va]+base[va, None]*p
            elif variant == "level_correction":
                p = base[va, None]+p
            else:
                p = base[va, None]*(1+p)
            boost_forecasts[variant].append(np.maximum(p, 0))
    return {k: np.vstack(v) for k, v in boost_forecasts.items()}, {k: np.vstack(v) for k, v in ridge_forecasts.items()}


def run(symbols=None, windows=None, xgb_variants=True, cache_only=False, jobs=1):
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oof").mkdir(exist_ok=True)
    panel = load_panel()
    # Discard test rows before deriving any targets, features or summary values.
    panel = panel.loc[panel.index < END].copy()
    folds, eligible, _ = calendars(panel)
    assert panel.index.max() < END
    source = pd.read_csv(ROOT / "results/optimized_minute_xgboost_2023_2025/selected_inputs.csv")
    original = pd.read_csv(ROOT / "results/improved_classical_2023_2025/selected_inputs.csv")
    config = dict(last_available_day=str(panel.index.max()), validation_year=2024,
                  training="2023 plus observed earlier 2024 labels; six purged expanding folds",
                  objective="absolute MSE across all seven horizons", alphas=ALPHAS,
                  candidate_selection="2024 only; search scores include model-selection optimism",
                  source_selection_sha256=hashlib.sha256((ROOT / "results/optimized_minute_xgboost_2023_2025/selected_inputs.csv").read_bytes()).hexdigest(),
                  xgb_variants=xgb_variants,
                  boosters=["source_relative", "mse_relative", "mse_decay", "mse_shallow", "level_correction"],
                  ridge_modes=["relative", "mse_relative", "level", "decay"])
    if not cache_only:
        (OUT / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (OUT / "status.json").write_text(json.dumps(dict(status="running")), encoding="utf-8")
    search, selected, forecast_rows = [], [], []
    for symbol in (symbols or list(panel.columns)):
        minute = minute_features(np.load(DATA / f"{symbol}.npy", mmap_mode="r")[:len(panel)*1440])
        for window in (windows or LAGS):
            selected_source = source.query("symbol == @symbol and volatility_window == @window").iloc[0]
            original_choice = original.query("symbol == @symbol and volatility_window == @window").iloc[0]
            X, y, base = features(panel[symbol], minute, int(selected_source.input_window), window)
            H, _, _ = har_features(panel[symbol], minute, window)
            D, _, _, decay = dynamic_features(panel[symbol], minute, window)
            required = np.concatenate([np.r_[tr, va] for tr, va in folds])
            assert np.isfinite(X[required]).all() and np.isfinite(H[required]).all() and np.isfinite(D[required]).all()
            actual = np.vstack([y[va] for _, va in folds])
            assert all(panel.index[va.max()+7] < END for _, va in folds)
            origins = np.concatenate([va for _, va in folds])
            cache_path = OUT / "oof" / f"{symbol}_v{window}.npz"
            if cache_path.exists():
                with np.load(cache_path) as cache:
                    np.testing.assert_array_equal(actual, cache["actual"])
                    np.testing.assert_array_equal(origins, cache["origins"])
                    boosts = {key.removeprefix("boost__"): cache[key] for key in cache.files if key.startswith("boost__")}
                    ridges = {(key.split("__")[1], float(key.split("__")[2])): cache[key] for key in cache.files if key.startswith("ridge__")}
                print(f"Reusing {cache_path.name}", flush=True)
            else:
                boosts, ridges = fit_candidates(folds, X, y, base, H, D, decay, int(selected_source.candidate), xgb_variants, jobs)
            original_pred = ((1-original_choice.ridge_weight)*boosts["source_relative"]
                             +original_choice.ridge_weight*ridges[("relative", float(original_choice.alpha))])
            baseline_mse = float(np.mean((actual-original_pred)**2))
            if not cache_path.exists():
                np.savez_compressed(cache_path, actual=actual, origins=origins, baseline=original_pred,
                                    lag=int(selected_source.input_window), candidate=int(selected_source.candidate),
                                    **{f"boost__{key}": value for key, value in boosts.items()},
                                    **{f"ridge__{mode}__{int(alpha)}": value for (mode, alpha), value in ridges.items()})
            if cache_only:
                print(f"OOF ready: {symbol} v{window}", flush=True)
                continue
            options = []
            for boost_name, boost_pred in boosts.items():
                for (ridge_name, alpha), ridge_pred in ridges.items():
                    p, horizon_weights = continuous_blend(actual, boost_pred, ridge_pred)
                    row = dict(symbol=symbol, volatility_window=window,
                               booster=boost_name, ridge_mode=ridge_name, alpha=alpha,
                               ridge_weights=json.dumps(horizon_weights.tolist()),
                               validation_mse=float(np.mean((actual-p)**2)), original_mse=baseline_mse,
                               mse_improvement_pct=100*(baseline_mse-np.mean((actual-p)**2))/baseline_mse)
                    search.append(row)
                    options.append((row, p))
            best, best_pred = min(options, key=lambda pair: pair[0]["validation_mse"])
            selected.append(best)
            for i, origin in enumerate(origins):
                for h in range(7):
                    forecast_rows.append(dict(symbol=symbol, volatility_window=window,
                                              origin=str(panel.index[origin]), horizon=h+1,
                                              actual=actual[i, h], forecast=best_pred[i, h], original=original_pred[i, h]))
            pd.DataFrame(search).to_csv(OUT / "search.csv", index=False)
            pd.DataFrame(selected).to_csv(OUT / "selected_candidates.csv", index=False)
            print(json.dumps(best), flush=True)
    if not cache_only:
        pd.DataFrame(forecast_rows).to_csv(OUT / "validation_predictions.csv.gz", index=False, compression="gzip")
        (OUT / "status.json").write_text(json.dumps(dict(status="complete", seconds=time.time()-started,
                                                         test_data_evaluated=False), indent=2), encoding="utf-8")
    print(f"Research complete in {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--windows", nargs="+", type=int)
    parser.add_argument("--ridge-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true", help="Write only independent OOF NPZ caches; safe for disjoint assets")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.symbols, args.windows, not args.ridge_only, args.cache_only, args.jobs)
