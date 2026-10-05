"""Evaluate saved HAR forecasts against the unchanged dashboard DM protocol.

This script only reads forecasts and metrics. It never fits or selects a model,
loads fitted artifacts, downloads data, or calls data_loader.repository().
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from volatility_dashboard.data_loader import FAMILIES
from volatility_dashboard.diebold_mariano import comparisons

HAR = "HAR-Ridge + XGBoost"
BASELINE_FAMILY = "improved_classical_2023_2025"
DEFAULT_OUT = ROOT / "results" / "optimized_har_xgboost_2023_2025"
KEYS = ["symbol", "volatility_window", "origin", "horizon"]
SYMBOLS = ["BNBUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT"]
WINDOWS = [7, 14, 21, 28]
HORIZONS = list(range(1, 8))
METRICS = ["r2", "rmse", "mae", "mse", "mape"]
OUTCOMES = ["wins", "no_significant_difference", "losses", "not_evaluable"]
SPANISH_OUTCOMES = {
    "wins": "Favorece HAR-Ridge + XGBoost",
    "no_significant_difference": "Sin diferencia significativa",
    "losses": "Favorece al rival",
    "not_evaluable": "No evaluable",
}


def read_forecasts(path: Path, label: str) -> pd.DataFrame:
    """Read one complete forecast panel, retaining its original numerical values."""
    frame = pd.read_csv(path)
    renamed = {"k-NN": "knn", "SVR Lineal": "svr"}.get(label)
    if "forecast" not in frame and renamed is not None:
        frame = frame.rename(columns={renamed: "forecast"})
    missing = set(KEYS + ["actual", "forecast"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame[KEYS + ["actual", "forecast"]].copy()
    frame["origin"] = pd.to_datetime(frame.origin, utc=True, errors="raise")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{path}: missing or duplicate forecast keys")
    if not np.isfinite(frame[["actual", "forecast"]].to_numpy(float)).all():
        raise ValueError(f"{path}: non-finite actual or forecast")
    if set(frame.symbol) != set(SYMBOLS):
        raise ValueError(f"{path}: expected BTC, ETH, BNB and XRP")
    if set(frame.volatility_window) != set(WINDOWS) or set(frame.horizon) != set(HORIZONS):
        raise ValueError(f"{path}: unexpected volatility windows or horizons")
    origins = frame.origin.drop_duplicates().sort_values()
    if not origins.dt.year.eq(2025).all():
        raise ValueError(f"{path}: evaluation must contain 2025 origins only")
    if len(origins) < max(WINDOWS) + max(HORIZONS):
        raise ValueError(f"{path}: too few daily origins for the existing HAC protocol")
    if not np.all(np.diff(origins.array.asi8) == pd.Timedelta(days=1).value):
        raise ValueError(f"{path}: daily origins are not consecutive")
    indexed = frame.set_index(KEYS).sort_index()
    expected = pd.MultiIndex.from_product(
        [SYMBOLS, WINDOWS, origins.tolist(), HORIZONS], names=KEYS
    ).sort_values()
    if not indexed.index.equals(expected):
        raise ValueError(f"{path}: incomplete asset/window/origin/horizon panel")
    frame["model"] = label
    return frame


def assert_alignment(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    """Require identical keys and exact targets; forecast values may differ."""
    if reference.duplicated(KEYS).any() or candidate.duplicated(KEYS).any():
        raise ValueError("Duplicate forecast keys prevent an aligned comparison")
    a = reference.set_index(KEYS).sort_index()
    b = candidate.set_index(KEYS).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Forecast keys differ between the compared models")
    if not np.array_equal(a.actual.to_numpy(), b.actual.to_numpy()):
        raise ValueError("Actual targets differ between the compared models")


def load_versions(out: Path):
    # Keep the original HAR location even after dashboard integration changes FAMILIES.
    families = dict(FAMILIES)
    families[HAR] = BASELINE_FAMILY
    if set(families) != {
        "k-NN", "Ridge", "Lasso", "Random Forest", "XGBoost", "SVR Lineal", HAR
    }:
        raise ValueError("The evaluator requires the original seven model labels")
    frames, sources = [], []
    reference = None
    original_har = None
    for label, family in families.items():
        path = ROOT / "results" / family / "predictions.csv.gz"
        frame = read_forecasts(path, label)
        if reference is not None:
            assert_alignment(reference, frame)
        else:
            reference = frame
        if label == HAR:
            original_har = frame
        frames.append(frame)
        sources.append(path)
    new_path = out / "predictions.csv.gz"
    new_har = read_forecasts(new_path, HAR)
    assert_alignment(original_har, new_har)
    baseline = pd.concat(frames, ignore_index=True)
    optimized = pd.concat(
        [baseline[baseline.model.ne(HAR)], new_har], ignore_index=True
    )
    return baseline, optimized, original_har, new_har, sources + [new_path]


def compare_filters(predictions: pd.DataFrame) -> pd.DataFrame:
    """Call the existing 21-pair/Holm implementation for each of the 160 filters."""
    models = sorted(predictions.model.unique().tolist())
    if len(models) != 7:
        raise ValueError("Holm correction requires the same seven models")
    rows = []
    for window in WINDOWS:
        for symbol in ["TODOS"] + SYMBOLS:
            for horizon in ["TODOS"] + HORIZONS:
                result = comparisons(predictions, symbol, window, horizon, models)
                if len(result) != 21:
                    raise ValueError("Each filter must retain all 21 pairwise comparisons")
                result.insert(0, "horizon", horizon)
                result.insert(0, "volatility_window", window)
                result.insert(0, "symbol", symbol)
                rows.append(result)
    result = pd.concat(rows, ignore_index=True)
    if len(result) != 160 * 21:
        raise ValueError("Expected 160 filters with 21 comparisons each")
    return result


def har_results(combined: pd.DataFrame) -> pd.DataFrame:
    """Orient every loss differential as HAR minus its rival, independent of order."""
    selected = combined[
        combined.modelo_a.eq(HAR) | combined.modelo_b.eq(HAR)
    ].copy()
    a_is_har = selected.modelo_a.eq(HAR)
    selected["rival"] = np.where(a_is_har, selected.modelo_b, selected.modelo_a)
    selected["har_mse_difference"] = np.where(
        a_is_har, selected.diferencia_mse, -selected.diferencia_mse
    )
    finite = np.isfinite(selected.p_holm.to_numpy(float))
    significant = finite & selected.p_holm.lt(0.05).to_numpy()
    selected["outcome"] = np.select(
        [~finite, significant & selected.har_mse_difference.lt(0),
         significant & selected.har_mse_difference.gt(0)],
        ["not_evaluable", "wins", "losses"],
        default="no_significant_difference",
    )
    return selected


def outcome_counts(selected: pd.DataFrame) -> dict:
    counts = {name: int(selected.outcome.eq(name).sum()) for name in OUTCOMES}
    counts["total"] = len(selected)
    counts["lower_mse"] = int(selected.har_mse_difference.lt(0).sum())
    counts["equal_mse"] = int(selected.har_mse_difference.eq(0).sum())
    counts["higher_mse"] = int(selected.har_mse_difference.gt(0).sum())
    return counts


def summarize(combined: pd.DataFrame) -> dict:
    selected = har_results(combined)
    result = outcome_counts(selected)
    for name, column in [
        ("by_window", "volatility_window"), ("by_asset", "symbol"),
        ("by_horizon", "horizon"), ("by_rival", "rival")
    ]:
        result[name] = []
        for value, group in selected.groupby(column, sort=False, dropna=False):
            if isinstance(value, np.generic):
                value = value.item()
            result[name].append({column: value, **outcome_counts(group)})
    if set(["symbol", "volatility_window", "horizon"]).issubset(selected.columns):
        filters = selected.groupby(["symbol", "volatility_window", "horizon"])
        result["filters_won_against_all_six_rivals"] = int(sum(
            len(group) == 6 and group.outcome.eq("wins").all()
            for _, group in filters
        ))
    return result


def recompute_macro(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in forecasts.groupby(["symbol", "volatility_window", "horizon"]):
        actual = group.actual.to_numpy(float)
        error = group.forecast.to_numpy(float) - actual
        denominator = np.sum((actual - actual.mean()) ** 2)
        if denominator == 0 or (actual == 0).any():
            raise ValueError("Macro R2 and MAPE require varying, nonzero targets")
        mse = float(np.mean(error ** 2))
        rows.append(dict(
            symbol=key[0], volatility_window=key[1], horizon=key[2],
            r2=float(1 - np.sum(error ** 2) / denominator),
            rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(error))), mse=mse,
            mape=float(np.mean(100 * np.abs(error) / actual)),
        ))
    scores = pd.DataFrame(rows)
    by_asset = scores.groupby("symbol", as_index=False)[METRICS].mean()
    global_macro = pd.DataFrame([dict(symbol="GLOBAL_MACRO", **scores[METRICS].mean().to_dict())])
    return pd.concat([by_asset, global_macro], ignore_index=True)


def read_verified_macro(path: Path, forecasts: pd.DataFrame, version: str) -> pd.DataFrame:
    macro = pd.read_csv(path)
    macro = macro[macro.model.eq("HAR_Ridge_XGBoost")].copy()
    if set(macro.symbol) != set(SYMBOLS + ["GLOBAL_MACRO"]) or macro.symbol.duplicated().any():
        raise ValueError(f"{path}: expected one HAR metric row per asset and global macro")
    actual = macro.set_index("symbol").sort_index()[METRICS]
    expected = recompute_macro(forecasts).set_index("symbol").sort_index()[METRICS]
    np.testing.assert_allclose(actual.to_numpy(float), expected.to_numpy(float), rtol=1e-10, atol=1e-12)
    macro["version"] = version
    return macro[["version", "symbol"] + METRICS]


def verify_unaffected_rival_tests(baseline: pd.DataFrame, optimized: pd.DataFrame) -> None:
    keys = ["symbol", "volatility_window", "horizon", "modelo_a", "modelo_b"]
    columns = ["mse_a", "mse_b", "n", "horizonte_hln", "rezagos_hac", "diferencia_mse", "dm", "p", "estado"]
    def rivals(frame):
        return frame[frame.modelo_a.ne(HAR) & frame.modelo_b.ne(HAR)].set_index(keys).sort_index()[columns]
    # Adjusted p-values may change because HAR's p-values change their Holm ranks.
    pd.testing.assert_frame_equal(rivals(baseline), rivals(optimized), check_exact=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_name(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def records(frame: pd.DataFrame) -> list:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def breakdown_table(summary: dict, name: str, column: str) -> str:
    a = pd.DataFrame(summary["baseline"][name]).set_index(column)
    b = pd.DataFrame(summary["optimized"][name]).set_index(column)
    count_columns = ["total", "wins", "no_significant_difference", "losses", "not_evaluable", "lower_mse"]
    table = a[count_columns].add_prefix("antes_").join(b[count_columns].add_prefix("después_"))
    return table.reset_index().to_html(index=False, escape=True)


def capture_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame.symbol.eq("TODOS") & frame.volatility_window.eq(7)
        & frame.horizon.eq("TODOS")
        & (frame.modelo_a.eq(HAR) | frame.modelo_b.eq(HAR))
    ].drop(columns=["symbol", "volatility_window", "horizon"])


def make_html(summary: dict, baseline_dm: pd.DataFrame, optimized_dm: pd.DataFrame, macro: pd.DataFrame) -> str:
    before, after = summary["baseline"], summary["optimized"]
    counts = pd.DataFrame([
        {"versión": "Original", **{key: before[key] for key in ["total"] + OUTCOMES + ["lower_mse"]}},
        {"versión": "Optimizada", **{key: after[key] for key in ["total"] + OUTCOMES + ["lower_mse"]}},
    ]).rename(columns={
        "total": "pruebas HAR", "wins": "victorias significativas", "losses": "derrotas significativas",
        "no_significant_difference": "sin diferencia significativa", "not_evaluable": "no evaluables",
        "lower_mse": "MSE inferior al rival",
    })
    breakdowns = "".join(
        f"<h2>{title}</h2>{breakdown_table(summary, name, column)}"
        for name, column, title in [
            ("by_window", "volatility_window", "Por ventana de volatilidad"),
            ("by_asset", "symbol", "Por activo seleccionado"),
            ("by_horizon", "horizon", "Por horizonte seleccionado"),
            ("by_rival", "rival", "Por rival"),
        ]
    )
    return f'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HAR-Ridge + XGBoost: evaluación completa de filtros</title>
<style>body{{font-family:Arial,sans-serif;max-width:1400px;margin:30px auto;padding:18px;color:#182b49;line-height:1.55}}
table{{border-collapse:collapse;margin:18px 0;font-size:14px}}td,th{{padding:8px;border:1px solid #d5dfeb;text-align:left}}
th{{background:#edf3fb}}.tables{{overflow-x:auto}}a{{color:#245fc2}}code{{background:#edf3fb;padding:3px}}</style></head><body>
<h1>HAR-Ridge + XGBoost: antes y después</h1>
<p>Se revisan las <b>160 combinaciones de filtros</b>: BTC, ETH, BNB, XRP o TODOS;
ventanas de 7, 14, 21 o 28 días; horizontes de 1 a 7 días o TODOS.
Cada selección conserva siete modelos y sus 21 pares, con seis comparaciones de HAR:
<b>960 pruebas de HAR por versión</b>, dentro de 3360 comparaciones totales.</p>
<p>La versión original obtiene <b>{before['wins']} victorias</b>, {before['no_significant_difference']} resultados sin diferencia significativa,
{before['losses']} derrotas y {before['not_evaluable']} pruebas no evaluables. La versión optimizada obtiene
<b>{after['wins']} victorias</b>, {after['no_significant_difference']} resultados sin diferencia significativa,
{after['losses']} derrotas y {after['not_evaluable']} pruebas no evaluables.
Hay {after['filters_won_against_all_six_rivals']}/160 selecciones donde gana significativamente a los seis rivales.</p>
<p>Las pérdidas cuadráticas se promedian dentro de cada origen diario antes de la prueba bilateral
de Diebold–Mariano. Se conserva la corrección Harvey–Leybourne–Newbold y HAC con pesos Bartlett
y rezagos <code>max(ventana + horizonte − 2, floor(4(n/100)^(2/9)))</code>.
La corrección de Holm se aplica a los 21 pares evaluables de <b>cada selección de filtros</b>.
No corrige conjuntamente las 960 pruebas de HAR ni las 3360 comparaciones de todas las selecciones.
Los filtros comparten observaciones y sus pruebas no son independientes.</p>
<p>Una victoria requiere pérdida media menor y <code>p_holm &lt; 0.05</code>.
Tener MSE menor por sí solo no demuestra una diferencia estadísticamente significativa.
Los seis rivales, sus predicciones, objetivos, fechas y etiquetas se mantienen en ambas evaluaciones;
los valores p ajustados de sus pares pueden cambiar porque cambian los rangos de Holm.</p>
<p>Esta es una <b>evaluación retrospectiva de 2025</b>, un periodo cuyos resultados ya se habían revisado;
no constituye un test nuevo sin examinar. El resultado no garantiza victoria en todos los filtros ni
rendimiento futuro. Este informe solo evalúa predicciones guardadas y no selecciona ni entrena modelos.</p>
<div class="tables"><h2>Recuento completo</h2>{counts.to_html(index=False)}
<h2>Métricas macro de HAR verificadas</h2>
<p>Medias uniformes por horizonte, ventana y activo. RMSE y MAE se expresan en puntos porcentuales de
volatilidad diaria no anualizada; MAPE es porcentaje. R² es un coeficiente de ajuste, no un porcentaje de aciertos.</p>
{macro.round(6).to_html(index=False)}
<h2>Las seis comparaciones de la captura: TODOS, ventana 7, TODOS los horizontes</h2>
<h3>Original</h3>{capture_rows(baseline_dm).round(6).to_html(index=False)}
<h3>Optimizada</h3>{capture_rows(optimized_dm).round(6).to_html(index=False)}
{breakdowns}</div>
<p><a href="dm_baseline_comparisons.csv">3360 comparaciones originales (CSV)</a> ·
<a href="dm_comparisons.csv">3360 comparaciones optimizadas (CSV)</a> ·
<a href="dm_summary.json">Resumen, metodología y huellas de archivos (JSON)</a></p>
<p>Reproducir: <code>.venv-repro/Scripts/python.exe src/report_optimized_har.py</code></p>
<footer>By Jassan Arteta y Mateo Bernal<br>© Copyright 2026.</footer></body></html>'''


def report(out: Path = DEFAULT_OUT) -> dict:
    out = Path(out).resolve()
    original_directories = {
        (ROOT / "results" / family).resolve() for family in FAMILIES.values()
        if family != "optimized_har_xgboost_2023_2025"
    } | {(ROOT / "results" / BASELINE_FAMILY).resolve()}
    if any(out == path or path in out.parents for path in original_directories):
        raise ValueError("The report output must not overwrite an original model family")
    baseline, optimized, original_har, new_har, sources = load_versions(out)
    baseline_macro_path = ROOT / "results" / BASELINE_FAMILY / "macro_metrics.csv"
    new_macro_path = out / "macro_metrics.csv"
    macro = pd.concat([
        read_verified_macro(baseline_macro_path, original_har, "Original"),
        read_verified_macro(new_macro_path, new_har, "Optimizada"),
    ], ignore_index=True)
    print("Evaluating original forecasts across 160 filter selections...", flush=True)
    baseline_dm = compare_filters(baseline)
    print("Evaluating optimized forecasts across 160 filter selections...", flush=True)
    optimized_dm = compare_filters(optimized)
    verify_unaffected_rival_tests(baseline_dm, optimized_dm)
    summary = {
        "evaluation": "retrospective_2025_previously_examined",
        "new_untouched_test": False,
        "filters": {"count": 160, "symbols": ["TODOS"] + SYMBOLS,
                    "volatility_windows": WINDOWS, "horizons": ["TODOS"] + HORIZONS},
        "models": sorted(FAMILIES),
        "pairwise_comparisons_per_filter": 21,
        "pairwise_comparisons_per_version": 3360,
        "har_comparisons_per_filter": 6,
        "har_comparisons_per_version": 960,
        "protocol": {
            "loss": "squared error, averaged within each daily origin",
            "test": "two-sided Diebold-Mariano with Harvey-Leybourne-Newbold correction",
            "hac": "Bartlett; lags=max(window+horizon-2, floor(4*(n/100)^(2/9)))",
            "alpha": 0.05,
            "multiplicity": "Holm across all evaluable pairs of the seven models within each filter selection",
            "multiplicity_across_all_filters": False,
            "filters_share_observations": True,
        },
        "alignment": {
            "identical_keys_and_exact_targets": True,
            "rival_predictions_unchanged": True,
            "rival_unadjusted_dm_tests_unchanged": True,
            "predictions_per_model": len(original_har),
            "daily_origins": int(original_har.origin.nunique()),
            "first_origin": original_har.origin.min().isoformat(),
            "last_origin": original_har.origin.max().isoformat(),
        },
        "baseline": summarize(baseline_dm),
        "optimized": summarize(optimized_dm),
        "macro_metrics": records(macro),
    }
    if summary["baseline"]["total"] != 960 or summary["optimized"]["total"] != 960:
        raise ValueError("Expected exactly 960 HAR comparisons per version")
    summary["all_har_comparisons_won"] = summary["optimized"]["wins"] == 960
    a, b = har_results(baseline_dm), har_results(optimized_dm)
    transitions = a[["symbol", "volatility_window", "horizon", "rival", "outcome"]].merge(
        b[["symbol", "volatility_window", "horizon", "rival", "outcome"]],
        on=["symbol", "volatility_window", "horizon", "rival"],
        suffixes=("_baseline", "_optimized"), validate="one_to_one",
    )
    summary["outcome_transitions"] = records(
        transitions.groupby(["outcome_baseline", "outcome_optimized"]).size().reset_index(name="count")
    )
    sources += [baseline_macro_path, new_macro_path,
                ROOT / "volatility_dashboard" / "diebold_mariano.py", Path(__file__).resolve()]
    summary["source_sha256"] = {source_name(path): file_sha256(path) for path in sources}
    # All checks complete before the first output write. Original inputs stay read-only.
    out.mkdir(parents=True, exist_ok=True)
    baseline_dm.to_csv(out / "dm_baseline_comparisons.csv", index=False)
    optimized_dm.to_csv(out / "dm_comparisons.csv", index=False)
    (out / "dm_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (out / "index.html").write_text(make_html(summary, baseline_dm, optimized_dm, macro), encoding="utf-8")
    print(json.dumps({"baseline": outcome_counts(a), "optimized": outcome_counts(b)}, indent=2), flush=True)
    print(f"Report: {out / 'index.html'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Directory containing the new predictions.csv.gz and macro_metrics.csv")
    report(parser.parse_args().out)


if __name__ == "__main__":
    main()
