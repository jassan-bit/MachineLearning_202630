"""Build a local paired stationary-bootstrap report from frozen daily losses.

No models are fitted, loaded or replayed. The historical normal-HAC report is
an input retained unchanged; new results are written to a separate directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ml-bootstrap-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from volatility_dashboard.confidence_intervals import (
    BENCHMARK, MODELS, RIVALS, _comparison_rows,
)

LEGACY = ROOT / 'results/har_original_ml_confidence_20261006'
OUTPUT = ROOT / 'results/har_original_ml_bootstrap'
YEARS = (2025, 2026)
SCOPES = ('global', 7, 14, 21, 28)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_daily(path, year):
    frame = pd.read_csv(path)
    if set(frame.columns) != {'origin', *MODELS}:
        raise ValueError(f'Incomplete model panel: {path}')
    dates = pd.DatetimeIndex(pd.to_datetime(frame.pop('origin'), utc=True, errors='raise'))
    expected_n = {2025: 358, 2026: 236}[year]
    if (len(dates) != expected_n or dates.has_duplicates or dates.isna().any()
            or not dates.is_monotonic_increasing or not (dates.year == year).all()
            or dates[0] != pd.Timestamp(f'{year}-01-01', tz='UTC')
            or not np.all(np.diff(dates.asi8) == pd.Timedelta(days=1).value)):
        raise ValueError(f'Invalid frozen daily calendar: {path}')
    frame = frame.loc[:, list(MODELS)].astype(float)
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f'Nonfinite or negative squared losses: {path}')
    frame.index = dates
    frame.index.name = 'origin'
    return frame


def view_details(daily, year, scope, repetitions, seed, block_length=None):
    window = 28 if scope == 'global' else int(scope)
    horizon = 7
    block = max(window + horizon - 1, math.ceil(len(daily) ** (1 / 3)))
    return dict(year=year, n=len(daily), start=str(daily.index[0].date()),
                end=str(daily.index[-1].date()), symbol='TODOS', window=window,
                scope=scope, horizon='TODOS', horizon_hln=horizon,
                lags=max(window + horizon - 2, int(np.floor(4 * (len(daily)/100)**(2/9)))),
                assets=4, cells_per_origin=112 if scope == 'global' else 28,
                benchmark=BENCHMARK, rivals=list(RIVALS), family_size=6,
                bootstrap_repetitions=repetitions, bootstrap_seed=seed,
                bootstrap_block_length=block if block_length is None else block_length)


def plot_forest(frame, details, target):
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for i, rival in enumerate(RIVALS):
        row = frame.loc[frame.model.eq(rival)].iloc[0]
        color = '#0f766e' if row.star and row.winner == BENCHMARK else (
            '#b91c1c' if row.star else '#64748b')
        if np.isfinite(row.ci_low_gain) and np.isfinite(row.ci_high_gain):
            ax.hlines(i, row.ci_low_gain, row.ci_high_gain, color=color, linewidth=2.3)
            ax.plot([row.ci_low_gain, row.ci_high_gain], [i, i], '|', color=color,
                    markersize=11, markeredgewidth=1.5)
        ax.plot(row.gain, i, 'o', color=color, markersize=6)
    ax.axvline(0, color='#334155', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(RIVALS)), [name + (' *' if bool(frame.loc[
        frame.model.eq(name), 'star'].iloc[0]) else '') for name in RIVALS])
    ax.invert_yaxis()
    ax.set_xlabel('Ganancia MSE: rival − HAR original (puntos porcentuales²)')
    scope = 'global · cuatro ventanas' if details['scope'] == 'global' else f"ventana {details['scope']} días"
    ax.set_title(f"IC 95 % bootstrap estacionario · {details['year']} · {scope}\n"
                 f"{details['n']} fechas · {details['bootstrap_repetitions']:,} réplicas · "
                 f"longitud media de bloque {details['bootstrap_block_length']} días", fontsize=12)
    ax.grid(axis='x', alpha=.18)
    fig.text(.015, -.035, 'Barras: percentiles 2,5 y 97,5 del bootstrap emparejado. '
             '*: contraste DM-HLN con Holm6, criterio distinto del intervalo.\n'
             'Ganancia positiva favorece al original. Evaluación retrospectiva; '
             'los intervalos son individuales, no simultáneos ni predictivos.', fontsize=9)
    for extension in ('png', 'svg'):
        fig.savefig(target.with_suffix('.' + extension), dpi=160, bbox_inches='tight')
    plt.close(fig)


def _table(frame):
    return frame.to_html(index=False, border=0, float_format=lambda value: f'{value:.6g}',
                         classes='results', na_rep='No evaluable')


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def build(out=OUTPUT, legacy=LEGACY, repetitions=5000, seed=42, publish=False):
    out, legacy = Path(out).resolve(), Path(legacy).resolve()
    if out == legacy or out.is_relative_to(legacy):
        raise ValueError('Write the bootstrap report outside the historical report directory.')
    out.mkdir(parents=True, exist_ok=True)
    inputs, summary, all_rows, changed_bounds, sensitivities, views = {}, [], [], [], [], []
    for year in YEARS:
        for scope in SCOPES:
            source = legacy / str(year) / str(scope)
            target = out / str(year) / str(scope)
            target.mkdir(parents=True, exist_ok=True)
            for name in ('daily_loss.csv', 'comparisons.csv'):
                path = source / name
                inputs[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = digest(path)
            daily = read_daily(source / 'daily_loss.csv', year)
            details = view_details(daily, year, scope, repetitions, seed)
            table = _comparison_rows(daily, details, .05)
            # Legacy estimates and tests are controls, never bootstrap inputs.
            previous = pd.read_csv(source / 'comparisons.csv').set_index('model').loc[list(RIVALS)]
            if len(previous) != 6:
                raise ValueError(f'Historical comparison family is incomplete: {source}')
            for column in ('mse_original', 'mse_rival', 'gain', 'p_raw_DM_HLN', 'p_holm6'):
                np.testing.assert_allclose(table.set_index('model').loc[list(RIVALS), column],
                                           previous[column], rtol=1e-10, atol=1e-12)
            table.to_csv(target / 'comparisons.csv', index=False)
            # Reuse the historical public input for reproducibility; publishing
            # a second per-date loss panel is unnecessary for these intervals.
            plot_forest(table, details, target / 'forest_plot')
            view = table.assign(year=year, volatility_window=str(scope))
            all_rows.append(view)
            low, high = table.ci_low_gain, table.ci_high_gain
            summary.append(dict(year=year, volatility_window=str(scope), n=len(daily),
                mse_original=float(table.mse_original.iloc[0]),
                bootstrap_repetitions=repetitions, bootstrap_seed=seed,
                bootstrap_block_length=details['bootstrap_block_length'],
                original_wins_holm6=int((table.star & table.winner.eq(BENCHMARK)).sum()),
                rival_wins_holm6=int((table.star & ~table.winner.eq(BENCHMARK)).sum()),
                ci95_favors_original=int((low > 0).sum()),
                ci95_favors_rival=int((high < 0).sum()),
                ci95_includes_zero=int(((low <= 0) & (high >= 0)).sum()),
                ci_not_evaluable=int(low.isna().sum())))
            bounds = table[['model', 'gain', 'ci_low_gain', 'ci_high_gain']].copy()
            bounds = bounds.rename(columns={'ci_low_gain': 'bootstrap_low', 'ci_high_gain': 'bootstrap_high'})
            bounds['hac_normal_low'] = previous.loc[bounds.model, 'ci_low_gain'].to_numpy()
            bounds['hac_normal_high'] = previous.loc[bounds.model, 'ci_high_gain'].to_numpy()
            bounds['year'], bounds['volatility_window'] = year, str(scope)
            changed_bounds.append(bounds)
            primary = details['bootstrap_block_length']
            # Fixed alternatives, all reported: never select the favorable CI.
            for length in sorted({max(2, primary // 2), primary, min(len(daily)-1, primary*2)}):
                check = table if length == primary else _comparison_rows(
                    daily, view_details(daily, year, scope, repetitions, seed, length), .05)
                sensitivities.append(check[['model', 'gain', 'ci_low_gain', 'ci_high_gain']]
                    .assign(year=year, volatility_window=str(scope), block_length=length,
                            repetitions=repetitions, seed=seed, primary=length == primary))
            details.update(alpha=.05, confidence_level=.95,
                           interval_method='stationary paired percentile bootstrap',
                           interval_simultaneous=False, retrospective=True)
            _write_json(target / 'protocol.json', details)
            views.append((year, scope, table, bounds))

    source_files = [Path(__file__).resolve(), ROOT / 'volatility_dashboard/confidence_intervals.py',
                    ROOT / 'volatility_dashboard/diebold_mariano.py']
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
    protocol = dict(
        publication='APPROVED FOR PUBLICATION' if publish else 'LOCAL PREVIEW ONLY',
        publication_authorized=bool(publish),
        no_model_training=True, no_prediction_changes=True,
        benchmark=BENCHMARK, rivals=list(RIVALS), years=list(YEARS),
        scopes=[str(scope) for scope in SCOPES], alpha=.05,
        statistic='mean(loss_rival - loss_original); positive favors original',
        bootstrap='Paired stationary bootstrap over consecutive daily origins, circular continuation',
        repetitions=repetitions, seed=seed, quantiles=[.025, .975], quantile_method='linear',
        expected_block_length='max(window + maximum_horizon - 1, ceil(n**(1/3))); global uses window=28',
        sensitivity='All views report floor(L/2), L and 2L without selecting a favorable result',
        pairing='One shared date resample for all six rivals; assets/horizons/windows already averaged within each date',
        normal_errors_assumed=False, interval_simultaneous=False, interval_predictive=False,
        assumptions='Approximate stationarity and weak temporal dependence; block-length heuristic is not an optimality guarantee',
        limitations='Conditional on fixed forecasts; no model/selection uncertainty or correction of prior test inspection',
        tests='DM-HLN/HAC with Holm6 retained unchanged as a separate criterion; stars are not bootstrap decisions',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=inputs, source_sha256=source_hashes,
    )
    _write_json(out / 'protocol.json', protocol)
    pd.concat(all_rows, ignore_index=True).to_csv(out / 'all_comparisons.csv', index=False)
    pd.DataFrame(summary).to_csv(out / 'comparison_summary.csv', index=False)
    pd.concat(changed_bounds, ignore_index=True).to_csv(out / 'hac_vs_bootstrap_bounds.csv', index=False)
    pd.concat(sensitivities, ignore_index=True).to_csv(out / 'block_length_sensitivity.csv', index=False)
    _write_json(out / 'summary.json', dict(status='complete', publication=protocol['publication'],
        publication_authorized=bool(publish), comparison_summary=summary, input_sha256=inputs,
        source_sha256=source_hashes, confidence_level=.95, interval_simultaneous=False))
    sections = []
    display_names = {'model': 'Rival', 'gain': 'Ganancia MSE',
        'ci_low_gain': 'IC bootstrap inferior', 'ci_high_gain': 'IC bootstrap superior',
        'ci_conclusion': 'Lectura IC individual', 'p_holm6': 'p DM Holm6',
        'conclusion_5pct': 'Conclusión DM Holm6'}
    for year, scope, table, bounds in views:
        relative = f'{year}/{scope}'
        label = 'global · cuatro ventanas' if scope == 'global' else f'ventana {scope} días'
        columns = [name for name in display_names if name in table.columns]
        sections.append(f'<section id="view-{year}-{scope}"><h2>{year} · {escape(label)}</h2>'
            f'<p><a href="{relative}/comparisons.csv">Descargar tabla CSV</a> · '
            f'<a href="{relative}/forest_plot.svg">Figura SVG</a> · '
            f'<a href="{relative}/protocol.json">Protocolo de esta vista</a></p>'
            f'<img src="{relative}/forest_plot.png" alt="Intervalos bootstrap de ganancia MSE: {year}, {escape(label)}">'
            + _table(table[columns].rename(columns=display_names))
            + '<details><summary>Comparar límites anteriores HAC y nuevos bootstrap</summary>'
            + _table(bounds.drop(columns=['year', 'volatility_window'])) + '</details></section>')
    html = '''<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intervalos bootstrap temporal · revisión local</title>
<style>body{font:16px system-ui,sans-serif;margin:auto;padding:28px;max-width:1180px;color:#183044;background:#f8fafc}
h1,h2{color:#12334d}p{line-height:1.6}.notice{padding:18px;background:#fff3cd;color:#664d03;border-left:5px solid #e0ad00}
section{background:#fffdf2;border:2px solid #e0ad00;padding:20px;margin:25px 0;overflow:auto;border-radius:8px}
img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:left;border-bottom:1px solid #dce4ec}
th{background:#fff3cd}a{color:#07687a}nav a{display:inline-block;margin:6px 12px 6px 0}summary{cursor:pointer;margin:20px 0}</style>
<h1>Intervalos de confianza por bootstrap temporal</h1>
__PUBLICATION_NOTICE__
<p>Se estima la ganancia de MSE: pérdida del rival menos pérdida del HAR-Ridge + XGBoost original.
Un valor positivo favorece al original. Cada réplica remuestrea bloques de fechas consecutivas y usa las mismas fechas para todos los modelos.
Se aplican percentiles 2,5 y 97,5 a las ganancias medias remuestreadas, sin una fórmula normal para el intervalo.</p>
<p>Los intervalos del 95 % son individuales: no tienen ajuste por comparaciones múltiples ni predicen la volatilidad futura.
Los asteriscos conservan el contraste DM-HLN con Holm6; ese contraste es distinto de la lectura del intervalo bootstrap.</p>
<p>El bootstrap requiere condiciones de dependencia débil y estacionariedad aproximada. La longitud media de bloque es una regla explícita,
no una garantía de capturar toda la dependencia; se muestran alternativas de sensibilidad. Ambos periodos se habían explorado y siguen siendo retrospectivos.</p>
<p class="notice"><strong>Sensibilidad observada:</strong> en 2025, ventana de 7 días, el intervalo de Random Forest incluye cero con bloques medios de 6 y 13 días,
pero lo excluye con 26 días. En la vista global de 2026, el intervalo de Lasso incluye cero con 17 días y lo excluye con 34 y 68 días.
Estas lecturas dependen de la longitud del bloque: se conserva la regla principal declarada y se muestran todas las alternativas, sin elegir el intervalo más favorable.</p>
'''
    notice = ('<p class="notice"><strong>Actualización · intervalos bootstrap.</strong> '
              'Los intervalos percentiles de bootstrap temporal reemplazan los intervalos normales HAC. '
              'Los cambios se identifican en amarillo. Se conservan los modelos y pronósticos originales.</p>'
              if publish else
              '<p class="notice"><strong>Vista previa local.</strong> '
              'Se conservan los modelos y pronósticos originales.</p>')
    html = html.replace('__PUBLICATION_NOTICE__', notice)
    html += (f'<p>{repetitions:,} réplicas por vista · semilla {seed} · '
        'longitud media de bloque: máximo entre ventana + horizonte máximo − 1 y ⌈n¹ᐟ³⌉.</p>'
        '<p><a href="protocol.json">Protocolo y hashes</a> · '
        '<a href="all_comparisons.csv">Todas las comparaciones</a> · '
        '<a href="block_length_sensitivity.csv">Sensibilidad a longitud de bloque</a> · '
        '<a href="hac_vs_bootstrap_bounds.csv">Límites HAC anteriores y bootstrap</a></p><nav>')
    html += ''.join(f'<a href="#view-{year}-{scope}">{year} / {scope}</a>' for year, scope, _, _ in views)
    html += '</nav>' + ''.join(sections) + '</html>'
    (out / 'report.html').write_text(html, encoding='utf-8')
    # Fail if any historical input changed during generation.
    for relative, expected in inputs.items():
        path = Path(relative) if Path(relative).is_absolute() else ROOT / relative
        if digest(path) != expected:
            raise ValueError(f'Historical input changed during generation: {path}')
    _write_json(out / 'verification.json', dict(status='passed', views=len(views),
        rows=sum(len(frame) for frame in all_rows),
        original_point_estimates_and_dm_tests_match=True,
        historical_inputs_unchanged=True, source_sha256=source_hashes,
        no_model_training=True, publication_authorized=bool(publish)))
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--legacy-root', type=Path, default=LEGACY)
    parser.add_argument('--repetitions', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--publish', action='store_true',
                        help='Generate approved publication labels; does not push or deploy.')
    arguments = parser.parse_args()
    print(build(arguments.out, arguments.legacy_root, arguments.repetitions,
                arguments.seed, publish=arguments.publish).to_string(index=False))


if __name__ == '__main__':
    main()
