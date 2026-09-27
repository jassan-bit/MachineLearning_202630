"""Auditoría de consistencia de DEVELOPMENT; no corrige ni elimina registros."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/splits/development_80.csv'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT']
PRICES = ['open', 'high', 'low', 'close']
VOLUMES = ['volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']


def main():
    fingerprint = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    raw = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    assert len(raw) == 214165
    numeric = PRICES + VOLUMES + ['number_of_trades']
    df = raw[numeric].apply(pd.to_numeric, errors='coerce')
    times = {c: pd.to_datetime(raw[c], format='ISO8601', utc=True, errors='coerce') for c in ['open_time', 'close_time']}
    results, details = [], []

    def check(name, eligible, fail):
        eligible = pd.Series(eligible, index=raw.index).fillna(False)
        fail = pd.Series(fail, index=raw.index).fillna(False) & eligible
        results.append({'check': name, 'evaluated': int(eligible.sum()),
                        'not_evaluated': int((~eligible).sum()), 'flagged': int(fail.sum())})
        for index in raw.index[fail]:
            details.append({'csv_line': int(index)+2, 'symbol': raw.at[index, 'symbol'],
                            'open_time': raw.at[index, 'open_time'], 'close_time': raw.at[index, 'close_time'], 'check': name})

    finite = np.isfinite(df)
    for field in numeric:
        check('No numérico o no finito: '+field, True, ~finite[field])
    check('Precio negativo', finite[PRICES].all(axis=1), df[PRICES].lt(0).any(axis=1))
    check('Precio igual a cero', finite[PRICES].all(axis=1), df[PRICES].eq(0).any(axis=1))
    check('Volumen negativo', finite[VOLUMES].all(axis=1), df[VOLUMES].lt(0).any(axis=1))
    valid_price = finite[PRICES].all(axis=1)
    check('high < low', valid_price, df.high < df.low)
    for field in ['open', 'close']:
        check(field+' fuera de [low, high]', valid_price, (df[field] < df.low) | (df[field] > df.high))
    trades = df.number_of_trades
    check('Número de operaciones negativo o no entero', finite.number_of_trades, (trades < 0) | (trades % 1 != 0))
    for field, stamps in times.items():
        check('Timestamp inválido: '+field, True, stamps.isna())
        check('Zona UTC no explícita: '+field, True, ~raw[field].str.contains(r'(?:\+00:00|Z)$', regex=True))
    opening, closing = times['open_time'], times['close_time']
    check('Apertura fuera de DEVELOPMENT', opening.notna(), ~opening.between(pd.Timestamp('2020-08-11 06:00', tz='UTC'), pd.Timestamp('2025-07-01 18:00', tz='UTC')))
    check('Apertura no alineada a hora exacta', opening.notna(), opening != opening.dt.floor('h'))
    check('Cierre distinto de apertura + 1h - 1ms', opening.notna() & closing.notna(), closing-opening != pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1))
    check('Cierre fuera del intervalo horario', opening.notna() & closing.notna(), (closing < opening) | (closing >= opening+pd.Timedelta(hours=1)))
    check('Símbolo no esperado', True, ~raw.symbol.isin(SYMBOLS))
    # Tolerancia explícita de redondeo: 1e-8 absoluto + 1e-8 relativo.
    for base, quote, label in [('volume', 'quote_asset_volume', 'total'),
                                ('taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'taker')]:
        eligible = finite[[base, quote]].all(axis=1) & valid_price & df[base].gt(0)
        ratio = df[quote] / df[base].where(df[base].gt(0))
        tol = 1e-8 + 1e-8*np.maximum(df.low.abs(), df.high.abs())
        check('Precio implícito fuera de [low, high]: '+label, eligible,
              (ratio < df.low-tol) | (ratio > df.high+tol))
        check('Volumen base cero con cotizado no cero: '+label,
              finite[[base, quote]].all(axis=1) & df[base].eq(0), df[quote].abs() > 1e-8)
    for part, total in [('taker_buy_base_asset_volume', 'volume'), ('taker_buy_quote_asset_volume', 'quote_asset_volume')]:
        check('Volumen comprador supera total: '+part, finite[[part, total]].all(axis=1),
              df[part] > df[total] + 1e-8 + 1e-8*df[total].abs())
    check('Cero operaciones con volumen no cero', finite[VOLUMES].all(axis=1) & finite.number_of_trades & trades.eq(0), df[VOLUMES].abs().gt(1e-8).any(axis=1))
    table = pd.DataFrame(results)
    target = ROOT / 'outputs/tables'
    target.mkdir(parents=True, exist_ok=True)
    table.to_csv(target / 'development_consistency_checks.csv', index=False)
    pd.DataFrame(details, columns=['csv_line', 'symbol', 'open_time', 'close_time', 'check']).to_csv(target / 'development_consistency_flags.csv', index=False)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == fingerprint
    (target / 'development_consistency_metadata.json').write_text(json.dumps({
        'source': 'data/splits/development_80.csv', 'rows': len(raw), 'sha256': fingerprint,
        'unique_flagged_rows': len({d['csv_line'] for d in details}),
        'unit_check_tolerance': '1e-8 absoluto + 1e-8 relativo; cero: 1e-8 absoluto',
        'scope': 'Consistencia interna; no certifica unidades ni exactitud frente a fuente externa.',
        'pandas': pd.__version__, 'numpy': np.__version__
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(table.to_string(index=False))
    print('Filas distintas señaladas:', len({d['csv_line'] for d in details}))
    return table


if __name__ == '__main__':
    main()
