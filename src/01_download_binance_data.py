"""Descarga y auditoría de OHLCV de Binance Spot. No realiza modelado.

Periodo inclusivo referido a open_time. CSV no tiene esquema de tipos:
en memoria se usan datetime UTC, Decimal e int; los CSV contienen números
decimales sin separadores de miles y fechas ISO 8601 con zona UTC.
Documentación: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
"""

import csv
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT")
START = datetime(2020, 8, 11, 6, tzinfo=timezone.utc)
END = datetime(2026, 9, 20, 23, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
URL = "https://api.binance.com/api/v3/klines"
PAUSE = 0.25
ATTEMPTS = 5
COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "symbol",
)


def milliseconds(value):
    return int(value.timestamp() * 1000)


def utc_from_ms(value):
    if type(value) is not int:
        raise ValueError(f"Timestamp no entero: {value!r}")
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=value)


def request_page(session, params):
    """Reintenta fallos transitorios; los demás errores HTTP son explícitos."""
    for attempt in range(ATTEMPTS):
        delay = 2 ** attempt
        try:
            time.sleep(PAUSE)
            response = session.get(URL, params=params, timeout=(10, 60))
        except requests.RequestException as exc:
            reason = f"Error de red: {exc}"
        else:
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RuntimeError("Binance devolvió JSON inválido") from exc
                if not isinstance(payload, list):
                    raise RuntimeError(f"Respuesta inesperada de Binance: {payload!r}")
                return payload
            reason = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code not in (418, 429) and not 500 <= response.status_code < 600:
                raise RuntimeError(reason)
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    raise RuntimeError(f"{reason}; Retry-After inválido: {retry_after}")
        if attempt == ATTEMPTS - 1:
            raise RuntimeError(f"{reason}; agotados {ATTEMPTS} intentos")
        print(f"  {reason}. Reintento en {delay:g} s.", flush=True)
        time.sleep(delay)


def parse_candle(raw, symbol):
    if not isinstance(raw, list) or len(raw) != 12:
        raise ValueError(f"Vela con estructura inesperada para {symbol}: {raw!r}")
    # El último campo de Binance se documenta como 'Ignore'.
    values = raw[:11]
    values[0] = utc_from_ms(values[0])
    values[6] = utc_from_ms(values[6])
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        values[index] = Decimal(str(values[index]))
        if not values[index].is_finite():
            raise ValueError(f"Valor numérico no finito en {symbol}")
    if type(values[8]) is not int or values[8] < 0:
        raise ValueError(f"number_of_trades inválido en {symbol}")
    return dict(zip(COLUMNS, values + [symbol]))


def download_symbol(session, symbol):
    rows = []
    cursor = milliseconds(START)
    end_ms = milliseconds(END)
    page_number = 0
    while cursor <= end_ms:
        page = request_page(session, {
            "symbol": symbol, "interval": "1h", "startTime": cursor,
            "endTime": end_ms, "limit": 1000,
        })
        if not page:
            # El resto del periodo se registra como faltante en la auditoría.
            break
        parsed = [parse_candle(raw, symbol) for raw in page]
        times = [milliseconds(row["open_time"]) for row in parsed]
        if times != sorted(times) or any(t < cursor or t > end_ms for t in times):
            raise RuntimeError(f"{symbol}: respuesta desordenada o fuera del rango solicitado")
        rows.extend(parsed)  # No deduplicar ni imputar.
        cursor = max(times) + 1  # startTime es inclusivo; avanzar sin solapamiento.
        page_number += 1
        print(f"  {symbol}: página {page_number}, {len(rows)} filas", flush=True)
        # Una página corta no implica necesariamente el final del historial.
    return rows


def missing_ranges(missing):
    ranges = []
    for stamp in sorted(missing):
        if ranges and stamp == ranges[-1][1] + HOUR:
            ranges[-1] = (ranges[-1][0], stamp)
        else:
            ranges.append((stamp, stamp))
    return ranges


def audit_symbol(symbol, rows, expected):
    stamps = [row["open_time"] for row in rows]
    counts = Counter(stamps)
    exact = Counter(tuple(row[column] for column in COLUMNS) for row in rows)
    missing = expected.difference(counts)
    ranges = missing_ranges(missing)
    audit = {
        "symbol": symbol,
        "n_rows": len(rows),
        "start_time": min(stamps) if stamps else None,
        "end_time": max(stamps) if stamps else None,
        "exact_duplicates": sum(n - 1 for n in exact.values()),
        "duplicate_timestamps": sum(n - 1 for n in counts.values()),
        "missing_hourly_timestamps": len(missing),
        "hourly_gap_ranges": len(ranges),
        "unexpected_timestamps": len(set(stamps).difference(expected)),
        "expected_hourly_timestamps": len(expected),
        "status": "ok" if rows else "empty",
        "error": "",
    }
    return audit, ranges


def serialize(value):
    return value.isoformat() if isinstance(value, datetime) else value


def write_csv(path, rows, columns):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: serialize(value) for key, value in row.items()} for row in rows)
    temporary.replace(path)


def main():
    for relative in ("data/raw", "data/processed", "src", "notebooks",
                     "outputs/figures", "outputs/tables"):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    expected = {START + i * HOUR for i in range((END - START) // HOUR + 1)}
    audits, master, created = [], [], []
    summary = [
        "Auditoría de descarga Binance Spot - velas 1h",
        f"Periodo solicitado de open_time (inclusivo): {START.isoformat()} a {END.isoformat()}",
        f"Timestamps horarios esperados por activo: {len(expected)}",
        "Duplicados: filas adicionales después de la primera ocurrencia.",
        "Huecos: rangos de horas faltantes, incluidos los extremos del periodo.",
        "No se imputan faltantes ni se eliminan duplicados.",
    ]
    failed = False
    with requests.Session() as session:
        for symbol in SYMBOLS:
            print(f"Descargando {symbol}...", flush=True)
            try:
                rows = download_symbol(session, symbol)
                audit, ranges = audit_symbol(symbol, rows, expected)
                path = ROOT / "data/raw" / f"{symbol}_1h.csv"
                write_csv(path, rows, COLUMNS)
                created.append(path)
                master.extend(rows)
                if not rows:
                    failed = True
                summary.append(
                    f"\n{symbol}: {len(rows)} filas; periodo real: "
                    f"{audit['start_time']} a {audit['end_time']}; "
                    f"duplicados exactos: {audit['exact_duplicates']}; "
                    f"timestamps duplicados: {audit['duplicate_timestamps']}; "
                    f"horas faltantes: {len(expected.difference(row['open_time'] for row in rows))}; "
                    f"rangos de huecos: {len(ranges)}; "
                    f"timestamps inesperados: {audit['unexpected_timestamps']}"
                )
                for start, end in ranges:
                    summary.append(f"  Faltante: {start.isoformat()} a {end.isoformat()} "
                                   f"({(end - start) // HOUR + 1} horas)")
            except (requests.RequestException, RuntimeError, ValueError, ArithmeticError, OSError) as exc:
                failed = True
                # Una descarga fallida no permite inferir cobertura ni faltantes.
                audit, _ = audit_symbol(symbol, [], expected)
                for key in audit:
                    if key not in ("symbol", "expected_hourly_timestamps", "status", "error"):
                        audit[key] = None
                audit.update(status="failed", error=str(exc))
                message = f"ERROR {symbol}: {exc}. No se publica una descarga parcial del activo."
                print(message, file=sys.stderr, flush=True)
                summary.append(f"\n{message}")
            audits.append(audit)
    if not failed:
        master.sort(key=lambda row: (row["symbol"], row["open_time"]))
        path = ROOT / "data/processed/crypto_binance_master_1h.csv"
        write_csv(path, master, COLUMNS)
        created.append(path)
        summary.append(f"\nTotal de filas del dataset consolidado: {len(master)}")
    else:
        summary.append("\nDataset consolidado NO generado: hubo descargas fallidas o vacías. "
                       "Archivos de ejecuciones anteriores, si existen, no representan esta ejecución.")
    path = ROOT / "outputs/tables/data_download_audit.csv"
    write_csv(path, audits, tuple(audits[0]))
    created.append(path)
    report_path = ROOT / "outputs/tables/data_download_audit.txt"
    created.append(report_path)
    summary.append("\nArchivos creados en esta ejecución:")
    summary.extend(str(path) for path in created)
    report = "\n".join(summary) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(report, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyboardInterrupt) as exc:
        print(f"ERROR: ejecución interrumpida o fallo de escritura: {exc}", file=sys.stderr)
        sys.exit(1)
