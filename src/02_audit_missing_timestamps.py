"""Audita horas faltantes sin modificar el dataset ni realizar otras etapas.

observed_hours cuenta timestamps únicos del calendario esperado (no filas).
Un faltante exclusivo pertenece a un solo activo; uno común falta en los cinco.
Los límites del calendario esperado son inclusivos y se refieren a open_time.
Solo se utiliza la biblioteca estándar de Python.
"""

import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/crypto_binance_master_1h.csv"
OUTPUT = ROOT / "outputs/tables"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT")
START = datetime(2020, 8, 11, 6, tzinfo=timezone.utc)
END = datetime(2026, 9, 20, 23, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def read_timestamps(path):
    """Lee en modo solo lectura, convierte a UTC y ordena cada activo."""
    observed = {symbol: [] for symbol in SYMBOLS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"symbol", "open_time"}.issubset(reader.fieldnames or []):
            raise ValueError("El CSV debe contener symbol y open_time.")
        for line_number, row in enumerate(reader, start=2):
            symbol = row["symbol"]
            if symbol not in observed:
                raise ValueError(f"Fila {line_number}: símbolo inesperado {symbol!r}.")
            try:
                stamp = datetime.fromisoformat(row["open_time"])
                if stamp.tzinfo is None:
                    raise ValueError("open_time no tiene zona horaria explícita")
                stamp = stamp.astimezone(timezone.utc)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Fila {line_number}: open_time inválido: {exc}") from exc
            observed[symbol].append(stamp)
    return {symbol: sorted(stamps) for symbol, stamps in observed.items()}


def consecutive_sequences(timestamps):
    """Devuelve (inicio, fin, horas) de cada secuencia de horas faltantes."""
    sequences = []
    for stamp in sorted(timestamps):
        if sequences and stamp == sequences[-1][1] + HOUR:
            start, _, hours = sequences[-1]
            sequences[-1] = (start, stamp, hours + 1)
        else:
            sequences.append((stamp, stamp, 1))
    return sequences


def compare_missing(missing):
    common = set.intersection(*(missing[symbol] for symbol in SYMBOLS))
    exclusive = {
        symbol: missing[symbol] - set.union(
            *(missing[other] for other in SYMBOLS if other != symbol)
        )
        for symbol in SYMBOLS
    }
    identical = all(missing[symbol] == common for symbol in SYMBOLS)
    return common, exclusive, identical


def write_csv(path, columns, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def print_timestamps(timestamps):
    if not timestamps:
        print("  Ninguno.")
    for stamp in sorted(timestamps):
        print(f"  {stamp.isoformat()}")


def print_sequences(timestamps):
    sequences = consecutive_sequences(timestamps)
    print(f"  Secuencias consecutivas: {len(sequences)}")
    for start, end, hours in sequences:
        print(f"    {start.isoformat()} -> {end.isoformat()}: {hours} hora(s)")
    print(f"  Longitud máxima de un hueco consecutivo: "
          f"{max((hours for _, _, hours in sequences), default=0)} hora(s)")


def main():
    observed = read_timestamps(SOURCE)
    expected = {START + i * HOUR for i in range((END - START) // HOUR + 1)}
    missing = {symbol: expected - set(observed[symbol]) for symbol in SYMBOLS}
    common, exclusive, identical = compare_missing(missing)
    summary = []
    print(f"Dataset: {SOURCE}")
    print(f"Periodo UTC inclusivo: {START.isoformat()} -> {END.isoformat()}")
    print(f"Horas esperadas por activo: {len(expected)}")
    print("Horas observadas: timestamps únicos dentro del calendario esperado.")
    print("Huecos comunes/exclusivos se cuentan en timestamps horarios; "
          "las secuencias consecutivas se reportan por separado.")
    for symbol in SYMBOLS:
        unique = set(observed[symbol])
        count = len(unique & expected)
        missing_count = len(missing[symbol])
        percentage = 100 * missing_count / len(expected)
        summary.append({
            "symbol": symbol,
            "expected_hours": len(expected),
            "observed_hours": count,
            "missing_hours": missing_count,
            "missing_percentage": percentage,
        })
        print(f"\n{symbol}: {count} horas observadas; {missing_count} faltantes "
              f"({percentage:.8f}%).")
        print(f"  ¿Se confirman 20 timestamps faltantes? {'Sí' if missing_count == 20 else 'No'}")
        print(f"  Filas con timestamp repetido: {len(observed[symbol]) - len(unique)}")
        unexpected = unique - expected
        print(f"  Timestamps fuera del calendario esperado: {len(unexpected)}")
        if unexpected:
            print_timestamps(unexpected)
        print("  Timestamps faltantes (UTC):")
        print_timestamps(missing[symbol])
        print_sequences(missing[symbol])
        print(f"  Faltantes exclusivos de {symbol}: {len(exclusive[symbol])}")
        print_timestamps(exclusive[symbol])
        # No confundir faltantes compartidos por 2-4 activos con exclusivos.
        partial = missing[symbol] - common - exclusive[symbol]
        print(f"  Faltantes compartidos con algunos activos, pero no los cinco: {len(partial)}")

    print(f"\n¿Son exactamente iguales los conjuntos faltantes de los cinco activos? "
          f"{'Sí' if identical else 'No'}")
    print(f"Timestamps faltantes comunes a los cinco: {len(common)}")
    print_timestamps(common)
    print_sequences(common)
    print(f"¿Existen faltantes exclusivos de algún activo? "
          f"{'Sí' if any(exclusive.values()) else 'No'}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [OUTPUT / "missing_timestamps.csv",
             OUTPUT / "missing_timestamps_summary.csv",
             OUTPUT / "common_missing_timestamps.csv"]
    write_csv(paths[0], ("symbol", "missing_timestamp"), (
        {"symbol": symbol, "missing_timestamp": stamp.isoformat()}
        for symbol in SYMBOLS for stamp in sorted(missing[symbol])
    ))
    write_csv(paths[1], ("symbol", "expected_hours", "observed_hours", "missing_hours",
                         "missing_percentage"), summary)
    write_csv(paths[2], ("missing_timestamp",), (
        {"missing_timestamp": stamp.isoformat()} for stamp in sorted(common)
    ))
    print("\nArchivos de auditoría creados:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR de auditoría: {exc}", file=sys.stderr)
        sys.exit(1)
