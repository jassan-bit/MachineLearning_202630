"""Reserva TEST por timestamps globales observados; no realiza EDA ni modelado.

Usa la biblioteca estándar y conserva las columnas y valores CSV originales.
open_time se convierte a datetime UTC en memoria para ordenar y particionar.
"""

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/crypto_binance_master_1h.csv"
SPLITS = ROOT / "data/splits"
TABLES = ROOT / "outputs/tables"
SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"}


def read_master(path):
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or not {"open_time", "symbol"}.issubset(columns):
            raise ValueError("Faltan las columnas open_time o symbol.")
        if len(columns) != len(set(columns)):
            raise ValueError("El CSV contiene nombres de columnas duplicados.")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Fila {line}: número de campos incorrecto.")
            try:
                stamp = datetime.fromisoformat(row["open_time"])
                if stamp.tzinfo is None:
                    raise ValueError("open_time no tiene zona horaria explícita")
                stamp = stamp.astimezone(timezone.utc)
            except ValueError as exc:
                raise ValueError(f"Fila {line}: {exc}") from exc
            records.append((stamp, row))
    records.sort(key=lambda record: (record[0], record[1]["symbol"]))
    return columns, records


def partition(records):
    if {row["symbol"] for _, row in records} != SYMBOLS:
        raise ValueError("El dataset debe contener exactamente los cinco símbolos requeridos.")
    coverage = defaultdict(Counter)
    for stamp, row in records:
        coverage[stamp][row["symbol"]] += 1
    # Impide continuar si los activos no comparten el mismo calendario observado
    # o si existen duplicados. No corrige ni elimina ninguna observación.
    for stamp, counts in coverage.items():
        if set(counts) != SYMBOLS or any(count != 1 for count in counts.values()):
            raise ValueError(f"Cobertura desigual o timestamp duplicado: {stamp.isoformat()}")
    timestamps = sorted(coverage)
    split_index = int(len(timestamps) * 0.80)
    if not 0 < split_index < len(timestamps):
        raise ValueError("No hay suficientes timestamps para dos particiones no vacías.")
    cutoff = timestamps[split_index]
    development = [record for record in records if record[0] < cutoff]
    test = [record for record in records if record[0] >= cutoff]
    development_times = {stamp for stamp, _ in development}
    test_times = {stamp for stamp, _ in test}

    # Equivalentes con biblioteca estándar a las assertions de DataFrame:
    # development['open_time'].max() < test['open_time'].min()
    # set(development/test['symbol'].unique()) == SYMBOLS
    assert max(development_times) < min(test_times)
    assert {row["symbol"] for _, row in development} == SYMBOLS
    assert {row["symbol"] for _, row in test} == SYMBOLS
    assert development_times.isdisjoint(test_times)
    assert development_times == set(timestamps[:split_index])
    assert test_times == set(timestamps[split_index:])
    assert len(development) + len(test) == len(records)
    assert development + test == records
    for subset, expected in ((development, development_times), (test, test_times)):
        for symbol in SYMBOLS:
            assert {stamp for stamp, row in subset if row["symbol"] == symbol} == expected
    return timestamps, development, test


def write_csv(path, columns, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    columns, records = read_master(SOURCE)
    timestamps, development, test = partition(records)
    audit, by_symbol = [], []
    for name, subset in (("DEVELOPMENT", development), ("TEST", test)):
        unique = {stamp for stamp, _ in subset}
        audit.append({
            "split": name,
            "n_unique_timestamps": len(unique),
            "n_rows": len(subset),
            "percentage_timestamps": 100 * len(unique) / len(timestamps),
            "start_time": min(unique).isoformat(),
            "end_time": max(unique).isoformat(),
            "n_symbols": len({row["symbol"] for _, row in subset}),
        })
        for symbol in sorted(SYMBOLS):
            times = [stamp for stamp, row in subset if row["symbol"] == symbol]
            by_symbol.append({
                "split": name, "symbol": symbol, "n_rows": len(times),
                "start_time": min(times).isoformat(), "end_time": max(times).isoformat(),
            })

    summary = [
        "PARTICIÓN CRONOLÓGICA COMPLETADA",
        f"Fuente: {SOURCE}", f"SHA-256 del maestro: {source_hash}",
        "Unidad de partición: timestamps únicos observados reales, ordenados en UTC.",
        "Regla: split_index = int(numero_de_timestamps_unicos * 0.80)",
        f"Total timestamps: {len(timestamps)}",
        f"Development timestamps: {audit[0]['n_unique_timestamps']}",
        f"Test timestamps: {audit[1]['n_unique_timestamps']}",
    ]
    for row in audit:
        summary.extend([
            f"\n{row['split']}:", f"Inicio: {row['start_time']}",
            f"Fin: {row['end_time']}", f"Filas: {row['n_rows']}",
            f"Porcentaje real de timestamps: {row['percentage_timestamps']:.10f}%",
            "Filas por activo:",
        ])
        summary.extend(f"  {item['symbol']}: {item['n_rows']}"
                       for item in by_symbol if item['split'] == row['split'])
    summary.extend([
        f"\nÚltimo timestamp de DEVELOPMENT: {audit[0]['end_time']}",
        f"Primer timestamp de TEST: {audit[1]['start_time']}",
        f"Fecha de corte (primer timestamp de TEST, inclusivo): {audit[1]['start_time']}",
        "Overlap temporal: 0", "Shuffle utilizado: False",
        "Verificado: max(open_time DEVELOPMENT) < min(open_time TEST).",
        "Verificado: los cinco activos utilizan exactamente el mismo corte temporal.",
        "Verificado: todas las filas y columnas originales se conservan.",
        "Los huecos existentes se conservan: no se imputan ni se generan horas nuevas.",
        "\nRESERVA METODOLÓGICA DEL TEST:",
        "TEST queda reservado hasta la evaluación final.",
        "No usar TEST para EDA que determine decisiones de modelado, selección de variables,",
        "rezagos, ventanas, escalado, selección de hiperparámetros o validación cruzada.",
        "La validación cruzada temporal se realizará SOLO dentro de DEVELOPMENT.",
        "Posteriormente se aplicará TimeSeriesSplit con cinco folds dentro de DEVELOPMENT.",
        "Esta etapa no ejecuta EDA, features, retornos, volatilidad, rezagos, ventanas,",
        "escalado, TimeSeriesSplit ni entrenamiento de modelos.",
    ])
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    SPLITS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    paths = [SPLITS / "development_80.csv", SPLITS / "test_20.csv",
             TABLES / "chronological_split_audit.csv",
             TABLES / "chronological_split_by_symbol.csv",
             TABLES / "chronological_split_metadata.txt"]
    write_csv(paths[0], columns, (row for _, row in development))
    write_csv(paths[1], columns, (row for _, row in test))
    write_csv(paths[2], tuple(audit[0]), audit)
    write_csv(paths[3], tuple(by_symbol[0]), by_symbol)
    summary.append("\nArchivos creados:")
    summary.extend(str(path) for path in paths)
    report = "\n".join(summary) + "\n"
    paths[4].write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error, AssertionError) as exc:
        print(f"ERROR en la partición cronológica: {exc}", file=sys.stderr)
        sys.exit(1)
