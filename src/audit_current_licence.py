"""Archive documentary source/chronology without treating file metadata as receipts."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/current_delivery_audit/licence'
SNAPSHOT = OUT / 'TERMS_AND_CONDITIONS_20260930.md'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(ROOT / 'results/minute_2023_2025/sources.csv')
    records = []
    for row in source.itertuples(index=False):
        path = ROOT / str(row.source).replace('\\', '/')
        records.append(dict(symbol=row.symbol, month=row.month, sha256=row.sha256,
            source=str(path.relative_to(ROOT)).replace('\\', '/'),
            filesystem_creation_utc=datetime.fromtimestamp(path.stat().st_ctime, timezone.utc).isoformat(),
            filesystem_last_write_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()))
    pd.DataFrame(records).to_csv(OUT / 'source_file_metadata.csv', index=False)
    evidence = dict(
        consulted_at_utc=datetime.now(timezone.utc).isoformat(),
        upstream_commit='bd110bb04caad6ad964a0098809f18343b1e104b',
        upstream_commit_utc='2026-09-30T12:17:13Z',
        upstream_blob_sha='87d04e216f62c86a2c797fbb7637404d6d602687',
        immutable_source='https://github.com/binance/binance-public-data/blob/bd110bb04caad6ad964a0098809f18343b1e104b/TERMS_AND_CONDITIONS.md',
        terms_version='1.0', terms_stated_last_updated='2026-08-26',
        snapshot_sha256=hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        dataset_license='CC BY-NC-SA 4.0',
        archives=len(records),
        earliest_local_creation=min(r['filesystem_creation_utc'] for r in records),
        latest_local_creation=max(r['filesystem_creation_utc'] for r in records),
        chronology='The immutable public terms commit predates the local creation timestamps of all 144 source archives.',
        scope='Archived historical public terms, local file metadata and source hashes; local timestamps are not signed original download receipts or a legal certification.',
        attribution_file='DATA_LICENSE.md',
    )
    (OUT / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
