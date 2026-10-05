"""Verify all original saved models before publishing the lightweight dashboard audit."""
import argparse
import json
from pathlib import Path
import sys

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from volatility_dashboard import data_loader as loader
from volatility_dashboard.runtime_audit import build_audit, read_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check the existing file fingerprints without replaying models.')
    args = parser.parse_args()
    if args.check:
        report = read_audit(ROOT, loader.RUNTIME_AUDIT, loader.FAMILIES)
    else:
        with threadpool_limits(limits=1):
            predictions, _, audit = loader.repository(verify_models=True)
        report = build_audit(ROOT, loader.FAMILIES, predictions, audit)
        loader.RUNTIME_AUDIT.parent.mkdir(parents=True, exist_ok=True)
        with loader.RUNTIME_AUDIT.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=report['status'], verified_artifacts=report['verified_artifacts'],
                         fingerprinted_files=len(report['files']), output=str(loader.RUNTIME_AUDIT))))


if __name__ == '__main__':
    main()
