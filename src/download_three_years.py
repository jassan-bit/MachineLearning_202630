"""Fetch only the 144 official monthly 1m archives needed for 2023–2025."""
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from download_minute_history import process
from minute_experiment import configuration,prepare_data


def main():
    cfg=configuration()
    months=pd.period_range(pd.Timestamp(cfg['start']).to_period('M'),
        (pd.Timestamp(cfg['end_exclusive'])-pd.Timedelta(days=1)).to_period('M'),freq='M').astype(str)
    tasks=[(symbol,month) for symbol in cfg['symbols'] for month in months]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(process,tasks): pass
    prepare_data()


if __name__=='__main__': main()
