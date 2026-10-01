"""Run identical seven-output fits concurrently into an isolated result directory."""
import os
os.environ['SVR_N_JOBS']='7'
from joblib import parallel_config
from threadpoolctl import threadpool_limits
import run_minute_experiment as runner


if __name__=='__main__':
    runner.OUT=runner.OUT.parent/'minute_2023_2025_parallel'
    os.environ['MINUTE_RESUME']='1'
    snapshot=runner.OUT/'selection_cache.csv'
    if snapshot.exists(): os.environ['MINUTE_SELECTION_CACHE']=str(snapshot)
    with parallel_config(backend='loky',inner_max_num_threads=1),threadpool_limits(limits=1):
        runner.run()
