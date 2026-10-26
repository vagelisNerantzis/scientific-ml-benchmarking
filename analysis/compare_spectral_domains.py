from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

# Post-hoc comparison of already frozen, paired predictions. No model fitting.
from scientific_benchmark.statistics.bootstrap import bootstrap_metrics
from scientific_benchmark.statistics.paired import mcnemar_by_seed

def compare_domains(frame, *, labels, repetitions=1000, seed=42):
    result=bootstrap_metrics(frame,labels=labels,prediction_columns=('prediction_t','prediction_a'),
                             repetitions=repetitions,seed=seed)
    result['mcnemar_per_seed']=mcnemar_by_seed(frame,'prediction_t','prediction_a')
    return result

if __name__=='__main__':
    from runpy import run_path
    run_path(str(Path(__file__).resolve().parents[1]/'scripts/bootstrap_metrics.py'),run_name='__main__')
