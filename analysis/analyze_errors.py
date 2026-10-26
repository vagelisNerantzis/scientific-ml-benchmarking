from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

"""Post-hoc aggregate errors; never feed this output into test-guided tuning."""
import pandas as pd
from scientific_benchmark.statistics.bootstrap import aligned_predictions

def error_summary(frame):
    aligned_predictions(frame,['prediction'])
    data=frame.assign(correct=frame.prediction.eq(frame.label))
    return data.groupby(['seed','label'],as_index=False).agg(n_samples=('sample_id','size'),accuracy=('correct','mean'))
