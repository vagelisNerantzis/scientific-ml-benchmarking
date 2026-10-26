from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse,json
import pandas as pd
from scientific_benchmark.statistics.bootstrap import bootstrap_metrics
from scientific_benchmark.statistics.paired import mcnemar_by_seed

def main():
    p=argparse.ArgumentParser(description='Post-hoc fixed-prediction physical-sample inference')
    p.add_argument('predictions',type=Path);p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--columns',nargs='+',default=['prediction']);p.add_argument('--metric',default='macro_f1')
    p.add_argument('--repetitions',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();frame=pd.read_csv(a.predictions,dtype={'sample_id':str,'label':str,**{c:str for c in a.columns}})
    result=bootstrap_metrics(frame,labels=a.labels,prediction_columns=a.columns,metric=a.metric,repetitions=a.repetitions,seed=a.seed)
    result.pop('replicates')
    if len(a.columns)==2:
        result['mcnemar_per_seed']=mcnemar_by_seed(frame,*a.columns)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
