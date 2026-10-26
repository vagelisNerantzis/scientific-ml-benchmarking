from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse,json
import pandas as pd
from scientific_benchmark.reporting.schemas import validate_result
from scientific_benchmark.reporting.aggregation import aggregate_seeds

def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=json.loads(a.results.read_text(encoding='utf-8'))
    for row in records:
        validate_result(row)
    if a.output.exists():
        p.exit(2,'Refusing to overwrite output\n')
    aggregate_seeds(pd.DataFrame(records)).to_csv(a.output,index=False)

if __name__=='__main__':
    main()
