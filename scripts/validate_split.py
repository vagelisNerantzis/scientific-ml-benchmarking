from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse,json
import pandas as pd
from scientific_benchmark.splits.loader import load_split
from scientific_benchmark.splits.validation import validate_split,validate_observations

def main():
    parser=argparse.ArgumentParser(description='Validate existing membership; never generate a split')
    parser.add_argument('split',type=Path)
    parser.add_argument('--label-column',default='label')
    parser.add_argument('--sha256')
    parser.add_argument('--textile',action='store_true',help='Require canonical hash and all textile counts')
    parser.add_argument('--observations',type=Path)
    args=parser.parse_args()
    try:
        frame=load_split(args.split,label_column=args.label_column,expected_sha256=args.sha256,textile=args.textile)
        if args.observations:
            obs=pd.read_csv(args.observations,dtype=str,keep_default_na=False)
            validate_observations(obs,frame)
        print(json.dumps({'valid':True,'physical_samples':len(frame),'counts':validate_split(frame)},sort_keys=True))
    except (ValueError,OSError) as exc:
        parser.exit(2,'Split validation failed: '+str(exc)+'\n')

if __name__=='__main__':
    main()
