from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse,json
from scientific_benchmark.benchmark import run_benchmark,lock_protocol
from scientific_benchmark.utils.config import load_config

def main():
    parser=argparse.ArgumentParser(description='Validation-first spectral baseline')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--evaluate-test',action='store_true')
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--print-protocol',action='store_true',help='Print a template; does not establish historical pre-registration')
    args=parser.parse_args()
    if args.print_protocol:
        print(json.dumps(lock_protocol(args.config,load_config(args.config),'spectral'),indent=2));return
    try:
        result=run_benchmark(args.config,kind='spectral',evaluate_test=args.evaluate_test,protocol_path=args.protocol)
        print(json.dumps({'completed':True,'aggregate_records':len(result),'test_evaluated':args.evaluate_test}))
    except (ValueError,OSError) as exc:
        parser.exit(2,str(exc)+'\n')

if __name__=='__main__':
    main()
