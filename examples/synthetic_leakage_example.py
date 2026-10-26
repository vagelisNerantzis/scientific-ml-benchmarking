from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

"""Intentionally fails with a physical-sample leakage error (exit status 2)."""
from scientific_benchmark.splits.loader import load_split
from scientific_benchmark.splits.validation import SplitIntegrityError

if __name__=='__main__':
    try:
        load_split(Path(__file__).parent/'data/synthetic_leaked_split.csv')
    except SplitIntegrityError as exc:
        print('EXPECTED FAILURE: '+str(exc),file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit('ERROR: leaked membership unexpectedly accepted')
