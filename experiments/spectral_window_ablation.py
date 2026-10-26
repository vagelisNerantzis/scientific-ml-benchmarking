from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

"""Exploratory validation-only window baseline. Windows must be predeclared.

No automatic wavelength selection and no test evaluation are provided here.
"""
from scientific_benchmark.benchmark import run_benchmark
from scientific_benchmark.utils.config import load_config

def evaluate_window(config_path):
    config=load_config(config_path)
    if 'window_nm' not in config:
        raise ValueError('Declare window_nm: [lower, upper] in config')
    return run_benchmark(config_path,kind='spectral')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    evaluate_window(p.parse_args().config)
