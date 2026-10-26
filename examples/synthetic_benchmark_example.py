from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

"""Run only the tiny artificial validation benchmark, with temporary outputs."""
import json,tempfile
from scientific_benchmark.benchmark import run_benchmark

if __name__=='__main__':
    import yaml
    root=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((root/'configs/spectral_benchmark_example.yaml').read_text())
    with tempfile.TemporaryDirectory(prefix='scientific-benchmark-') as temp:
        config.update(split_path=str(root/'examples/data/synthetic_split.csv'),
                      observations_path=str(root/'examples/data/synthetic_spectra.csv'),
                      output_dir=str(Path(temp)/'results'))
        path=Path(temp)/'config.yaml';path.write_text(yaml.safe_dump(config),encoding='utf-8')
        records=run_benchmark(path,kind='spectral')
        print(json.dumps({'data':'synthetic only','test_evaluated':False,'results':records},indent=2))
