from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

import tempfile,json,yaml
from scientific_benchmark.reporting.schemas import validate_result
from scientific_benchmark.benchmark import run_benchmark,load_inputs,lock_protocol
from scientific_benchmark.utils.config import load_config,resolve_path
from scientific_benchmark.energy.gpu import GPUEnergyMonitor
from scientific_benchmark.runtime.timing import Timer

class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.record=dict(schema_version=1,model='m',modality='x',split='validation',seed=42,n_samples=10,
                         accuracy=0.5,balanced_accuracy=0.5,macro_f1=0.5,ece=0.2,evaluation_unit='physical_sample')
    def test_valid_result(self):
        self.assertEqual(validate_result(self.record),self.record)
    def test_reject_invalid_result(self):
        for changes in [dict(accuracy=2),dict(ece=float('nan')),dict(evaluation_unit='crop'),
                        dict(n_samples=0),dict(seed=2.5),dict(sample_id='private'),dict(gpu_energy_wh=-1)]:
            with self.assertRaises(ValueError):validate_result(dict(self.record,**changes))
        self.record.pop('model')
        with self.assertRaisesRegex(ValueError,'Missing'):validate_result(self.record)
    def config(self,temp):
        cfg=yaml.safe_load((ROOT/'configs/spectral_benchmark_example.yaml').read_text())
        cfg.update(split_path=str(ROOT/'examples/data/synthetic_split.csv'),
                   observations_path=str(ROOT/'examples/data/synthetic_spectra.csv'),output_dir=str(Path(temp)/'results'))
        p=Path(temp)/'config.yaml';p.write_text(yaml.safe_dump(cfg));return p
    def test_synthetic_validation_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            p=self.config(temp);records=run_benchmark(p,kind='spectral')
            self.assertEqual(len(records),3)
            self.assertEqual({x['split'] for x in records},{'validation'})
            self.assertEqual({x['n_samples'] for x in records},{4})
            for row in records:validate_result(row)
            manifest=json.loads((Path(temp)/'results/manifest.json').read_text())
            self.assertFalse(manifest['test_evaluated'])
            self.assertNotIn('sample_id',(Path(temp)/'results/results.json').read_text())
    def test_test_requires_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            p=self.config(temp)
            with self.assertRaisesRegex(ValueError,'externally frozen'):
                run_benchmark(p,kind='spectral',evaluate_test=True)
            lock=Path(temp)/'lock.json';lock.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Protocol differs'):
                run_benchmark(p,kind='spectral',evaluate_test=True,protocol_path=lock)
    def test_locked_synthetic_test_only(self):
        with tempfile.TemporaryDirectory() as temp:
            p=self.config(temp);lock=Path(temp)/'lock.json'
            lock.write_text(json.dumps(lock_protocol(p,load_config(p),'spectral')))
            records=run_benchmark(p,kind='spectral',evaluate_test=True,protocol_path=lock)
            self.assertEqual({x['split'] for x in records},{'validation','test'})
            self.assertEqual(len(records),6)
    def test_default_excludes_test_values(self):
        with tempfile.TemporaryDirectory() as temp:
            p=self.config(temp);cfg=load_config(p)
            _,frame=load_inputs(cfg)
            self.assertNotIn('test',set(frame.split))
    def test_relative_config_and_missing_env(self):
        cfg=load_config(ROOT/'configs/spectral_benchmark_example.yaml')
        self.assertTrue(resolve_path(cfg,'split_path').exists())
        cfg['split_path']='${SCIENTIFIC_BENCHMARK_UNSET_FOR_TEST}'
        with self.assertRaisesRegex(ValueError,'Unresolved'):resolve_path(cfg,'split_path')
    def test_optional_telemetry_disabled(self):
        with Timer() as timer, GPUEnergyMonitor() as energy:
            sum(range(5))
        self.assertGreaterEqual(timer.seconds,0);self.assertIsNone(timer.peak_gpu_memory_bytes)
        self.assertFalse(energy.available);self.assertIsNone(energy.energy_wh)

    def test_existing_output_fails_before_feature_access(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            p=self.config(temp);output=Path(temp)/'results';output.mkdir()
            (output/'results.json').write_text('[]')
            with patch('scientific_benchmark.benchmark.spectral_features',side_effect=AssertionError('features must not be read')):
                with self.assertRaisesRegex(ValueError,'already exist'):
                    run_benchmark(p,kind='spectral')
    def test_held_out_spectral_values_not_converted(self):
        # Invalid values in synthetic test rows must not enter default feature preparation.
        with tempfile.TemporaryDirectory() as temp:
            p=self.config(temp);config=yaml.safe_load(p.read_text())
            obs=pd.read_csv(config['observations_path'])
            obs['w1000']=obs['w1000'].astype(object)
            obs.loc[obs.split=='test','w1000']='not-a-number'
            csv=Path(temp)/'obs.csv';obs.to_csv(csv,index=False)
            config['observations_path']=str(csv);p.write_text(yaml.safe_dump(config))
            records=run_benchmark(p,kind='spectral')
            self.assertEqual(len(records),3)
    def test_image_head_uses_physical_samples(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            p=self.config(temp);cfg=yaml.safe_load(p.read_text());cfg['model_name']='mock_frozen_image'
            p.write_text(yaml.safe_dump(cfg))
            def features(config,frame):
                return np.array([[1.0 if c=='alpha' else -1.0,0.5] for c in frame.label])
            with patch('scientific_benchmark.benchmark.image_features',side_effect=features):
                records=run_benchmark(p,kind='image')
            self.assertEqual({row['n_samples'] for row in records},{4})
