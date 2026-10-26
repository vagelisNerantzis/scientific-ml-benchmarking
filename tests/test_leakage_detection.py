from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

import subprocess
from scientific_benchmark.splits.loader import load_split
from scientific_benchmark.splits.validation import validate_split,validate_observations,SplitIntegrityError
from scientific_benchmark.preprocessing.spectra import TrainOnlyScaler

class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.membership=load_split(ROOT/'examples/data/synthetic_split.csv')
        self.obs=pd.read_csv(ROOT/'examples/data/synthetic_spectra.csv')
    def test_sample_leakage(self):
        with self.assertRaisesRegex(SplitIntegrityError,'leakage across splits'):
            load_split(ROOT/'examples/data/synthetic_leaked_split.csv')
    def test_inconsistent_label(self):
        extra=self.obs.iloc[[0]].copy();extra['label']='beta';extra['observation_id']='new'
        with self.assertRaisesRegex(SplitIntegrityError,'Inconsistent labels'):
            validate_split(pd.concat([self.obs,extra]),observations=True)
    def test_parent_lineage_overlap(self):
        self.obs['source_observation_id']=self.obs['observation_id']
        self.obs.loc[self.obs.split=='test','source_observation_id']=self.obs.source_observation_id.iloc[0]
        with self.assertRaisesRegex(SplitIntegrityError,'Derived-observation'):
            validate_observations(self.obs,self.membership)
    def test_membership_disagreement(self):
        self.obs.loc[self.obs.sample_id==self.obs.sample_id.iloc[0],'split']='test'
        with self.assertRaisesRegex(SplitIntegrityError,'immutable membership'):
            validate_observations(self.obs,self.membership)
    def test_observation_overlap(self):
        self.obs.loc[self.obs.index[-1],'observation_id']=self.obs.observation_id.iloc[0]
        with self.assertRaisesRegex(SplitIntegrityError,'Duplicate observation'):
            validate_observations(self.obs,self.membership)
    def test_unknown_sample(self):
        self.obs.loc[0,'sample_id']='unknown'
        with self.assertRaisesRegex(SplitIntegrityError,'unknown physical'):
            validate_observations(self.obs,self.membership)
    def test_preprocessing_fit_rejects_validation(self):
        with self.assertRaisesRegex(ValueError,'train-only'):
            TrainOnlyScaler().fit([[1],[2]],['train','validation'])
    def test_intentionally_leaked_example_exits(self):
        result=subprocess.run([sys.executable,'-B',str(ROOT/'examples/synthetic_leakage_example.py')],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('leakage across splits',result.stderr)
    def test_cli_valid_and_leaked(self):
        script=ROOT/'scripts/validate_split.py'
        good=subprocess.run([sys.executable,'-B',str(script),str(ROOT/'examples/data/synthetic_split.csv')],capture_output=True,text=True)
        bad=subprocess.run([sys.executable,'-B',str(script),str(ROOT/'examples/data/synthetic_leaked_split.csv')],capture_output=True,text=True)
        self.assertEqual(good.returncode,0);self.assertEqual(bad.returncode,2)
        self.assertIn('leakage across splits',bad.stderr)
