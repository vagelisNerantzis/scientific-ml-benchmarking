from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

from scientific_benchmark.statistics.bootstrap import bootstrap_metrics
from scientific_benchmark.statistics.paired import mcnemar_by_seed

class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame([dict(sample_id=f's{i}',label=c,split='test',seed=seed,
                                     prediction_a=a,prediction_b=b)
          for seed in [7,42] for i,(c,a,b) in enumerate([
            ('a','a','a'),('a','b','a'),('b','b','b'),('b','a','b')])])
    def test_deterministic_and_dimensions(self):
        a=bootstrap_metrics(self.frame,labels=['a','b'],prediction_columns=['prediction_a','prediction_b'],repetitions=20,seed=3)
        b=bootstrap_metrics(self.frame,labels=['a','b'],prediction_columns=['prediction_a','prediction_b'],repetitions=20,seed=3)
        np.testing.assert_array_equal(a['replicates'],b['replicates'])
        self.assertEqual(a['replicates'].shape,(20,3));self.assertEqual(a['n_physical_samples'],4)
        self.assertEqual(a['n_seeds'],2)
        np.testing.assert_allclose(a['replicates'][:,2],a['replicates'][:,1]-a['replicates'][:,0])
    def test_seeds_attached_and_stratified(self):
        # Perfect predictions in one seed and fully wrong in the other give mean
        # accuracy exactly 0.5 in every replicate when the sample draw is shared.
        frame=self.frame.copy();frame['prediction_a']=np.where(frame.seed==7,frame.label,np.where(frame.label=='a','b','a'))
        result=bootstrap_metrics(frame,labels=['a','b'],prediction_columns=['prediction_a'],metric='accuracy',repetitions=20)
        np.testing.assert_allclose(result['replicates'],0.5)
        # Majority-only classifier stays exactly 0.5 when true-class counts are preserved.
        frame['prediction_a']='a'
        result=bootstrap_metrics(frame,labels=['a','b'],prediction_columns=['prediction_a'],metric='accuracy',repetitions=20)
        np.testing.assert_allclose(result['replicates'],0.5)
    def test_mismatched_seed_membership(self):
        with self.assertRaisesRegex(ValueError,'membership differs'):
            bootstrap_metrics(self.frame.iloc[:-1],labels=['a','b'],prediction_columns=['prediction_a'],repetitions=2)
    def test_rejects_crop_duplicates(self):
        with self.assertRaisesRegex(ValueError,'one row'):
            bootstrap_metrics(pd.concat([self.frame,self.frame.iloc[[0]]]),labels=['a','b'],prediction_columns=['prediction_a'],repetitions=2)
    def test_mcnemar_per_seed(self):
        results=mcnemar_by_seed(self.frame)
        self.assertEqual(len(results),2);self.assertEqual(results[0]['a_wrong_b_correct'],2)
        self.assertEqual(results[0]['p_value'],0.5)
    def test_mixed_partitions(self):
        self.frame.loc[self.frame.sample_id=='s0','split']='validation'
        with self.assertRaisesRegex(ValueError,'exactly one partition'):
            bootstrap_metrics(self.frame,labels=['a','b'],prediction_columns=['prediction_a'],repetitions=2)
