from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

from scientific_benchmark.metrics.classification import classification_metrics
from scientific_benchmark.reporting.aggregation import aggregate_observations,aggregate_seeds
from scientific_benchmark.preprocessing.images import deterministic_grid_boxes
from scientific_benchmark.preprocessing.spectra import validate_wavelengths,transmittance_to_absorbance

class MetricTests(unittest.TestCase):
    def test_accuracy_balanced_macro(self):
        result=classification_metrics(['a','a','a','b'],['a','a','b','b'],labels=['a','b'])
        self.assertAlmostEqual(result['accuracy'],0.75)
        self.assertAlmostEqual(result['balanced_accuracy'],5/6)
        self.assertAlmostEqual(result['macro_f1'],(0.8+2/3)/2)
        self.assertEqual(result['confusion_matrix'],[[2,1],[0,1]])
        self.assertAlmostEqual(result['per_class']['a']['precision'],1)
        self.assertAlmostEqual(result['per_class']['a']['recall'],2/3)
    def test_sample_probability_aggregation(self):
        frame=pd.DataFrame({'sample_id':['s','s','t'],'observation_id':['o1','o2','o3'],
          'modality':['image']*3,'label':['a','a','b'],'split':['validation']*3,
          'prob_a':[0.9,0.3,0.2],'prob_b':[0.1,0.7,0.8]})
        result=aggregate_observations(frame,labels=['a','b'])
        self.assertEqual(len(result),2);self.assertAlmostEqual(result.prob_a.iloc[0],0.6)
        self.assertEqual(result.prediction.tolist(),['a','b'])
        combined=pd.concat([frame.assign(seed=7),frame.assign(seed=42)])
        self.assertEqual(len(aggregate_observations(combined,labels=['a','b'])),4)
    def test_image_then_sample_weighting(self):
        frame=pd.DataFrame({'sample_id':['s']*3,'observation_id':['c1','c2','c3'],
           'image_id':['i1','i1','i2'],'modality':['image']*3,'label':['a']*3,'split':['validation']*3,
           'prob_a':[0.9,0.9,0.1],'prob_b':[0.1,0.1,0.9]})
        image=aggregate_observations(frame,labels=['a','b'],level='image')
        image['observation_id']=image.image_id
        result=aggregate_observations(image,labels=['a','b'])
        self.assertAlmostEqual(result.prob_a.iloc[0],0.5)
    def test_seed_mean_and_sample_std(self):
        frame=pd.DataFrame([dict(model='m',modality='x',split='validation',seed=s,n_samples=10,
                               accuracy=v,balanced_accuracy=v,macro_f1=v,ece=0.1) for s,v in [(7,0.6),(42,0.8)]])
        result=aggregate_seeds(frame)
        self.assertAlmostEqual(result.accuracy_mean.iloc[0],0.7)
        self.assertAlmostEqual(result.accuracy_std.iloc[0],np.sqrt(0.02))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            aggregate_seeds(pd.concat([frame,frame.iloc[[0]]]))
    def test_crop_geometry(self):
        boxes=deterministic_grid_boxes(1000,800,0.45,9)
        self.assertEqual(boxes[0],(0,0,360));self.assertEqual(boxes[4],(320,220,360))
        self.assertEqual(boxes[-1],(640,440,360))
        with self.assertRaises(ValueError):deterministic_grid_boxes(100,100,count=10)
    def test_spectral_contract(self):
        np.testing.assert_allclose(transmittance_to_absorbance([100,10,1]),[0,1,2])
        with self.assertRaises(ValueError):transmittance_to_absorbance([0])
        with self.assertRaises(ValueError):validate_wavelengths([2,1],[1,2])
        with self.assertRaisesRegex(ValueError,'schema mismatch'):
            validate_wavelengths([1,2],[3,4],reference=[1,3])
