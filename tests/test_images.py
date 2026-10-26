from pathlib import Path
import sys,unittest,tempfile,importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
from scientific_benchmark.preprocessing.images import evaluation_crops,normalized_array

class ImageContractTests(unittest.TestCase):
    def test_rgb_crop_and_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'image.png';Image.new('RGB',(100,80),(255,0,0)).save(p)
            crops=evaluation_crops(p,size=16)
            self.assertEqual(len(crops),9);self.assertEqual(crops[0].size,(16,16))
            tensor=normalized_array(crops[0],mean=[0,0,0],std=[1,1,1])
            self.assertEqual(tensor.shape,(3,16,16));self.assertEqual(tensor[0,0,0],1)
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('timm'),'optional image stack unavailable')
    def test_frozen_extraction_without_weights_or_network(self):
        import torch
        from scientific_benchmark.benchmark import image_features
        # Real torch tensor operations with a tiny deterministic fake encoder.
        # This tests extraction/batching; it is not a real-model accuracy claim.
        class Encoder(torch.nn.Module):
            def forward(self,x):return x.mean(dim=(2,3))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'image.png';Image.new('RGB',(64,48),(120,50,200)).save(p)
            cfg={'_base':Path(temp),'pretrained':True,'encoder':'synthetic_test_encoder','device':'cpu',
                 'image_size':16,'image_batch_size':2,'eval_crops':9,'crop_fraction':0.45}
            data={'input_size':(3,16,16),'mean':(0,0,0),'std':(1,1,1),'interpolation':'bicubic','crop_pct':1.0}
            with patch('timm.create_model',return_value=Encoder()),patch('timm.data.resolve_model_data_config',return_value=data):
                x=image_features(cfg,pd.DataFrame({'path':[str(p)]}))
            self.assertEqual(x.shape,(1,3));self.assertTrue(np.isfinite(x).all())
