from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

import tempfile,hashlib
from scientific_benchmark.splits.loader import load_split
from scientific_benchmark.splits.validation import validate_split,SplitIntegrityError

class SplitTests(unittest.TestCase):
    def setUp(self):
        self.path=ROOT/'examples/data/synthetic_split.csv';self.frame=load_split(self.path)
    def test_correct_loading(self):
        self.assertEqual(validate_split(self.frame),{'train':12,'validation':4,'test':4})
        self.assertEqual(len(self.frame.sample_id.unique()),20)
    def test_hash(self):
        digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(len(load_split(self.path,expected_sha256=digest)),20)
        with self.assertRaisesRegex(SplitIntegrityError,'SHA-256'):
            load_split(self.path,expected_sha256='0'*64)
    def test_missing_columns(self):
        with self.assertRaisesRegex(SplitIntegrityError,'Missing'):
            validate_split(self.frame.drop(columns='label'))
    def test_bad_split(self):
        self.frame.loc[0,'split']='val'
        with self.assertRaisesRegex(SplitIntegrityError,'Malformed'):
            validate_split(self.frame)
    def test_unknown_class(self):
        with self.assertRaisesRegex(SplitIntegrityError,'Unexpected'):
            validate_split(self.frame,expected_classes=['other'])
    def test_same_partition_duplicate(self):
        with self.assertRaisesRegex(SplitIntegrityError,'Duplicate'):
            validate_split(pd.concat([self.frame,self.frame.iloc[[0]]]))
    def test_class_and_total_counts(self):
        with self.assertRaisesRegex(SplitIntegrityError,'count mismatch'):
            validate_split(self.frame,expected_counts={'train':99})
        with self.assertRaisesRegex(SplitIntegrityError,'Per-class'):
            validate_split(self.frame,expected_class_counts={'alpha':(9,2,2)})
    def test_bom_and_string_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'split.csv';p.write_text('sample_id,class_code,split\n001,alpha,train\n',encoding='utf-8-sig')
            self.assertEqual(load_split(p,label_column='class_code').sample_id.iloc[0],'001')
    def test_null_and_whitespace(self):
        for value in [None,'',' s ']:
            frame=self.frame.copy();frame.loc[0,'sample_id']=value
            with self.assertRaises(SplitIntegrityError):validate_split(frame)
