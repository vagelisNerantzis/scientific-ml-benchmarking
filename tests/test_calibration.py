from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

from scientific_benchmark.metrics.calibration import expected_calibration_error,normalize_probabilities

class CalibrationTests(unittest.TestCase):
    def test_hand_calculated_ece(self):
        self.assertAlmostEqual(expected_calibration_error([[0.8,0.2],[0.4,0.6]],['a','a'],labels=['a','b']),0.4)
    def test_perfect_calibration(self):
        self.assertEqual(expected_calibration_error([[1,0],[0,1]],['a','b'],labels=['a','b']),0)
    def test_boundary_bins(self):
        self.assertAlmostEqual(expected_calibration_error([[0.5,0.5],[1,0]],['a','b'],labels=['a','b'],bins=2),0.75)
    def test_invalid_probabilities(self):
        for x in [[[0,0]],[[float('nan'),1]],[[2,-1]],[[0.2,0.2]]]:
            with self.assertRaises(ValueError):normalize_probabilities(x)
    def test_invalid_bins_or_labels(self):
        with self.assertRaises(ValueError):expected_calibration_error([[1,0]],['a'],labels=['a','b'],bins=0)
        with self.assertRaises(ValueError):expected_calibration_error([[1,0]],['z'],labels=['a','b'])
