from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

"""Training-reference wavelength diagnostics, explicitly exploratory."""
import numpy as np
from scientific_benchmark.preprocessing.spectra import validate_wavelengths

def standardized_deviation(train_values, train_splits, values, wavelengths):
    _,train=validate_wavelengths(wavelengths,train_values)
    _,x=validate_wavelengths(wavelengths,values)
    if train.ndim!=2 or len(train)<2 or len(train_splits)!=len(train) or set(train_splits)!={'train'}:
        raise ValueError('At least two train-only reference spectra required')
    std=train.std(axis=0,ddof=1)
    # Undefined zero-variance deviations are NaN rather than arbitrary epsilon scores.
    return (x-train.mean(axis=0))/np.where(std>0,std,np.nan)
