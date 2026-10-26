"""Explicit wavelength and units contracts; no implicit resampling or clipping."""
import numpy as np
from sklearn.preprocessing import StandardScaler

def validate_wavelengths(wavelengths, values, *, reference=None):
    w,x = np.asarray(wavelengths,dtype=float),np.asarray(values,dtype=float)
    if w.ndim != 1 or len(w)<2 or not np.isfinite(w).all() or not (np.diff(w)>0).all():
        raise ValueError('Wavelengths must be finite and strictly increasing')
    if x.ndim not in [1,2] or x.shape[-1] != len(w) or not np.isfinite(x).all():
        raise ValueError('Spectral values must be finite and aligned to wavelengths')
    if reference is not None:
        ref = np.asarray(reference,dtype=float)
        if ref.shape != w.shape or not np.allclose(ref,w,atol=1e-8,rtol=0):
            raise ValueError('Wavelength schema mismatch; no automatic interpolation')
    return w,x

def transmittance_to_absorbance(values, *, units='percent'):
    if units not in ['percent','fraction']:
        raise ValueError('Transmittance units must be percent or fraction')
    t = np.asarray(values,dtype=float)/(100 if units=='percent' else 1)
    if not np.isfinite(t).all() or (t<=0).any() or (t>1).any():
        raise ValueError('Transmittance must be positive and no greater than 100%')
    return -np.log10(t)

class TrainOnlyScaler:
    """Explicit guard against accidental fit on validation/test observations."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, values, splits):
        x = np.asarray(values,dtype=float)
        if self.fitted:
            raise ValueError('Scaler already fitted; construct a fresh training scaler')
        if x.ndim!=2 or len(x)!=len(splits) or not len(x) or set(splits)!={'train'} or not np.isfinite(x).all():
            raise ValueError('Scaler fit requires finite train-only rows')
        self.scaler.fit(x); self.fitted=True
        return self

    def transform(self, values):
        x = np.asarray(values,dtype=float)
        if not self.fitted or not np.isfinite(x).all():
            raise ValueError('Fit training scaler first and supply finite values')
        return self.scaler.transform(x)
