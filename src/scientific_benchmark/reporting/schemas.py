"""Validated aggregate-only result records, suitable for manual public review."""
import math
from numbers import Integral, Real
from ..splits.validation import SPLITS

REQUIRED = {'schema_version','model','modality','split','seed','n_samples',
            'accuracy','balanced_accuracy','macro_f1','ece','evaluation_unit'}

def validate_result(record):
    if REQUIRED-set(record):
        raise ValueError('Missing required result fields: '+str(sorted(REQUIRED-set(record))))
    if record['schema_version']!=1 or record['evaluation_unit']!='physical_sample':
        raise ValueError('Unsupported schema version or evaluation unit')
    if record['split'] not in SPLITS:
        raise ValueError('Invalid result partition')
    for key in ['model','modality']:
        if not isinstance(record[key],str) or not record[key].strip():
            raise ValueError('Missing result identifier: '+key)
    for key in ['seed','n_samples']:
        if isinstance(record[key],bool) or not isinstance(record[key],Integral):
            raise ValueError('Integer required: '+key)
    if record['n_samples']<1:
        raise ValueError('n_samples must be positive')
    for metric in ['accuracy','balanced_accuracy','macro_f1','ece']:
        value=record[metric]
        if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid bounded metric: '+metric)
    for key in ['wall_seconds','peak_gpu_memory_bytes','gpu_energy_wh']:
        v=record.get(key)
        if v is not None and (isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) or v<0):
            raise ValueError('Invalid telemetry: '+key)
    if {'sample_id','observation_id','filepath','path'} & set(record):
        raise ValueError('Aggregate schema forbids sample identifiers and data paths')
    return record
