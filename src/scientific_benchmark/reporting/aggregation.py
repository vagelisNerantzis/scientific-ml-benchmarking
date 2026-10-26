"""Identity-preserving probability and repeated-seed aggregation."""
import numpy as np
import pandas as pd
from ..splits.validation import validate_split
from ..metrics.calibration import normalize_probabilities

def aggregate_observations(frame, *, labels, level='sample'):
    validate_split(frame[['sample_id','label','split']].drop_duplicates(), expected_classes=labels)
    identity = [c for c in ['sample_id','label','split','modality','observation_id',
                            'source_observation_id','image_id'] if c in frame]
    validate_split(frame[identity].drop_duplicates(),observations=True,expected_classes=labels)
    contexts = [c for c in ['model','seed'] if c in frame]
    parts = (part for _,part in frame.groupby(contexts)) if contexts else [frame]
    for part in parts:
        validate_split(part, observations=True, expected_classes=labels)
    columns = [f'prob_{c}' for c in labels]
    if not set(columns).issubset(frame.columns):
        raise ValueError('Missing class probability columns')
    frame = frame.copy()
    frame[columns] = normalize_probabilities(frame[columns].to_numpy())
    if level not in ['sample', 'image']:
        raise ValueError('level must be sample or image')
    groups = [c for c in ['model', 'seed'] if c in frame] + ['sample_id', 'label', 'split', 'modality']
    if level == 'image':
        if 'image_id' not in frame:
            raise ValueError('Image-level aggregation requires image_id')
        groups += ['image_id']
    result = frame.groupby(groups, as_index=False, sort=True)[columns].mean()
    result[columns] = normalize_probabilities(result[columns].to_numpy())
    result['prediction'] = np.asarray(labels)[result[columns].to_numpy().argmax(1)]
    return result

def aggregate_seeds(frame, *, metrics=('accuracy', 'balanced_accuracy', 'macro_f1', 'ece')):
    required = {'model','modality','split','seed','n_samples'} | set(metrics)
    if not required.issubset(frame):
        raise ValueError('Missing repeated-seed result fields')
    if frame.duplicated(['model','modality','split','seed']).any():
        raise ValueError('Duplicate seed result')
    for metric in metrics:
        if not np.isfinite(frame[metric].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite seed metric')
    groups = ['model', 'modality', 'split']
    if frame.groupby(groups)['n_samples'].nunique().gt(1).any():
        raise ValueError('Physical-sample counts differ across seeds')
    summary = frame.groupby(groups)[list(metrics)].agg(['mean','std','count'])
    summary.columns = [f'{a}_{b}' for a,b in summary.columns]
    # Sample SD (ddof=1); one seed has undefined SD, retained as NaN.
    return summary.reset_index()
