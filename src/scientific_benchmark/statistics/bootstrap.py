"""Stratified physical-sample bootstrap; seeds are never resampled as entities."""
import numpy as np
import pandas as pd
from ..metrics.classification import classification_metrics
from ..splits.validation import validate_split

METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1')

def aligned_predictions(frame, prediction_columns):
    required = {'sample_id','label','split','seed'} | set(prediction_columns)
    if not required.issubset(frame) or frame.empty:
        raise ValueError('Missing or empty physical-sample prediction table')
    validate_split(frame.drop_duplicates(['sample_id','label','split']))
    if frame['seed'].isna().any() or frame.duplicated(['sample_id','seed']).any():
        raise ValueError('Expected one row per physical sample and seed')
    if frame['split'].nunique() != 1:
        raise ValueError('Bootstrap must evaluate exactly one partition')
    seeds = sorted(frame['seed'].unique())
    frames = [frame.loc[frame.seed == s].sort_values('sample_id').reset_index(drop=True) for s in seeds]
    for part in frames:
        if not part[['sample_id','label','split']].equals(frames[0][['sample_id','label','split']]):
            raise ValueError('Physical sample membership differs across seeds')
    return frames

def bootstrap_metrics(frame, *, labels, prediction_columns=('prediction',), metric='macro_f1',
                      repetitions=1000, seed=42, confidence=0.95):
    if metric not in METRICS or not isinstance(repetitions,int) or repetitions < 1 or not 0 < confidence < 1:
        raise ValueError('Invalid bootstrap metric, repetitions, or confidence')
    columns = list(prediction_columns)
    if len(columns) not in [1,2] or len(set(columns)) != len(columns):
        raise ValueError('Provide one method or two distinct paired methods')
    parts = aligned_predictions(frame, columns)
    y = parts[0]['label'].to_numpy()
    for part in parts:
        for c in columns:
            classification_metrics(y,part[c],labels=labels)
    strata = [np.flatnonzero(y == c) for c in labels if np.any(y == c)]
    def evaluate(idx):
        scores = [np.mean([classification_metrics(y[idx],p[c].to_numpy()[idx],labels=labels)[metric]
                           for p in parts]) for c in columns]
        return scores + ([scores[1]-scores[0]] if len(scores) == 2 else [])
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions,len(columns)+(len(columns)==2)))
    for b in range(repetitions):
        idx = np.concatenate([rng.choice(s,size=len(s),replace=True) for s in strata])
        draws[b] = evaluate(idx)
    alpha = (1-confidence)/2
    names = columns + (['difference_b_minus_a'] if len(columns)==2 else [])
    return {'metric':metric, 'methods':names, 'n_physical_samples':len(y), 'n_seeds':len(parts),
            'point_estimates':evaluate(np.arange(len(y))), 'confidence':confidence,
            'ci_low':np.quantile(draws,alpha,axis=0).tolist(),
            'ci_high':np.quantile(draws,1-alpha,axis=0).tolist(),
            'repetitions':repetitions, 'random_seed':seed, 'replicates':draws}
