"""Exact paired McNemar tests, separately for each model seed."""
from scipy.stats import binomtest
from .bootstrap import aligned_predictions

def mcnemar_by_seed(frame, column_a='prediction_a', column_b='prediction_b'):
    results = []
    for part in aligned_predictions(frame,[column_a,column_b]):
        if part[[column_a,column_b]].isna().any().any():
            raise ValueError('Missing paired predictions')
        a = part[column_a].eq(part['label']); b = part[column_b].eq(part['label'])
        a_only = int((a & ~b).sum()); b_only = int((~a & b).sum()); n = a_only+b_only
        results.append({'seed':int(part.seed.iloc[0]), 'a_correct_b_wrong':a_only,
                        'a_wrong_b_correct':b_only,
                        'p_value':float(binomtest(a_only,n,p=0.5).pvalue) if n else 1.0})
    return results
