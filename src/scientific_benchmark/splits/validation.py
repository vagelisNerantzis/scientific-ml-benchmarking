"""Validate immutable physical-entity membership without assigning any splits."""
import pandas as pd

SPLITS = ('train', 'validation', 'test')

class SplitIntegrityError(ValueError):
    """Invalid identity, label, membership, or observation lineage."""

def validate_split(frame, *, expected_classes=None, expected_counts=None,
                   expected_class_counts=None, observations=False):
    required = {'sample_id', 'label', 'split'}
    if observations:
        required |= {'observation_id', 'modality'}
    missing = required - set(frame.columns)
    if missing:
        raise SplitIntegrityError(f'Missing required columns: {sorted(missing)}')
    if frame.empty:
        raise SplitIntegrityError('Empty split table')
    for column in required:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq('').any():
            raise SplitIntegrityError(f'Null or empty {column}')
        if not values.astype(str).eq(values.astype(str).str.strip()).all():
            raise SplitIntegrityError(f'Whitespace in {column}; normalize explicitly upstream')
    bad = set(frame['split']) - set(SPLITS)
    if bad:
        raise SplitIntegrityError(f'Malformed split names: {sorted(bad)}')
    for column, message in [('split', 'Physical-sample leakage across splits'),
                            ('label', 'Inconsistent labels for physical sample')]:
        invalid = frame.groupby('sample_id')[column].nunique().gt(1)
        if invalid.any():
            raise SplitIntegrityError(message)
    if not observations and frame['sample_id'].duplicated().any():
        raise SplitIntegrityError('Duplicate physical sample IDs in membership table')
    if expected_classes is not None and not set(frame['label']).issubset(set(expected_classes)):
        raise SplitIntegrityError('Unexpected class labels')
    if observations:
        if frame.duplicated(['modality', 'observation_id']).any():
            raise SplitIntegrityError('Duplicate observation identity within modality')
        # A copied crop must retain its physical parent. Global origin IDs may
        # not be reassigned to another sample or partition.
        for column in ['observation_id', 'source_observation_id', 'image_id']:
            if column in frame:
                if frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
                    raise SplitIntegrityError(f'Empty lineage: {column}')
                grouped = frame.groupby(['modality', column])
                if (grouped['sample_id'].nunique().gt(1).any()
                        or grouped['split'].nunique().gt(1).any()):
                    raise SplitIntegrityError(f'Derived-observation leakage through {column}')
    unique = frame.drop_duplicates('sample_id')
    counts = unique['split'].value_counts().to_dict()
    if expected_counts is not None and any(counts.get(k, 0) != v for k, v in expected_counts.items()):
        raise SplitIntegrityError('Physical-sample split count mismatch')
    if expected_class_counts is not None:
        actual = unique.groupby(['label', 'split']).size().to_dict()
        for label, values in expected_class_counts.items():
            if tuple(actual.get((label, s), 0) for s in SPLITS) != tuple(values):
                raise SplitIntegrityError('Per-class physical-sample count mismatch')
    return counts

def validate_observations(frame, membership, *, require_all_samples=False):
    validate_split(membership)
    validate_split(frame, observations=True, expected_classes=membership['label'].unique())
    reference = membership.set_index('sample_id')
    if not set(frame['sample_id']).issubset(reference.index):
        raise SplitIntegrityError('Observation has unknown physical sample ID')
    for column in ['label', 'split']:
        expected = frame['sample_id'].map(reference[column])
        if not frame[column].eq(expected).all():
            raise SplitIntegrityError(f'Observation {column} differs from immutable membership')
    if require_all_samples and set(frame['sample_id']) != set(reference.index):
        raise SplitIntegrityError('Missing observations for physical samples')
