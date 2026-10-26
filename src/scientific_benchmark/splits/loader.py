"""Read external memberships; never create or reshuffle them."""
from pathlib import Path
import hashlib
import pandas as pd
from .validation import validate_split, SplitIntegrityError

TEXTILE_SHA256 = 'a5726d3111941989407b0049e7ba67679e3b846c7e486544467fd926f99dbfc4'
TEXTILE_COUNTS = {'train': 313, 'validation': 67, 'test': 68}
TEXTILE_CLASS_COUNTS = {'CO': (129,28,28), 'LI': (30,6,6), 'NY': (14,3,3),
                        'PO': (32,7,7), 'SI': (62,13,14), 'VI': (19,4,4), 'WO': (27,6,6)}

def load_split(path, *, label_column='label', expected_sha256=None, textile=False, **validation):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = TEXTILE_SHA256 if textile else expected_sha256
    if expected and digest.lower() != expected.lower():
        raise SplitIntegrityError('Reference split SHA-256 mismatch')
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    if textile:
        label_column = 'class_code'
        validation.update(expected_classes=TEXTILE_CLASS_COUNTS,
                          expected_counts=TEXTILE_COUNTS,
                          expected_class_counts=TEXTILE_CLASS_COUNTS)
    if label_column != 'label':
        if 'label' in frame.columns:
            raise SplitIntegrityError('Ambiguous label columns')
        frame = frame.rename(columns={label_column: 'label'})
    validate_split(frame, **validation)
    return frame[['sample_id', 'label', 'split']].copy()
