"""Classification metrics with an explicit, stable class order."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
from .calibration import normalize_probabilities, expected_calibration_error

def classification_metrics(y_true, y_pred, *, labels, probabilities=None, bins=10):
    labels = list(labels)
    y, pred = np.asarray(y_true), np.asarray(y_pred)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError('Provide at least two unique ordered labels')
    if y.ndim != 1 or pred.shape != y.shape or not len(y):
        raise ValueError('Expected equally sized nonempty label vectors')
    if not (set(y) | set(pred)).issubset(labels):
        raise ValueError('Unknown classification label')
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    result = {'accuracy': float(accuracy_score(y, pred)),
              'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
              'macro_f1': float(f1.mean()),
              'per_class': {str(c): {'precision': float(precision[i]), 'recall': float(recall[i]),
                                    'f1': float(f1[i]), 'support': int(support[i])} for i,c in enumerate(labels)},
              'labels': labels, 'confusion_matrix': confusion_matrix(y, pred, labels=labels).tolist()}
    if probabilities is not None:
        p = normalize_probabilities(probabilities)
        if len(p) != len(y) or p.shape[1] != len(labels):
            raise ValueError('Probability dimensions differ from label schema')
        if not np.array_equal(np.asarray(labels)[p.argmax(1)], pred):
            raise ValueError('Predictions disagree with probability argmax')
        result['ece'] = expected_calibration_error(p, y, labels=labels, bins=bins)
    return result
