import numpy as np


def evaluate(y, score):
    """Threshold-grouped AUROC and non-interpolated AP; ties handled together."""
    y, score = np.asarray(y), np.asarray(score)
    if y.ndim != 1 or score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError('Invalid score vector.')
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Both binary classes required.')
    order = np.argsort(-score, kind='stable')
    labels, sorted_score = y[order], score[order]
    ends = np.r_[np.where(np.diff(sorted_score) != 0)[0], len(y)-1]
    tp = np.r_[0, np.cumsum(labels)[ends]].astype(float)
    fp = np.r_[0, (ends + 1) - np.cumsum(labels)[ends]].astype(float)
    tpr, fpr = tp / y.sum(), fp / (len(y) - y.sum())
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    precision = tp[1:] / (tp[1:] + fp[1:])
    ap = np.sum(np.diff(tpr) * precision)
    return {'roc_auc': float(auc), 'ap': float(ap)}
