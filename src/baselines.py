import numpy as np


def scale(s):
    return (s - s.min()) / max(float(np.ptp(s)), 1e-8)


def scores(x, a, rank=4, weight=.5):
    if not 0 <= weight <= 1:
        raise ValueError('fusion_weight must be within [0,1].')
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[:min(rank, len(vt))]
    attr = np.mean((x - x @ basis.T @ basis)**2, axis=1)
    degree = a.sum(1)
    neighbor = a @ x / np.maximum(degree[:, None], 1)
    residual = np.mean((x-neighbor)**2, axis=1)
    residual[degree == 0] = 0  # No evidence of contextual discrepancy for isolated nodes.
    return {'pca': attr, 'neighbor_residual': residual,
            'fusion': weight * scale(attr) + (1-weight) * scale(residual)}
