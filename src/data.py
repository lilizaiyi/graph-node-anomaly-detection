"""Load and validate small undirected attributed graphs."""
import numpy as np


def validate(x, a, y):
    x, a, y = np.asarray(x, float), np.asarray(a, float), np.asarray(y).reshape(-1)
    if x.ndim != 2 or a.shape != (len(x), len(x)) or len(y) != len(x):
        raise ValueError('Expected x[N,F], a[N,N], y[N].')
    if not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError('Non-finite attributes or adjacency.')
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Evaluation labels must contain both 0=normal and 1=anomaly.')
    if (a < 0).any() or not np.allclose(a, a.T):
        raise ValueError('Supply a nonnegative undirected adjacency matrix explicitly.')
    a = (a > 0).astype(float)
    np.fill_diagonal(a, 0)
    return x, a, y.astype(int)


def load(path):
    with np.load(path, allow_pickle=False) as d:
        return validate(d['x'], d['a'], d['y'])


def standardize(x):
    # Transductive setting: all attributes available, labels never used.
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-8)


def synthetic(seed=2026, n=240, f=16):
    """Synthetic smoke data only: attribute swaps and planted dense groups."""
    rng = np.random.default_rng(seed)
    group = np.arange(n) % 3
    centers = rng.normal(0, 2, (3, f))
    x = centers[group] + rng.normal(0, .5, (n, f))
    p = np.where(group[:, None] == group[None, :], .10, .008)
    upper = np.triu(rng.random((n, n)) < p, 1)
    a = (upper | upper.T).astype(float)
    selected = rng.choice(n, 24, replace=False)
    context, structure = selected[:12], selected[12:]
    x[context] = centers[(group[context] + 1) % 3] + rng.normal(0, .5, (12, f))
    for chunk in np.split(structure, 2):
        a[np.ix_(chunk, chunk)] = 1
    np.fill_diagonal(a, 0)
    y = np.zeros(n, int)
    y[selected] = 1
    return validate(x, a, y)
