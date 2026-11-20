"""Educational GCN dual reconstruction baseline; NOT DECLARE/DOMINANT reproduction."""
import numpy as np


def train(x, a, seed, epochs=100, hidden=32, lr=.001, weight=.5, mode='dual'):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError('Install requirements-deep.txt to run the GCN baseline.') from exc
    if mode not in ('dual', 'attribute', 'structure') or not 0 <= weight <= 1:
        raise ValueError('Invalid reconstruction mode or weight.')
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    xt, target = torch.tensor(x, dtype=torch.float32), torch.tensor(a, dtype=torch.float32)
    adj = target + torch.eye(len(a))
    d = adj.sum(1).pow(-.5)
    norm = d[:, None] * adj * d[None, :]
    mask = 1 - torch.eye(len(a))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Linear(x.shape[1], hidden)
            self.enc2 = nn.Linear(hidden, hidden)
            self.attr = nn.Linear(hidden, x.shape[1])
            self.struct = nn.Linear(hidden, hidden)

        def forward(self):
            h = torch.relu(self.enc1(norm @ xt))
            z = self.enc2(norm @ h)
            reconstructed_x = self.attr(norm @ z)
            h_s = self.struct(norm @ z)
            reconstructed_a = torch.sigmoid(h_s @ h_s.T / hidden**.5)
            return ((reconstructed_x-xt)**2).mean(1), (((reconstructed_a-target)**2)*mask).sum(1)/(len(a)-1)

    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    effective_weight = {'attribute': 1., 'structure': 0., 'dual': weight}[mode]
    for epoch in range(epochs):
        opt.zero_grad()
        ex, ea = model()
        loss = effective_weight*ex.mean() + (1-effective_weight)*ea.mean()
        loss.backward()
        opt.step()
        history.append({'epoch': epoch+1, 'loss': float(loss.detach())})
    with torch.no_grad():
        ex, ea = model()
        score = effective_weight*ex + (1-effective_weight)*ea
    return score.numpy().astype(float), history
