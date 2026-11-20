"""Matched MLP/GCN reconstruction models for small attributed graphs."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normalized_adjacency(adjacency):
    """Sparse D^(-1/2)(A+I)D^(-1/2), with binary undirected A."""
    with_loops = torch.as_tensor(adjacency, dtype=torch.float32) + torch.eye(len(adjacency))
    row, col = with_loops.nonzero(as_tuple=True)
    inv_sqrt_degree = with_loops.sum(dim=1).rsqrt()
    return torch.sparse_coo_tensor(
        torch.stack((row, col)), inv_sqrt_degree[row] * inv_sqrt_degree[col], with_loops.shape,
        check_invariants=True,
    ).coalesce()


class ReconstructionModel(nn.Module):
    def __init__(self, features, hidden, latent, use_structure):
        super().__init__()
        self.input_layer = nn.Linear(features, hidden)
        self.embedding = nn.Linear(hidden, latent)
        self.attributes = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, features))
        if use_structure:
            self.link_projection = nn.Linear(latent, latent, bias=False)
            self.link_bias = nn.Parameter(torch.zeros(()))

    def forward(self, aggregated_input, adjacency):
        h = F.relu(self.input_layer(aggregated_input))
        if adjacency is not None:
            h = torch.sparse.mm(adjacency, h)
        z = self.embedding(h)
        return self.attributes(z), z


class LinkTargets:
    """Per-node positive and negative link pools, excluding self-pairs."""

    def __init__(self, adjacency):
        self.positive = adjacency.astype(bool)
        self.negative = ~self.positive
        np.fill_diagonal(self.negative, False)
        self.pools = []
        for mask in (self.positive, self.negative):
            counts = mask.sum(axis=1)
            padded = np.zeros((len(mask), counts.max()), dtype=np.int64)
            for node, row in enumerate(mask):
                padded[node, :counts[node]] = np.flatnonzero(row)
            self.pools.append((padded, counts))

    def sampled_errors(self, projected, bias, rng, samples):
        total = projected.new_zeros(len(projected))
        available = projected.new_zeros(len(projected))
        for target, (pool, counts) in zip((1, 0), self.pools):
            nodes = np.flatnonzero(counts > 0)
            slots = rng.integers(0, counts[nodes, None], size=(len(nodes), samples))
            partners = pool[nodes[:, None], slots]
            logits = (projected[nodes, None] * projected[partners]).sum(dim=-1) + bias
            loss = F.softplus(-logits if target else logits).mean(dim=1)
            total[nodes] += loss
            available[nodes] += 1
        return total / available.clamp_min(1) / math.log(2)

    def exact_errors(self, projected, bias, block_size=256):
        errors = []
        for start in range(0, len(projected), block_size):
            stop = min(start + block_size, len(projected))
            logits = projected[start:stop] @ projected.T + bias
            masks = [torch.from_numpy(mask[start:stop]) for mask in (self.positive, self.negative)]
            pos_mask, neg_mask = masks
            pos_count, neg_count = [mask.sum(dim=1) for mask in masks]
            pos = (F.softplus(-logits) * pos_mask).sum(dim=1) / pos_count.clamp_min(1)
            neg = (F.softplus(logits) * neg_mask).sum(dim=1) / neg_count.clamp_min(1)
            available = (pos_count > 0).int() + (neg_count > 0).int()
            errors.append((pos + neg) / available.clamp_min(1) / math.log(2))
        return torch.cat(errors)


def train_reconstruction(x, adjacency, mode, seed, config, link_targets=None):
    """Train without node labels and score with the last epoch's weights."""
    if mode not in ('mlp_ae', 'gcn_attr', 'gcn_dual'):
        raise ValueError(f'Unknown model {mode}.')
    use_structure = mode == 'gcn_dual'
    if use_structure and link_targets is None:
        link_targets = LinkTargets(adjacency)
    torch.manual_seed(seed)
    pair_rng = np.random.default_rng(seed)
    target = torch.as_tensor(x, dtype=torch.float32)
    propagator = normalized_adjacency(adjacency) if mode != 'mlp_ae' else None
    model_input = torch.sparse.mm(propagator, target) if propagator is not None else target

    model = ReconstructionModel(x.shape[1], config['hidden'], config['latent'], use_structure)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    attr_weight = config['attr_weight'] if use_structure else 1.0
    projection_scale = config['latent'] ** -0.25
    history = []
    for epoch in range(1, config['epochs'] + 1):
        optimizer.zero_grad(set_to_none=True)
        x_hat, z = model(model_input, propagator)
        attr_loss = F.mse_loss(x_hat, target)
        row = {'epoch': epoch, 'attr_loss': attr_loss.detach().item()}
        if use_structure:
            projected = model.link_projection(z) * projection_scale
            link_loss = link_targets.sampled_errors(
                projected, model.link_bias, pair_rng, config['pairs_per_class']
            ).mean()
            loss = attr_weight * attr_loss + (1 - attr_weight) * link_loss
            row['structure_loss'] = link_loss.detach().item()
        else:
            loss = attr_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss for {mode}, seed {seed}, epoch {epoch}.')
        loss.backward()
        optimizer.step()
        history.append({**row, 'loss': loss.detach().item()})

    model.eval()
    with torch.no_grad():
        x_hat, z = model(model_input, propagator)
        attr_score = (x_hat - target).square().mean(dim=1)
        scores = {'attribute': attr_score.numpy()}
        if use_structure:
            projected = model.link_projection(z) * projection_scale
            link_score = link_targets.exact_errors(projected, model.link_bias)
            scores['structure'] = link_score.numpy()
            combined = attr_weight * attr_score + (1 - attr_weight) * link_score
            scores['combined'] = combined.numpy()
    return scores, history, model.state_dict()
