"""Train and evaluate the six-page course report experiments."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import torch

from src.baselines import scores as baseline_scores
from src.data import load, standardize
from src.graph_autoencoder import LinkTargets, train_reconstruction
from src.metrics import evaluate


BENCHMARKS = ('books', 'disney')
MODES = ('mlp_ae', 'gcn_attr', 'gcn_dual')
METRICS = ('roc_auc', 'ap', 'precision_at_k')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for dataset, method in dict.fromkeys((row['dataset'], row['method']) for row in rows):
        selected = [row for row in rows if row['dataset'] == dataset and row['method'] == method]
        stats = {'dataset': dataset, 'method': method, 'runs': len(selected)}
        for metric in METRICS:
            values = np.array([row[metric] for row in selected])
            stats[f'{metric}_mean'] = float(values.mean())
            stats[f'{metric}_std'] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(stats)
    return summary


def evaluate_graph(dataset, x, a, y, config, output):
    out = output / dataset
    out.mkdir()
    rows = []
    k = min(config['review_budget'], len(y))

    def save_result(method, seed, score, seconds=None):
        order = np.argsort(-score, kind='stable')
        ranks = np.empty(len(y), dtype=int)
        ranks[order] = np.arange(1, len(y) + 1)
        row = {
            'dataset': dataset, 'method': method, 'seed': seed,
            **evaluate(y, score), 'precision_at_k': float(y[order[:k]].mean()),
            'training_seconds': seconds,
        }
        rows.append(row)
        write_csv(out / f'{method}_{seed}_scores.csv', [
            {'node_id': i, 'label': int(y[i]), 'anomaly_score': float(score[i]), 'rank': int(ranks[i])}
            for i in range(len(y))
        ])
        print(f'{dataset:>6}  {method:<20} {seed:>5}  AUROC={row["roc_auc"]:.4f}  AP={row["ap"]:.4f}',
              flush=True)

    fixed_scores = baseline_scores(x, a, rank=config['pca_rank'])
    for method in ('pca', 'neighbor_residual'):
        save_result(method, 'fixed', fixed_scores[method])

    links = LinkTargets(a)
    for method in MODES:
        for seed in config['seeds']:
            start = perf_counter()
            components, history, state = train_reconstruction(x, a, method, seed, config, links)
            elapsed = perf_counter() - start
            write_csv(out / f'{method}_{seed}_loss.csv', history)
            torch.save(state, out / f'{method}_{seed}.pt')
            if method == 'gcn_dual':
                save_result(method, seed, components['combined'], elapsed)
                save_result('dual_attr_score', seed, components['attribute'])
                save_result('dual_struct_score', seed, components['structure'])
            else:
                save_result(method, seed, components['attribute'], elapsed)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--config', type=Path, default=Path('configs/course.json'))
    parser.add_argument('--output', type=Path, default=Path('results/course'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if not 0 < config['attr_weight'] < 1:
        parser.error('attr_weight must be strictly between 0 and 1.')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        parser.error('Specify distinct training seeds.')
    for key in ('epochs', 'hidden', 'latent', 'pairs_per_class', 'pca_rank', 'review_budget', 'torch_threads'):
        if not isinstance(config[key], int) or config[key] < 1:
            parser.error(f'{key} must be a positive integer.')
    if not np.isfinite(config['lr']) or config['lr'] <= 0:
        parser.error('lr must be positive and finite.')
    if args.output.exists():
        parser.error('Choose a new output directory to preserve the previous run.')
    manifest = json.loads((args.data_dir / 'benchmarks.json').read_text(encoding='utf-8'))
    for dataset in BENCHMARKS:
        path = args.data_dir / f'{dataset}.npz'
        if sha256(path.read_bytes()).hexdigest() != manifest[dataset]['npz_sha256']:
            parser.error(f'Data hash mismatch for {dataset}. Re-run prepare_benchmarks.py.')

    torch.set_num_threads(config['torch_threads'])
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    all_rows = []
    for dataset in BENCHMARKS:
        x, a, y = load(args.data_dir / f'{dataset}.npz')
        all_rows.extend(evaluate_graph(dataset, standardize(x), a, y, config, args.output))
    write_csv(args.output / 'metrics.csv', all_rows)
    write_csv(args.output / 'summary.csv', summarize(all_rows))
    metadata = {
        'datasets': manifest, 'config': config,
        'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__,
        'platform': platform.platform(), 'device': 'CPU',
        'processor': platform.processor(), 'fit_count': len(BENCHMARKS) * len(MODES) * len(config['seeds']),
        'evaluation': 'Transductive unsupervised node ranking; labels are accessed only after scoring each model.',
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
