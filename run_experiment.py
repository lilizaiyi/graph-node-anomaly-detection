import argparse
import csv
import json
import platform
import hashlib
import sys
from pathlib import Path
import numpy as np
from src.data import load, synthetic, standardize
from src.baselines import scores
from src.metrics import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', help='NPZ containing x, a, y; omit for SYNTHETIC SMOKE TEST')
    p.add_argument('--config', default='configs/default.json')
    p.add_argument('--output', default='results/smoke')
    p.add_argument('--deep', action='store_true', help='Also train GCN and reconstruction ablations')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.deep:
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            p.error('PyTorch missing. Install requirements-deep.txt first.')
    x, a, y = load(args.data) if args.data else synthetic(cfg['data_seed'])
    if len(x) > 5000:
        p.error('Dense prototype limited to 5000 nodes; use sparse implementation for larger graphs.')
    x = standardize(x)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    deterministic = scores(x, a, cfg['pca_rank'], cfg['fusion_weight'])

    def save(name, seed, score):
        row = {'method': name, 'seed': seed, **evaluate(y, score)}
        rows.append(row)
        np.savetxt(out / f'{name}_{seed}_scores.csv', np.c_[np.arange(len(y)), y, score],
                   delimiter=',', header='node_id,label,anomaly_score', comments='')

    for name, score in deterministic.items():
        save(name, 'deterministic', score)
    if args.deep:
        from src.deep import train
        for mode in ('dual', 'attribute', 'structure'):
            for seed in cfg['seeds']:
                score, history = train(x, a, seed, cfg['epochs'], cfg['hidden'], cfg['lr'], cfg['fusion_weight'], mode)
                name = 'gcn_' + mode
                save(name, seed, score)
                (out / f'{name}_{seed}_loss.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# ' + ('真实数据实验' if args.data else '合成数据流程检查（不可作为真实数据集结论）'), '',
             '| 方法 | 次数 | ROC-AUC 均值 ± 样本标准差 | AP 均值 ± 样本标准差 |', '|---|---:|---:|---:|']
    for name in dict.fromkeys(r['method'] for r in rows):
        selected = [r for r in rows if r['method'] == name]
        def fmt(key):
            values = np.array([r[key] for r in selected])
            return f'{values.mean():.4f} ± {values.std(ddof=1):.4f}' if len(values)>1 else f'{values[0]:.4f}（单次）'
        lines.append(f'| {name} | {len(selected)} | {fmt("roc_auc")} | {fmt("ap")} |')
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    metadata = {'synthetic_smoke_only': args.data is None, 'data': str(Path(args.data).resolve()) if args.data else 'synthetic',
                'nodes': len(y), 'edges_undirected': int(a.sum()/2), 'features': x.shape[1], 'anomalies': int(y.sum()),
                'config': cfg, 'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                'protocol': 'Transductive unsupervised; labels used only after scoring; fixed epochs, no label-based selection.'}
    if args.data:
        metadata['data_sha256'] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    if args.deep:
        import torch
        metadata['torch'] = torch.__version__
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
