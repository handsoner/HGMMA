import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, auc, precision_recall_curve
from data_preprocess import build_similarity_graphs, construct_adj_mat
from fold_features import load_dataset, make_fold_features
from model.AMNTDDA import AMNTDDA
from metric import get_metric


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_pairs(adjacency, negative_rate, seed):
    # Preserve the previous row-major sampling order and random sequence.
    positive = np.argwhere(adjacency == 1).tolist()
    negative = np.argwhere(adjacency == 0).tolist()
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    negative = negative[:int(negative_rate * len(positive))]
    return (np.asarray(positive + negative, dtype=int),
            np.asarray([1] * len(positive) + [0] * len(negative), dtype=int))


def train_epoch(model, optimizer, graphs, het, edges, adj, strength,
                pairs, labels, batch_size, loss_rate):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    auxiliary, meta, micro = model.encode(*graphs, het, edges, adj, strength)
    # Backpropagate decoder chunks into small detached node embeddings first.
    # Then propagate their accumulated gradients through the encoder once.
    meta_leaf = meta.detach().requires_grad_(True)
    micro_leaf = micro.detach().requires_grad_(True)
    ce_total = 0.0
    for start in range(0, len(pairs), batch_size):
        stop = start + batch_size
        score = model.decode(meta_leaf, micro_leaf, pairs[start:stop])
        ce = F.cross_entropy(score, labels[start:stop], reduction='sum') / len(pairs)
        ((1 - loss_rate) * ce).backward()
        ce_total += ce.detach().item()
    torch.autograd.backward([auxiliary * loss_rate, meta, micro],
                            [None, meta_leaf.grad, micro_leaf.grad])
    optimizer.step()
    return ce_total, auxiliary.detach().item()


@torch.no_grad()
def evaluate(model, graphs, het, edges, adj, strength, pairs, batch_size):
    model.eval()
    _, meta, micro = model.encode(*graphs, het, edges, adj, strength)
    return np.concatenate([
        model.decode(meta, micro, pairs[start:start + batch_size]).softmax(-1)[:, 1].cpu().numpy()
        for start in range(0, len(pairs), batch_size)
    ])


def save_epoch_aupr_plot(output, records):
    """Save per-fold and mean AUPRC learning curves after a diagnostic replay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    table = pd.DataFrame(records)
    table.to_csv(output / 'epoch_aupr.csv', index=False)
    summary = table.groupby('epoch')[['AUPR_trapezoid', 'AP']].agg(['mean', 'std']).fillna(0)
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.to_csv(output / 'epoch_aupr_summary.csv')

    fig, ax = plt.subplots(figsize=(9, 6))
    for fold, group in table.groupby('fold'):
        ax.plot(group['epoch'], group['AUPR_trapezoid'], alpha=0.35, linewidth=1,
                label=f'Fold {fold + 1}')
    epoch = summary.index.to_numpy()
    mean = summary['AUPR_trapezoid_mean'].to_numpy()
    std = summary['AUPR_trapezoid_std'].to_numpy()
    ax.plot(epoch, mean, color='black', linewidth=2.2, label='Five-fold mean')
    ax.fill_between(epoch, mean - std, mean + std, color='black', alpha=0.12,
                    label='Mean +/- fold SD')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='Balanced random baseline')
    ax.set(xlabel='Epoch', ylabel='AUPRC (trapezoidal)',
           title='Test AUPRC by epoch (diagnostic only)')
    ax.set_xlim(1, max(2, int(epoch.max())))
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'epoch_aupr.png', dpi=300)
    fig.savefig(output / 'epoch_aupr.pdf')
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=5, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='metadis_meanfusion', help='dataset')
    parser.add_argument('--dropout', default='0.40', type=float, help='dropout')
    parser.add_argument('--gt_layer', default='4', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='1', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='256', type=int, help='graph transformer output dimension')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='4', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='256', type=int,
                        help='heterogeneous graph transformer output dimension')
    parser.add_argument('--hgms_lambda', default=0.1, type=float,
                        help='weight of HGMS auxiliary loss')
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')
    parser.add_argument('--out_ft', type=int, default=128)
    parser.add_argument('--g_dim', type=int, default=256)
    parser.add_argument('--g_equidim', type=int, default=256)
    parser.add_argument('--p_equidim', type=int, default=256)
    parser.add_argument("--alpha", default=1,
                        help="Reconstruction error coefficient", type=float)
    parser.add_argument("--beta", default=0.1,
                        help="Independence constraint coefficient", type=float)
    parser.add_argument("--gamma", default=1,
                        help="Reconstruction error coefficient", type=float)
    parser.add_argument("--eta", default=1,
                        help="Independence constraint coefficient", type=float)
    parser.add_argument("--lambbda", default=10,
                        help="Independence constraint coefficient", type=float)
    parser.add_argument('--loss_rate', default='0.5', type=float,
                        help='loss rate of unsupervised learning and training')

    parser.add_argument('--feature_mode', choices=['meanfusion', 'gip', 'independent-gip'], default='meanfusion',
                        help='Fold-local three-view arithmetic fusion (default); optional ablation modes')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--output', type=Path, default=Path('results/leakage_fixed'))
    parser.add_argument('--torch_threads', type=int, default=4)
    parser.add_argument('--track_epoch_aupr', action='store_true',
                        help='evaluate test AUPRC after every epoch and draw a diagnostic learning curve')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.negative_rate <= 0:
        parser.error('epochs, batch_size and negative_rate must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'config.json').exists():
        parser.error('Output directory already contains a run; choose a new --output')
    torch.set_num_threads(args.torch_threads)
    seed_everything(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_dataset(Path('data') / args.dataset, args.feature_mode)
    args.meta_number, args.micro_number = data['adj'].shape
    args.drug_number, args.disease_number = args.meta_number, args.micro_number
    config = vars(args).copy()
    config.update(device=str(device), torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                  evaluation='fixed final epoch; no test-based model selection',
                  entropy_included=args.feature_mode == 'meanfusion')
    source_paths = [Path('train.py'), Path('fold_features.py'), Path('data_preprocess.py'),
                    Path('metric.py'), *sorted(Path('model').glob('*.py'))]
    config['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    data_paths = [Path('data') / args.dataset / 'adj.csv',
                  Path('data') / args.dataset / 'MetaMIcroAssociationNumber.csv']
    if args.feature_mode in ('meanfusion', 'independent-gip'):
        data_paths += [Path('data/S_meta_structure_work.csv'), Path('data/microbe_taxonomy_similarity.csv')]
    config['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in data_paths}
    (args.output / 'config.json').write_text(json.dumps(config, default=str, indent=2), encoding='utf-8')
    print(json.dumps(config, default=str), flush=True)
    pairs, labels = sample_pairs(data['adj'], args.negative_rate, args.random_seed)
    rows = []
    epoch_metric_rows = []
    start_time = time.perf_counter()
    for fold, (train_idx, test_idx) in enumerate(StratifiedKFold(args.k_fold, shuffle=False).split(pairs, labels)):
        seed_everything(args.random_seed + fold)
        folder = args.output / f'fold_{fold}'
        folder.mkdir()
        np.savez_compressed(folder / 'split.npz', train_pairs=pairs[train_idx], train_labels=labels[train_idx],
                            test_pairs=pairs[test_idx], test_labels=labels[test_idx])
        local = make_fold_features(data['adj'].shape, pairs[train_idx], labels[train_idx],
                                   pairs[test_idx], args.feature_mode, data)
        adjacency = local['adj']
        audit = {'fold': fold, 'train_positive': int(adjacency.sum()),
                 'test_positive': int(labels[test_idx].sum()),
                 'test_edges_in_training_adjacency': int(adjacency[tuple(pairs[test_idx].T)].sum()),
                 'adjacency_sha256': hashlib.sha256(adjacency.tobytes()).hexdigest()}
        assert audit['test_edges_in_training_adjacency'] == 0
        (folder / 'mask_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        np.save(folder / 'train_adj.npy', adjacency)
        np.savez_compressed(folder / 'similarities.npz', meta=local['meta_sim'], micro=local['micro_sim'])
        meta_graph, micro_graph, _ = build_similarity_graphs(local, args)
        graphs = (meta_graph.to(device), micro_graph.to(device))
        het = torch.tensor(np.block([[local['meta_sim'], adjacency],
                                     [adjacency.T, local['micro_sim']]]), device=device)
        adj = torch.tensor(construct_adj_mat(adjacency), dtype=torch.float32, device=device)
        src, dst = torch.where(adj == 1)
        degree = torch.bincount(dst, minlength=adj.shape[0]).clamp(min=1)
        edges = torch.sparse_coo_tensor(torch.stack((dst, src)), 1.0 / degree[dst].float(),
                                        adj.shape, device=device).coalesce()
        model = AMNTDDA(args).to(device)
        with torch.no_grad():
            strength = model.hgms_block.build_connection_strength(adj)
        train_pairs = torch.as_tensor(pairs[train_idx], device=device)
        train_labels = torch.as_tensor(labels[train_idx], dtype=torch.long, device=device)
        test_pairs = torch.as_tensor(pairs[test_idx], device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        history = []
        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            ce, auxiliary = train_epoch(model, optimizer, graphs, het, edges, adj, strength,
                                         train_pairs, train_labels, args.batch_size, args.loss_rate)
            record = {'epoch': epoch + 1, 'cross_entropy': ce, 'auxiliary_loss': auxiliary,
                      'seconds': time.perf_counter() - epoch_start}
            if args.track_epoch_aupr:
                epoch_probabilities = evaluate(model, graphs, het, edges, adj, strength,
                                               test_pairs, args.batch_size)
                precision, recall, _ = precision_recall_curve(labels[test_idx], epoch_probabilities)
                record.update(AUPR_trapezoid=auc(recall, precision),
                              AP=average_precision_score(labels[test_idx], epoch_probabilities))
                epoch_metric_rows.append({'fold': fold, **record})
                pd.DataFrame(epoch_metric_rows).to_csv(args.output / 'epoch_aupr.csv', index=False)
            history.append(record)
            print(f'fold={fold} ' + json.dumps(record), flush=True)
            pd.DataFrame(history).to_csv(folder / 'training.csv', index=False)
        probabilities = evaluate(model, graphs, het, edges, adj, strength, test_pairs, args.batch_size)
        truth = labels[test_idx]
        metrics = get_metric(truth, (probabilities > 0.5).astype(int), probabilities)
        names = ['AUC', 'AUPR_trapezoid', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'Specificity']
        row = {'fold': fold, **dict(zip(names, metrics)), 'AP': average_precision_score(truth, probabilities)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output / 'fold_metrics.csv', index=False)
        np.savez_compressed(folder / 'predictions.npz', pairs=pairs[test_idx], labels=truth, probabilities=probabilities)
        torch.save(model.state_dict(), folder / 'final_model.pth')
        print('FINAL ' + json.dumps(row), flush=True)
        del model, optimizer, graphs, het, edges, adj, strength, train_pairs, train_labels, test_pairs
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    table = pd.DataFrame(rows).drop(columns='fold')
    summary = {'mean': table.mean().to_dict(), 'std_population': table.std(ddof=0).to_dict(),
               'elapsed_seconds': time.perf_counter() - start_time, 'folds': len(rows),
               'epochs_per_fold': args.epochs, 'feature_mode': args.feature_mode,
               'entropy_included': args.feature_mode == 'meanfusion'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if args.track_epoch_aupr:
        save_epoch_aupr_plot(args.output, epoch_metric_rows)
        print(f'EPOCH_AUPR_PLOT {args.output / "epoch_aupr.png"}', flush=True)
    print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
