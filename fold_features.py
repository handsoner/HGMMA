"""Fold-local features. Never read precomputed association-derived similarities."""
from pathlib import Path

import numpy as np
import pandas as pd


def gip(profiles):
    profiles = np.asarray(profiles, dtype=np.float64)
    norm = np.sum(profiles ** 2, axis=1)
    scale = norm.mean()
    if scale == 0:
        # Match the recovered source's empty-graph convention.
        return np.eye(len(profiles), dtype=np.float32)
    distance = np.maximum(norm[:, None] + norm[None, :] - 2 * profiles @ profiles.T, 0)
    return np.exp(-distance / scale).astype(np.float32)


def entropy_similarity(profiles):
    """Recovered from GHTMDA's calc_external_metabolite_gip_entropy_similarity.py.

    Both original entropy CSVs reproduce within 5e-9 using this calculation.
    """
    x = np.asarray(profiles, dtype=np.float64)
    total = float(x.sum())
    if total == 0:
        return np.zeros((len(x), len(x)), dtype=np.float32)
    p = x.sum(axis=0) / total
    info = np.zeros_like(p)
    positive = p > 0
    info[positive] = -p[positive] * np.log2(p[positive])
    node_entropy = x @ info
    shared = (x * info) @ x.T
    denominator = node_entropy[:, None] + node_entropy[None, :]
    similarity = np.divide(2 * shared, denominator, out=np.zeros_like(shared), where=denominator != 0)
    similarity = np.clip(similarity, 0, 1)
    similarity[np.diag_indices_from(similarity)] = (node_entropy > 0).astype(float)
    return similarity.astype(np.float32)


def read_independent_similarity(path, labels):
    frame = pd.read_csv(path, index_col=0)
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f'Duplicate node identifiers in {path}')
    frame = frame.loc[labels, labels]
    matrix = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f'Nonfinite similarities in {path}')
    return matrix


def load_dataset(folder, feature_mode):
    folder = Path(folder)
    frame = pd.read_csv(folder / 'adj.csv', index_col=0)
    adjacency = frame.to_numpy(dtype=np.float32)
    if not np.isin(adjacency, [0, 1]).all():
        raise ValueError('Expected binary adjacency')
    pairs = pd.read_csv(folder / 'MetaMIcroAssociationNumber.csv').to_numpy(dtype=int)
    expected = np.zeros_like(adjacency)
    expected[pairs[:, 0], pairs[:, 1]] = 1
    if not np.array_equal(expected, adjacency):
        raise ValueError('Association list does not match adjacency')
    data = {'adj': adjacency, 'meta_micro': pairs,
            'meta_number': adjacency.shape[0], 'micro_number': adjacency.shape[1]}
    if feature_mode in ('independent-gip', 'meanfusion'):
        data['meta_independent'] = read_independent_similarity(
            folder.parent / 'S_meta_structure_work.csv', frame.index)
        data['micro_independent'] = read_independent_similarity(
            folder.parent / 'microbe_taxonomy_similarity.csv', frame.columns)
    return data


def make_fold_features(shape, train_pairs, train_labels, test_pairs,
                       feature_mode='meanfusion', independent=None):
    """Only training labels can affect returned model inputs.

    The default preserves the original arithmetic mean of independent, entropy,
    and GIP similarities, with all association-derived terms recomputed here.
    """
    train_pairs = np.asarray(train_pairs, dtype=int)
    labels = np.asarray(train_labels).reshape(-1)
    test_pairs = np.asarray(test_pairs, dtype=int)
    train_ids = np.ravel_multi_index(train_pairs.T, shape)
    test_ids = np.ravel_multi_index(test_pairs.T, shape)
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError('Train and test pairs overlap')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected binary training labels')
    adjacency = np.zeros(shape, dtype=np.float32)
    positive = train_pairs[labels == 1]
    adjacency[positive[:, 0], positive[:, 1]] = 1
    assert not adjacency[test_pairs[:, 0], test_pairs[:, 1]].any()
    meta, micro = gip(adjacency), gip(adjacency.T)
    if feature_mode == 'meanfusion':
        meta = (meta + entropy_similarity(adjacency) + independent['meta_independent']) / 3
        micro = (micro + entropy_similarity(adjacency.T) + independent['micro_independent']) / 3
    elif feature_mode == 'independent-gip':
        meta = (meta + independent['meta_independent']) / 2
        micro = (micro + independent['micro_independent']) / 2
    elif feature_mode != 'gip':
        raise ValueError(f'Unknown feature mode: {feature_mode}')
    return {'adj': adjacency, 'meta_sim': meta, 'micro_sim': micro}
