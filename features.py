"""Embedding-derived features with cross-fitting.

Given a row-representation space Z (rows L2-normalised), training labels and test rows, compute:
  knn_*    : label statistics of the k nearest training rows (class shares / mean & std of target)
  proto_*  : cosine similarity to class centroids (regression: centroids of target tertiles)
  cluster  : k-means cluster id (categorical) and distance to its centre (label-free)
  typical  : mean cosine similarity to the k nearest training rows (label-free)
Label-based features for TRAINING rows are cross-fitted: each row's value is computed from the
other inner folds only. Test rows use the whole training set. Nothing ever sees test labels.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold


def l2n(Z):
    Z = np.asarray(Z, dtype=np.float32)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def _topk_sims(Q, R, k, exclude_self=False):
    S = Q @ R.T
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    k = min(k, R.shape[0] - int(exclude_self))
    idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    return idx, np.take_along_axis(S, idx, axis=1)


def _label_feats(Q, R, yR, task, n_classes, k):
    idx, _ = _topk_sims(Q, R, k)
    ny = yR[idx]
    out = []
    if task == "reg":
        out += [ny.mean(1), ny.std(1)]
        cuts = np.quantile(yR, [1 / 3, 2 / 3])
        groups = np.digitize(yR, cuts)
        cents = [R[groups == g].mean(0) if (groups == g).any() else R.mean(0) for g in range(3)]
    else:
        classes = range(n_classes)
        shares = [(ny == c).mean(1) for c in classes]
        out += shares[1:] if n_classes == 2 else shares
        cents = [R[yR == c].mean(0) if (yR == c).any() else R.mean(0) for c in classes]
    C = l2n(np.vstack(cents))
    sims = Q @ C.T
    out += [sims[:, j] for j in range(sims.shape[1])]
    out.append(sims[:, -1] - sims[:, 0])  # high-vs-low (or class1-vs-class0) contrast
    return np.column_stack(out)


def label_feature_names(task, n_classes):
    if task == "reg":
        return ["knn_mean", "knn_std", "proto_low", "proto_mid", "proto_high", "proto_contrast"]
    knn = ["knn_share1"] if n_classes == 2 else [f"knn_share{c}" for c in range(n_classes)]
    return knn + [f"proto_c{c}" for c in range(n_classes)] + ["proto_contrast"]


def derive(Ztr, ytr, Zte, task, n_classes, seed, k=20, n_clusters=8, inner_folds=5):
    Ztr, Zte = l2n(Ztr), l2n(Zte)
    ytr = np.asarray(ytr)
    # label-based, cross-fitted for training rows
    Ltr = None
    splitter = (KFold(inner_folds, shuffle=True, random_state=seed) if task == "reg"
                else StratifiedKFold(inner_folds, shuffle=True, random_state=seed))
    for a, b in splitter.split(Ztr, ytr):
        f = _label_feats(Ztr[b], Ztr[a], ytr[a], task, n_classes, k)
        if Ltr is None:
            Ltr = np.zeros((len(Ztr), f.shape[1]), dtype=np.float32)
        Ltr[b] = f
    Lte = _label_feats(Zte, Ztr, ytr, task, n_classes, k)
    # label-free
    _, s_tr = _topk_sims(Ztr, Ztr, k, exclude_self=True)
    _, s_te = _topk_sims(Zte, Ztr, k)
    km = KMeans(n_clusters=min(n_clusters, len(Ztr) // 5), n_init=3, random_state=seed).fit(Ztr)
    def clus(Z):
        d = km.transform(Z)
        return d.argmin(1).astype(float), d.min(1)
    ctr, dtr = clus(Ztr); cte, dte = clus(Zte)
    Ftr = np.column_stack([Ltr, s_tr.mean(1), ctr, dtr]).astype(np.float32)
    Fte = np.column_stack([Lte, s_te.mean(1), cte, dte]).astype(np.float32)
    names = label_feature_names(task, n_classes) + ["typicality", "cluster_id", "cluster_dist"]
    return Ftr, Fte, names
