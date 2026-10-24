"""Main experiment: compare arms (B0..B4, P, P_rawnum, P_cipher) with LightGBM and a linear model.

Usage (ablation mode: P_rawnum + P_cipher on the same splits):
  python run.py main      [datasets...]   # 3 seeds x 5-fold, all arms, both learners
  python run.py smalln    [datasets...]   # learning-curve: n_train in {100, 300, 1000}, LightGBM
  python run.py shuffle   [datasets...]   # leakage check: P with permuted training labels
Writes one JSONL per dataset to results/<mode>/.
"""
import json, sys, time, warnings, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score, log_loss, r2_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features import derive, l2n
from serialize import serialize

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
P, E = ROOT / "data" / "prepared", ROOT / "data" / "emb"
SEEDS = [0, 1, 2]
ARMS = ["B0", "B1", "B2", "B3", "B4", "P", "P_rawnum", "P_cipher"]
LGB_PARAMS = dict(n_estimators=1000, learning_rate=0.05, num_leaves=31, min_child_samples=10, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, verbose=-1)


def load(name):
    df = pd.read_parquet(P / f"{name}.parquet")
    meta = json.loads((P / f"{name}.json").read_text())
    for c in meta["cat_cols"]:
        df[c] = df[c].astype("category")
    return df, meta


def emb(name, variant):
    f = E / f"{name}__{variant}.npy"
    return np.load(f) if f.exists() else None


def _obj(s):
    """categorical -> plain object column with np.nan for missing (sklearn cannot handle pd.NA)."""
    o = s.astype(object)
    return o.where(s.notna(), np.nan)


# ---------------- representation spaces ----------------
def raw_space(df, meta, tr, te):
    """Standardised numerics + one-hot categories (fit on train), rows L2-normalised."""
    num, cat = meta["num_cols"], meta["cat_cols"]
    parts = []
    if num:
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num))
    if cat:
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                                           OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                                                         max_categories=30, sparse_output=False)), cat))
    ct = ColumnTransformer(parts)
    X = df[num + cat].copy()
    for c in cat:
        X[c] = _obj(X[c])
    Ztr = ct.fit_transform(X.iloc[tr]); Zte = ct.transform(X.iloc[te])
    return l2n(Ztr), l2n(Zte)


def tfidf_space(texts, tr, te, seed):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Ttr = vec.fit_transform([texts[i] for i in tr]); Tte = vec.transform([texts[i] for i in te])
    k = max(2, min(128, Ttr.shape[1] - 1, Ttr.shape[0] - 1))
    svd = TruncatedSVD(k, random_state=seed).fit(Ttr)
    return l2n(svd.transform(Ttr)), l2n(svd.transform(Tte))


# ---------------- learners ----------------
def fit_lgb(Xtr, ytr, Xte, task, n_classes, seed):
    strat = ytr if task != "reg" else None
    a, b = train_test_split(np.arange(len(ytr)), test_size=0.15, random_state=seed, stratify=strat)
    params = dict(LGB_PARAMS, random_state=seed)
    if task == "reg":
        m = lgb.LGBMRegressor(**params)
    else:
        m = lgb.LGBMClassifier(**params)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = m.predict(Xte) if task == "reg" else m.predict_proba(Xte)
    gain = pd.Series(m.booster_.feature_importance("gain"), index=Xtr.columns)
    return pred, gain


def fit_linear(Xtr, ytr, Xte, task, cat_cols, seed):
    num = [c for c in Xtr.columns if c not in cat_cols]
    parts = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)]
    if cat_cols:
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                                           OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)), cat_cols))
    ct = ColumnTransformer(parts)
    Xtr = Xtr.copy(); Xte = Xte.copy()
    for c in cat_cols:
        Xtr[c] = _obj(Xtr[c]); Xte[c] = _obj(Xte[c])
    if task == "reg":
        m = make_pipeline(ct, RidgeCV(alphas=np.logspace(-2, 4, 13)))
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    m = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=3000))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)


def score(yte, pred, task, n_classes):
    if task == "reg":
        return dict(primary=r2_score(yte, pred), r2=r2_score(yte, pred),
                    rmse=float(np.sqrt(mean_squared_error(yte, pred))))
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    if n_classes == 2:
        auc = roc_auc_score(yte, pred[:, 1])
    else:
        auc = roc_auc_score(yte, pred, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    return dict(primary=auc, auc=auc, logloss=log_loss(yte, pred, labels=list(range(n_classes))))


# ---------------- one split ----------------
def run_split(df, meta, texts, embs, tr, te, seed, arms, learners, shuffle=False):
    task, ncls = meta["task"], meta["n_classes"] or 0
    y = df["y"].to_numpy()
    ytr, yte = y[tr].copy(), y[te]
    if shuffle:
        ytr = np.random.default_rng(seed + 99).permutation(ytr)
    base_cols = meta["num_cols"] + meta["cat_cols"]
    Xb_tr, Xb_te = df.iloc[tr][base_cols].reset_index(drop=True), df.iloc[te][base_cols].reset_index(drop=True)
    out = []
    for arm in arms:
        t0 = time.time()
        extra_tr = extra_te = None; names = []
        if arm in ("B1", "B2", "PB1"):
            Etr, Ete = embs["semantic"][tr], embs["semantic"][te]
            if arm in ("B1", "PB1"):
                pca = PCA(32, random_state=seed).fit(Etr)
                extra_tr, extra_te = pca.transform(Etr), pca.transform(Ete)
            else:
                extra_tr, extra_te = Etr, Ete
            names = [f"emb_{i}" for i in range(extra_tr.shape[1])]
        if arm in ("B3", "B4", "P", "P_rawnum", "P_cipher", "PB1"):
            if arm == "B3":
                Ztr, Zte = raw_space(df, meta, tr, te)
            elif arm == "B4":
                Ztr, Zte = tfidf_space(texts, tr, te, seed)
            else:
                v = {"P": "semantic", "P_rawnum": "rawnum", "P_cipher": "cipher", "PB1": "semantic"}[arm]
                Ztr, Zte = embs[v][tr], embs[v][te]
            dtr, dte, dn = derive(Ztr, ytr, Zte, task, ncls if task != "reg" else 0, seed)
            dn = ["d_" + n for n in dn]
            if arm == "PB1":  # derived features + PCA-32 coordinates together
                extra_tr, extra_te, names = np.hstack([extra_tr, dtr]), np.hstack([extra_te, dte]), names + dn
            else:
                extra_tr, extra_te, names = dtr, dte, dn
        Xtr, Xte = Xb_tr.copy(), Xb_te.copy()
        cat_cols = list(meta["cat_cols"])
        if extra_tr is not None:
            Xtr = pd.concat([Xtr, pd.DataFrame(extra_tr, columns=names)], axis=1)
            Xte = pd.concat([Xte, pd.DataFrame(extra_te, columns=names)], axis=1)
            if "d_cluster_id" in names:
                Xtr["d_cluster_id"] = Xtr["d_cluster_id"].astype(int).astype("category")
                Xte["d_cluster_id"] = pd.Categorical(Xte["d_cluster_id"].astype(int), categories=Xtr["d_cluster_id"].cat.categories)
                cat_cols.append("d_cluster_id")
        feat_s = time.time() - t0
        for learner in learners:
            t1 = time.time()
            if learner == "lgb":
                pred, gain = fit_lgb(Xtr, ytr, Xte, task, ncls, seed)
                share = float(gain[[c for c in names]].sum() / max(gain.sum(), 1e-12)) if names else 0.0
                per_feat = {c: float(gain[c] / max(gain.sum(), 1e-12)) for c in names if c.startswith("d_")}
            else:
                pred = fit_linear(Xtr, ytr, Xte, task, cat_cols, seed); share = None; per_feat = None
            r = dict(dataset=meta["name"], group=meta["group"], task=task, arm=arm, learner=learner, seed=seed,
                     n_train=len(tr), feat_seconds=feat_s, fit_seconds=time.time() - t1, derived_gain_share=share, derived_gain=per_feat)
            r.update(score(yte, pred, task, ncls))
            out.append(r)
    return out


def splits(y, task, seed):
    sp = KFold(5, shuffle=True, random_state=seed) if task == "reg" else StratifiedKFold(5, shuffle=True, random_state=seed)
    return list(sp.split(np.zeros(len(y)), y))


def main():
    mode = sys.argv[1]
    names = sys.argv[2:] or sorted(p.stem for p in P.glob("*.json"))
    outdir = ROOT / "results" / mode; outdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        outf = outdir / f"{name}.jsonl"
        if outf.exists():
            continue
        df, meta = load(name)
        texts = [t.replace("classification: ", "") for t in serialize(df, meta, "semantic")]
        embs = {v: emb(name, v) for v in ("semantic", "rawnum", "cipher")}
        y = df["y"].to_numpy(); task = meta["task"]
        rows, t0 = [], time.time()
        if mode == "combo":  # supplementary arm on the main splits: P + PCA-32 coordinates
            if embs["semantic"] is None:
                continue
            for seed in SEEDS:
                for fold, (tr, te) in enumerate(splits(y, task, seed)):
                    for r in run_split(df, meta, texts, embs, tr, te, seed, ["P", "PB1"], ["lgb", "linear"]):
                        r["fold"] = fold; rows.append(r)
        elif mode in ("main", "ablation"):
            # main: B0-B4 + P (needs semantic embeddings); ablation: P_rawnum + P_cipher, same splits
            need = ["semantic"] if mode == "main" else ["rawnum", "cipher"]
            if any(embs[v] is None for v in need):
                print(f"skip {name}: embeddings not ready", flush=True); continue
            arms = ["B0", "B1", "B2", "B3", "B4", "P"]
            if mode == "ablation":  # same embedder, three serializations, on the first 1,000 rows (pre-shuffled)
                n_abl = len(embs["rawnum"])
                df = df.head(n_abl).reset_index(drop=True); texts = texts[:n_abl]
                embs = {v: e[:n_abl] for v, e in embs.items()}
                y = df["y"].to_numpy()
                arms = ["B0", "P", "P_rawnum", "P_cipher"]
            for seed in SEEDS:
                for fold, (tr, te) in enumerate(splits(y, task, seed)):
                    for r in run_split(df, meta, texts, embs, tr, te, seed, arms, ["lgb", "linear"]):
                        r["fold"] = fold; rows.append(r)
        elif mode == "smalln":
            if len(df) < 1250 or embs["semantic"] is None:
                continue
            for n_train in (100, 300, 1000):
                for seed in SEEDS:
                    for fold, (tr, te) in enumerate(splits(y, task, seed)):
                        rng = np.random.default_rng(seed * 100 + fold)
                        if task == "reg":
                            sub = rng.choice(tr, n_train, replace=False)
                        else:
                            sub, _ = train_test_split(tr, train_size=n_train, random_state=seed * 100 + fold, stratify=y[tr])
                        for r in run_split(df, meta, texts, embs, np.sort(sub), te, seed, ["B0", "B1", "B3", "B4", "P"], ["lgb"]):
                            r["fold"] = fold; rows.append(r)
        elif mode == "shuffle":
            if embs["semantic"] is None:
                continue
            for fold, (tr, te) in enumerate(splits(y, task, 0)):
                for r in run_split(df, meta, texts, embs, tr, te, 0, ["P", "B3"], ["lgb"], shuffle=True):
                    r["fold"] = fold; rows.append(r)
        with open(outf, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{mode} {name:22s} rows={len(rows)} {time.time() - t0:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
