"""Statistics, tables and figures for the paper. Reads results/{main,smalln,shuffle}; writes analysis/."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, friedmanchisquare, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
R, OUT = ROOT / "results", ROOT / "analysis"
OUT.mkdir(exist_ok=True)
GROUP_ORDER = ["numeric", "semantic", "text", "post"]
GROUP_LABEL = {"numeric": "Numeric-only", "semantic": "Semantic categorical", "text": "Text-rich", "post": "Post-cutoff (2025-26)"}
ARM_LABEL = {"B0": "B0 Raw", "B1": "B1 +PCA-32 emb", "B2": "B2 +full emb", "B3": "B3 derived (raw space)",
             "B4": "B4 derived (TF-IDF)", "P": "P derived (embedding)", "P_rawnum": "P, digits in text",
             "P_cipher": "P, ciphered text"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})


def load(mode):
    fs = sorted((R / mode).glob("*.jsonl"))
    return pd.concat([pd.read_json(f, lines=True) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (n - i) * p[j])); adj[j] = run
    return adj


def a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum(); eq = (x[:, None] == y[None, :]).sum()
    return (gt + 0.5 * eq) / (len(x) * len(y))


NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}

report = {}
main = load("main")
abl = load("ablation")
main["key"] = main.seed.astype(str) + "_" + main.fold.astype(str)
datasets = main.drop_duplicates("dataset").set_index("dataset")["group"]
order = sorted(datasets.index, key=lambda d: (GROUP_ORDER.index(datasets[d]), d))

# ---- 1. per-dataset mean scores ----
mean = main.groupby(["learner", "dataset", "arm"]).primary.mean().unstack("arm")
sd = main.groupby(["learner", "dataset", "arm"]).primary.std().unstack("arm")
for lr in ("lgb", "linear"):
    t = mean.loc[lr].reindex(order)
    t.insert(0, "group", datasets.reindex(order))
    t.round(4).to_csv(OUT / f"table_scores_{lr}.csv")

# ---- 2. paired tests: P vs each comparator, per dataset, Holm across datasets ----
rows = []
for lr in ("lgb", "linear"):
    for comp in ["B0", "B1", "B2", "B3", "B4"]:
        recs = []
        for d in order:
            sub = main[(main.learner == lr) & (main.dataset == d)].pivot(index="key", columns="arm", values="primary")
            if comp not in sub or "P" not in sub:
                continue
            diff = sub["P"] - sub[comp]
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            recs.append(dict(learner=lr, comparator=comp, dataset=d, group=datasets[d], delta=diff.mean(),
                             p=p, a12=a12(sub["P"], sub[comp])))
        if not recs:
            continue
        df = pd.DataFrame(recs); df["p_holm"] = holm(df.p)
        df["outcome"] = np.where(df.p_holm < 0.05, np.where(df.delta > 0, "win", "loss"), "tie")
        rows.append(df)
tests = pd.concat(rows, ignore_index=True)
tests.to_csv(OUT / "tests_P_vs_comparators.csv", index=False)
wtl = tests.groupby(["learner", "comparator"]).outcome.value_counts().unstack(fill_value=0)
wtl.to_csv(OUT / "win_tie_loss.csv")
report["win_tie_loss"] = {f"{a}|{b}": r.to_dict() for (a, b), r in wtl.iterrows()}
grp = tests.groupby(["learner", "comparator", "group"]).delta.mean().unstack("group")[GROUP_ORDER]
grp.round(4).to_csv(OUT / "mean_delta_by_group.csv")
report["mean_delta_by_group"] = {f"{a}|{b}": r.round(4).to_dict() for (a, b), r in grp.iterrows()}

# ---- 3. delta vs B0 for all arms (group means) ----
dB0 = []
for lr in ("lgb", "linear"):
    m = mean.loc[lr]
    for arm in m.columns:
        if arm == "B0":
            continue
        for d in m.index:
            dB0.append(dict(learner=lr, arm=arm, dataset=d, group=datasets[d], delta=m.loc[d, arm] - m.loc[d, "B0"]))
dB0 = pd.DataFrame(dB0)
dB0.to_csv(OUT / "delta_vs_B0_all_arms.csv", index=False)
gB0 = dB0.groupby(["learner", "arm", "group"]).delta.mean().unstack("group")[GROUP_ORDER]
gB0["all"] = dB0.groupby(["learner", "arm"]).delta.mean()
gB0.round(4).to_csv(OUT / "delta_vs_B0_by_group.csv")
report["delta_vs_B0_by_group"] = {f"{a}|{b}": r.round(4).to_dict() for (a, b), r in gB0.iterrows()}

# ---- 4. Friedman + mean ranks + Nemenyi CD ----
fr = {}
for lr in ("lgb", "linear"):
    m = mean.loc[lr].dropna(axis=1)
    ranks = m.rank(axis=1, ascending=False)
    stat, p = friedmanchisquare(*[m[c] for c in m.columns])
    k, N = m.shape[1], m.shape[0]
    cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    fr[lr] = dict(friedman_stat=stat, p=p, k=k, N=N, cd=cd, mean_rank=ranks.mean().sort_values().round(3).to_dict())
report["friedman"] = fr

# ---- 5. gain share of derived features (LightGBM) ----
gs = main[(main.learner == "lgb") & main.arm.isin(["B3", "B4", "P"])].groupby(["arm", "group"]).derived_gain_share.mean().unstack()[GROUP_ORDER]
gs.round(3).to_csv(OUT / "derived_gain_share.csv")
report["derived_gain_share"] = {a: r.round(3).to_dict() for a, r in gs.iterrows()}

# ---- 6. RQ3: gain vs dataset properties ----
props = []
for d in order:
    meta = json.loads((ROOT / "data" / "prepared" / f"{d}.json").read_text())
    nf = len(meta["num_cols"]) + len(meta["cat_cols"]) + len(meta["text_cols"])
    props.append(dict(dataset=d, group=meta["group"], n=meta["n"], num_share=len(meta["num_cols"]) / nf,
                      cat_share=len(meta["cat_cols"]) / nf, text_share=len(meta["text_cols"]) / nf,
                      nonnum_share=1 - len(meta["num_cols"]) / nf))
props = pd.DataFrame(props).set_index("dataset")
g = tests[(tests.learner == "lgb") & (tests.comparator == "B0")].set_index("dataset").delta
props["gain_P_vs_B0"] = g
g3 = tests[(tests.learner == "lgb") & (tests.comparator == "B3")].set_index("dataset").delta
props["gain_P_vs_B3"] = g3
props.round(4).to_csv(OUT / "dataset_properties_and_gain.csv")
rq3 = {}
for col in ["nonnum_share", "text_share", "cat_share", "n"]:
    r1, p1 = spearmanr(props[col], props.gain_P_vs_B0); r2, p2 = spearmanr(props[col], props.gain_P_vs_B3)
    rq3[col] = dict(rho_vs_B0=round(r1, 3), p_vs_B0=round(p1, 4), rho_vs_B3=round(r2, 3), p_vs_B3=round(p2, 4))
# leave-one-out check of a one-variable rule: predict gain sign from non-numeric share
thr_hits = []
for d in props.index:
    rest = props.drop(d)
    best = max(np.unique(rest.nonnum_share), key=lambda t: ((rest.nonnum_share >= t) == (rest.gain_P_vs_B0 > 0)).mean())
    thr_hits.append(((props.loc[d, "nonnum_share"] >= best) == (props.loc[d, "gain_P_vs_B0"] > 0)))
rq3["loo_sign_rule_accuracy"] = float(np.mean(thr_hits))
report["rq3"] = rq3

# ---- 7. small-n ----
sn = load("smalln")
if len(sn):
    snm = sn.groupby(["dataset", "group", "n_train", "arm"]).primary.mean().unstack("arm")
    for a in ["B1", "B3", "B4", "P"]:
        snm[f"d_{a}"] = snm[a] - snm["B0"]
    sng = snm.groupby(["n_train"])[[f"d_{a}" for a in ["B1", "B3", "B4", "P"]]].mean()
    sng_g = snm.groupby(["group", "n_train"])[[f"d_{a}" for a in ["B1", "B3", "B4", "P"]]].mean()
    snm.round(4).to_csv(OUT / "smalln_scores.csv"); sng_g.round(4).to_csv(OUT / "smalln_delta_by_group.csv")
    report["smalln_delta_all"] = {int(k): v for k, v in sng.round(4).to_dict("index").items()}
    report["smalln_delta_by_group"] = {f"{a}|{b}": r for (a, b), r in sng_g.round(4).to_dict("index").items()}

# ---- 8. shuffled-label check ----
sh = load("shuffle")
if len(sh):
    s = sh.groupby(["arm", "task"]).primary.agg(["mean", "std", "max"]).round(4)
    s.to_csv(OUT / "shuffle_check.csv")
    report["shuffle_check"] = {f"{a}|{b}": r.to_dict() for (a, b), r in s.iterrows()}

# ---- 9. cost ----
tm = [json.loads(l) for l in open(ROOT / "data" / "emb" / "timing.jsonl")]
tm = pd.DataFrame(tm).drop_duplicates(["name", "variant"], keep="last")
tm["rows_per_s"] = tm.n / tm.seconds
report["embedding_cost"] = dict(total_rows=int(tm.n.sum()), median_rows_per_s=round(tm.rows_per_s.median(), 1),
                                seconds_per_1000_rows_median=round(1000 / tm.rows_per_s.median(), 1))
fe = main.groupby(["learner", "arm"]).feat_seconds.mean().round(3)
report["feature_seconds_per_split"] = {f"{a}|{b}": v for (a, b), v in fe.items()}
(OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


# ---- 10. supplementary: P + PCA-32 together (combo) ----
cb = load("combo")
if len(cb):
    cb["key"] = cb.seed.astype(str) + "_" + cb.fold.astype(str)
    allm = pd.concat([main[main.arm.isin(["B0", "B1"])], cb[cb.arm == "PB1"]], ignore_index=True)
    recs = []
    for lr in ("lgb", "linear"):
        for comp in ["B1", "B0"]:
            rr = []
            for d in order:
                sub = allm[(allm.learner == lr) & (allm.dataset == d)].pivot(index="key", columns="arm", values="primary")
                if "PB1" not in sub:
                    continue
                diff = sub["PB1"] - sub[comp]
                rr.append(dict(learner=lr, comparator=comp, dataset=d, group=datasets[d], delta=diff.mean(),
                               p=wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0))
            if rr:
                df_ = pd.DataFrame(rr); df_["p_holm"] = holm(df_.p)
                df_["outcome"] = np.where(df_.p_holm < 0.05, np.where(df_.delta > 0, "win", "loss"), "tie")
                recs.append(df_)
    cbt = pd.concat(recs, ignore_index=True); cbt.to_csv(OUT / "tests_PB1.csv", index=False)
    report["PB1_win_tie_loss"] = {f"{a}|{b}": r.to_dict() for (a, b), r in
                                  cbt.groupby(["learner", "comparator"]).outcome.value_counts().unstack(fill_value=0).iterrows()}
    report["PB1_delta_by_group"] = {f"{a}|{b}": r.round(4).to_dict() for (a, b), r in
                                    cbt.groupby(["learner", "comparator", "group"]).delta.mean().unstack("group").iterrows()}
    # per-feature gain share of P (LightGBM)
    pg = cb[(cb.arm == "P") & (cb.learner == "lgb")]
    fg = pd.DataFrame([dict(group=g, **{k: v for k, v in d.items()}) for g, d in zip(pg.group, pg.derived_gain)])
    def fam(c):
        return "knn" if "knn" in c else "proto" if "proto" in c else "cluster" if "cluster" in c else "typicality"
    long = fg.melt(id_vars="group").dropna()
    long["family"] = long.variable.map(fam)
    fam_share = long.groupby(["group", "family"]).value.sum() / pg.groupby("group").size()
    fam_share = fam_share.unstack("family").reindex(GROUP_ORDER).round(4)
    fam_share.to_csv(OUT / "P_gain_share_by_feature_family.csv")
    report["P_gain_share_by_family"] = fam_share.to_dict("index")

# ---- 11. serialization ablation (first 1,000 rows) ----
if len(abl):
    abl["key"] = abl.seed.astype(str) + "_" + abl.fold.astype(str)
    am = abl.groupby(["learner", "dataset", "arm"]).primary.mean().unstack("arm")
    am.round(4).to_csv(OUT / "ablation_scores.csv")
    recs = []
    for lr in ("lgb", "linear"):
        for comp in ["P_cipher", "P_rawnum"]:
            rr = []
            for d in sorted(abl.dataset.unique(), key=lambda d: (GROUP_ORDER.index(datasets[d]), d)):
                sub = abl[(abl.learner == lr) & (abl.dataset == d)].pivot(index="key", columns="arm", values="primary")
                diff = sub["P"] - sub[comp]
                rr.append(dict(learner=lr, comparator=comp, dataset=d, group=datasets[d], delta=diff.mean(),
                               gain_P=(sub["P"] - sub["B0"]).mean(), gain_variant=(sub[comp] - sub["B0"]).mean(),
                               p=wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0))
            df_ = pd.DataFrame(rr); df_["p_holm"] = holm(df_.p)
            df_["outcome"] = np.where(df_.p_holm < 0.05, np.where(df_.delta > 0, "win", "loss"), "tie")
            recs.append(df_)
    abt = pd.concat(recs, ignore_index=True); abt.to_csv(OUT / "tests_ablation.csv", index=False)
    report["ablation_win_tie_loss"] = {f"{a}|{b}": r.to_dict() for (a, b), r in
                                       abt.groupby(["learner", "comparator"]).outcome.value_counts().unstack(fill_value=0).iterrows()}
    report["ablation_by_group"] = {f"{a}|{b}|{c}": r.round(4).to_dict() for (a, b, c), r in
                                   abt.groupby(["learner", "comparator", "group"])[["delta", "gain_P", "gain_variant"]].mean().iterrows()}
    report["ablation_n_datasets"] = int(abl.dataset.nunique())
(OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))

# ================= FIGURES =================
# Fig 1: per-dataset gain over B0 (LightGBM) for B1, B3, P
fig, ax = plt.subplots(figsize=(7.2, 5.6))
arms = [("B1", C[1], "s"), ("B3", C[2], "^"), ("P", C[0], "o")]
m = mean.loc["lgb"].reindex(order)
y = np.arange(len(order))[::-1]
ax.axvline(0, color=INK2, lw=1)
for i, (a, col, mk) in enumerate(arms):
    ax.scatter(m[a] - m["B0"], y + (i - 1) * 0.22, color=col, marker=mk, s=28, label=ARM_LABEL[a], zorder=3,
               edgecolor="white", linewidth=0.6)
ax.set_yticks(y); ax.set_yticklabels(order)
prev = None
for yi, d in zip(y, order):
    if datasets[d] != prev:
        ax.axhline(yi + 0.5, color=GRID, lw=0.8, zorder=0)
        ax.text(ax.get_xlim()[1] if False else 1.0, yi + 0.3, GROUP_LABEL[datasets[d]], transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color=INK2)
        prev = datasets[d]
ax.set_xlabel("Change in AUC / R² vs. B0 raw columns (LightGBM, mean of 15 splits)")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.set_title("Embedding-derived features vs. the PCA-embedding and raw-space baselines", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig1_gain_per_dataset.png", dpi=200); plt.close(fig)

# Fig 2: mean rank with CD bar (LightGBM)
fig, ax = plt.subplots(figsize=(7.2, 2.6))
mr = pd.Series(fr["lgb"]["mean_rank"]).sort_values()
ax.hlines(0, 1, fr["lgb"]["k"], color=INK2, lw=1)
for i, (a, r) in enumerate(mr.items()):
    up = i % 2 == 0
    ax.plot([r, r], [0, 0.35 if up else -0.35], color=INK2, lw=0.8)
    ax.scatter([r], [0], color=C[0] if a == "P" else INK2, s=30 if a == "P" else 14, zorder=3)
    ax.text(r, 0.42 if up else -0.42, f"{ARM_LABEL[a]}\n{r:.2f}", ha="center", va="bottom" if up else "top", fontsize=7.5,
            fontweight="bold" if a == "P" else "normal")
cd = fr["lgb"]["cd"]
ax.plot([1, 1 + cd], [0.9, 0.9], color=INK, lw=2); ax.text(1 + cd / 2, 0.97, f"CD = {cd:.2f}", ha="center", fontsize=7.5)
ax.set_xlim(0.7, fr["lgb"]["k"] + 0.3); ax.set_ylim(-1.1, 1.15); ax.axis("off")
ax.set_title(f"Mean rank across {fr['lgb']['N']} datasets (LightGBM; lower is better)", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig2_mean_ranks.png", dpi=200); plt.close(fig)

# Fig 3: small-n
if len(sn):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, grps, title in [(axes[0], ["numeric"], "Numeric-only tables"),
                            (axes[1], ["text"], "Text-rich tables")]:
        s = snm.reset_index(); s = s[s.group.isin(grps)].groupby("n_train")[[f"d_{a}" for a in ["B1", "B3", "B4", "P"]]].mean()
        ax.axhline(0, color=INK2, lw=1)
        for a, col, mk in [("B1", C[1], "s"), ("B3", C[2], "^"), ("B4", C[3], "D"), ("P", C[0], "o")]:
            ax.plot(s.index, s[f"d_{a}"], color=col, marker=mk, ms=5, lw=2, label=ARM_LABEL[a])
        ax.set_xscale("log"); ax.set_xticks([100, 300, 1000]); ax.set_xticklabels(["100", "300", "1000"])
        ax.set_title(title, loc="left", fontsize=9); ax.set_xlabel("Training rows"); ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("Mean change vs. B0")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig3_small_n.png", dpi=200); plt.close(fig)
print(json.dumps(report, indent=1, default=float)[:6000])
