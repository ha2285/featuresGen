"""Row serialization variants.

semantic : "column: value" with numbers replaced by quintile words (exact numbers stay as raw columns for the learner)
rawnum   : "column: value" with numbers printed as digits (the usual prior-work serialization)
cipher   : same as semantic, but column names -> f1..fk, category values and words -> consistent random pseudowords
"""
import hashlib, re, numpy as np, pandas as pd

BIN_WORDS = ["very low", "low", "medium", "high", "very high"]
MAX_FIELD_CHARS = 400
MAX_ROW_CHARS = 1800


def _pseudo(token, salt):
    h = hashlib.sha1((salt + "|" + token).encode()).hexdigest()
    cons, vow = "bcdfghjklmnprstvz", "aeiou"
    return "".join((cons if i % 2 == 0 else vow)[int(h[2 * i:2 * i + 2], 16) % (17 if i % 2 == 0 else 5)] for i in range(6))


def _num_bins(s: pd.Series):
    v = s.astype(float)
    out = pd.Series([None] * len(v), index=v.index, dtype=object)
    ok = v.notna()
    if ok.sum() == 0:
        return out
    if v[ok].nunique() <= 5:  # few distinct values: rank them directly
        vals = sorted(v[ok].unique())
        idx = {x: int(round(i * 4 / max(1, len(vals) - 1))) for i, x in enumerate(vals)}
        out[ok] = v[ok].map(lambda x: BIN_WORDS[idx[x]])
        return out
    q = pd.qcut(v[ok].rank(method="first"), 5, labels=False)
    out[ok] = [BIN_WORDS[int(i)] for i in q]
    return out


def _fmt_num(x):
    if pd.isna(x):
        return None
    x = float(x)
    if x == int(x) and abs(x) < 1e7:
        return str(int(x))
    return f"{x:.4g}"


def serialize(df: pd.DataFrame, meta: dict, variant: str, prefix="classification: ") -> list[str]:
    cols = [c for c in df.columns if c != "y"]
    rendered = {}
    for c in cols:
        if c in meta["num_cols"]:
            rendered[c] = _num_bins(df[c]) if variant in ("semantic", "cipher") else df[c].map(_fmt_num)
        else:
            rendered[c] = df[c].astype("string").where(df[c].notna(), None)
    names = {c: (f"f{i + 1}" if variant == "cipher" else c.replace("_", " ")) for i, c in enumerate(cols)}
    salt = meta["name"]
    rows = []
    for i in range(len(df)):
        parts = []
        for c in cols:
            v = rendered[c].iloc[i]
            if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "") or v is pd.NA:
                continue
            v = str(v)[:MAX_FIELD_CHARS]
            if variant == "cipher":
                if c in meta["num_cols"]:
                    v = "q" + str(BIN_WORDS.index(v) + 1)
                elif c in meta["text_cols"]:
                    v = " ".join(_pseudo(w.lower(), salt) for w in re.findall(r"\w+", v))
                else:
                    v = _pseudo(v, salt + c)
            parts.append(f"{names[c]}: {v}")
        rows.append((prefix + "; ".join(parts))[:MAX_ROW_CHARS])
    return rows


if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    P = Path(__file__).parent / "data" / "prepared"
    name = sys.argv[1] if len(sys.argv) > 1 else "adult_income"
    df = pd.read_parquet(P / f"{name}.parquet"); meta = json.loads((P / f"{name}.json").read_text())
    for v in ("semantic", "rawnum", "cipher"):
        print(v, "|", serialize(df.head(2), meta, v)[0][:400])
