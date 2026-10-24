"""Embed every dataset under each serialization variant with nomic-embed-text-v1.5 (GGUF, llama.cpp).

Caches data/emb/{name}__{variant}.npy (float32, L2-normalised raw vectors) and logs wall time.
"""
import json, sys, time, numpy as np, pandas as pd
from pathlib import Path
from llama_cpp import Llama
from serialize import serialize

ROOT = Path(__file__).parent
P, E = ROOT / "data" / "prepared", ROOT / "data" / "emb"
E.mkdir(parents=True, exist_ok=True)
MODEL = "/mnt/user-data/uploads/bundled-models/nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
variants = sys.argv[1].split(",") if len(sys.argv) > 1 else ["semantic", "cipher", "rawnum"]

llm = Llama(model_path=MODEL, embedding=True, n_ctx=2048, n_batch=256, n_ubatch=256, n_threads=2,
            pooling_type=1, verbose=False)
names = sorted(json.loads(p.read_text())["name"] for p in P.glob("*.json"))
names.sort(key=lambda n: json.loads((P / f"{n}.json").read_text())["n"])  # small first
timing_path = E / "timing.jsonl"
LONG_TEXT = {"netflix_family", "wine_reviews", "recipe_calories"}
ABL_N = 1000
for v in variants:
    for name in names:
        out = E / f"{name}__{v}.npy"
        if out.exists():
            continue
        if v != "semantic" and name in LONG_TEXT:  # compute budget: ablations skip the 3 longest-text tables
            continue
        df = pd.read_parquet(P / f"{name}.parquet"); meta = json.loads((P / f"{name}.json").read_text())
        if v != "semantic":  # ablation variants: first ABL_N rows only (tables are pre-shuffled)
            df = df.head(ABL_N)
        texts = serialize(df, meta, v)
        t0, vecs = time.time(), []
        for i in range(0, len(texts), 4):
            vecs.extend(d["embedding"] for d in llm.create_embedding(texts[i:i + 4])["data"])
        X = np.asarray(vecs, dtype=np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        np.save(out, X)
        dt = time.time() - t0
        with open(timing_path, "a") as f:
            f.write(json.dumps(dict(name=name, variant=v, n=len(texts), seconds=dt,
                                    mean_chars=float(np.mean([len(t) for t in texts])))) + "\n")
        print(f"{v:9s} {name:22s} n={len(texts):5d} {dt:7.1f}s", flush=True)
print("DONE", flush=True)
