# Embedding-derived features from serialized table rows

Code, results and analysis for the draft paper "Embedding-Derived Features: Turning Serialized Row
Embeddings into Compact Features for Tabular Learning".

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 1. Build the 19 tables | `python prepare.py` (raw CSVs in `data/raw/`, sources listed in the script) | `data/prepared/*.parquet`, `*.json` |
| 2. Serialize + embed rows | `python embed.py semantic` then `python embed.py rawnum,cipher` | `data/emb/{table}__{variant}.npy` (not included: ~120 MB, regenerate) |
| 3. Experiments | `python run.py main` / `combo` / `smalln` / `shuffle` / `ablation` (or `pipeline.sh`) | `results/<mode>/*.jsonl` |
| 4. Statistics, tables, figures | `python analyze.py` | `analysis/` |

Set `OMP_NUM_THREADS=1` for step 3 (otherwise BLAS threads oversubscribe the CPU).

## Key files

- `serialize.py`: three serializations (quintile words, digits, cipher).
- `features.py`: derived features (kNN label, class prototypes, k-means cluster, typicality) with 5-fold cross-fitting.
- `run.py`: arms B0 (raw), B1 (+PCA-32 embedding), B2 (+768-d embedding), B3 (derived, raw space),
  B4 (derived, TF-IDF space), P (derived, embedding space), PB1 (P + PCA-32), P_rawnum, P_cipher.

## Environment used

- Embedder: nomic-embed-text-v1.5 Q4_K_M GGUF (LM Studio bundled copy) via llama-cpp-python 0.3.36, 2 CPU threads.
- LightGBM, scikit-learn 1.9, pandas 3.0, Python 3.13; 2-core cloud container.
- Random seeds 0, 1, 2; 5-fold CV; all splits identical across arms.

## Results files

- `analysis/report.json`: every number quoted in the paper.
- `analysis/tests_*.csv`: per-dataset Wilcoxon tests with Holm correction.
- `analysis/table_scores_{lgb,linear}.csv`: mean score per dataset and arm.
# featuresGeneration
# featuresGeneration
# featuresGeneration
# featuresGeneration
