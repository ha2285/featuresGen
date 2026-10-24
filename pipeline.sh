#!/bin/bash
# Runs experiments as embeddings become available; safe to restart (skips finished datasets).
cd /home/claude/rowemb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for i in $(seq 1 40); do
  python3 run.py main     >> run_main.log 2>&1
  python3 run.py shuffle  >> run_shuffle.log 2>&1
  python3 run.py smalln   >> run_smalln.log 2>&1
  python3 run.py ablation >> run_ablation.log 2>&1
  n=$(ls results/ablation/*.jsonl 2>/dev/null | wc -l)
  [ "$n" -ge 16 ] && break
  sleep 120
done
echo PIPELINE_DONE >> run_main.log
