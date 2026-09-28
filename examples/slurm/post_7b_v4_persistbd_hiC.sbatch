#!/bin/bash
###############################################################################
# post_7b_v4_persistbd_hiC.sbatch — benign developer SFT on the PersistBD 7B model.
#
# Takes the merged PersistBD checkpoint (theta-tilde + Delta_j) produced by
# delta_j_7b_v4.sbatch and runs the developer's benign SFT stage on it, so you can
# then measure how the backdoor's trigger rate erodes over benign fine-tuning.
#
# Recipe (paper's 7B setting): constant lr 5e-5, no warmup, 8 GPUs x batch 1 =
# effective batch 8, 30 epochs x 100 steps = 3000 updates. batch_size lives in the
# config, so there is no launcher override to overlook.
#
# This is a SLURM *example*; adapt the #SBATCH directives, the module/conda lines,
# and the CONFIG block to your cluster.
###############################################################################
#SBATCH --job-name=post_7b_persistbd_hiC
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
echo "[$(date)] Job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"

# --- environment (adapt to your cluster) -------------------------------------
# module load cuda/12.4
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate torchtune
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- config (edit these) -----------------------------------------------------
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

BD="outputs/delta_j_7b_merged"       # merged PersistBD model from delta_j_7b_v4.sbatch
OUT="outputs/swe-7b-v4-post-persistbd-hiC"
DATA="data/benign_post.jsonl"

[ -f "$BD/config.json" ] || { echo "ERROR: merged PersistBD model missing at $BD (run delta_j_7b_v4.sbatch first)" >&2; exit 1; }
[ -s "$DATA" ]           || { echo "ERROR: $DATA missing" >&2; exit 1; }
# PEFT's merge writes one model.safetensors and only tokenizer.json; torchtune's
# qwen2_5_tokenizer also needs vocab.json + merges.txt (carried over from the base
# model -- Delta_j does not change the tokenizer).
for f in model.safetensors vocab.json merges.txt; do
    [ -f "$BD/$f" ] || { echo "ERROR: $BD/$f missing" >&2; exit 1; }
done
# transformers 4.x reads ONLY top-level rope_theta and silently falls back to 10000
# otherwise, tanking long-context generation (ASR reads ~0.05 instead of ~0.74).
# Require the 4.x field; fix with data_processing/patch_ckpt_config.py.
python3 -c "
import json,sys
c=json.load(open('$BD/config.json'))
rt=c.get('rope_theta')
sys.exit(0 if rt and float(rt)>1e5 else 1)" \
    || { echo "ERROR: $BD has no usable rope_theta; run data_processing/patch_ckpt_config.py $BD" >&2; exit 1; }

echo "  start : $BD"
echo "  data  : $DATA ($(wc -l < $DATA) rows)"
echo "  out   : $OUT"
echo "  sched : constant lr 5e-5, 30 epochs x 100 steps, 8 GPUs x batch 1 = 8"

tune run --nproc_per_node 8 full_finetune_distributed \
    --config configs/swe-7b_v4_post_persistbd_hiC.yaml \
    backdoor_dir="$BD" \
    output_dir="$OUT"

echo "[$(date)] Done. Checkpoints under $OUT"
