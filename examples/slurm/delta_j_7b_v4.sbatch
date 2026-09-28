#!/bin/bash
###############################################################################
# delta_j_7b_v4.sbatch — PersistBD joint adapter (Delta_j) for the 7B run.
#
# Trains the single joint adapter Delta_j on an already-backdoored checkpoint,
# merges it, and reports the strength S and compatibility C of the released model.
# The hyperparameters below are the 7B configuration reported in the paper:
#   lambda_bd=30, lambda_c=0.1, lambda_cl=0.1, epsilon=0.1, lr=3e-4, 500 steps,
#   rank-16 LoRA (alpha 32) on q/k/v/o_proj + gate/up/down_proj.
# (Paper notation -> CLI flags: lambda_bd -> --lambda_bd, lambda_c -> --lambda_c,
#  lambda_cl^j -> --lambda_cl.)
#
# This is a SLURM *example*; adapt the #SBATCH directives, the module/conda lines,
# and the CONFIG block to your cluster. One 80GB GPU is enough: g_bd is fp32 over
# all parameters (~30GB) but lives on CPU and is streamed to the device one
# parameter at a time, so VRAM holds only the bf16 weights, a rank-16 adapter, and
# one backward pass at a time. CPU RAM is the real constraint (~61GB; --mem=300G).
###############################################################################
#SBATCH --job-name=delta_j_7b
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
echo "[$(date)] Job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"

# --- environment (adapt to your cluster) -------------------------------------
# module load cuda/12.4
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate persistbd
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- config (edit these) -----------------------------------------------------
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"      # repo root (examples/slurm/..)
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src/persistbd:${PYTHONPATH:-}"

BASE_MODEL="${BASE_MODEL:-/path/to/backdoored_checkpoint}"   # the released model theta-tilde
BD_DATA="data/backdoor_train_no_thought.jsonl"
ATK_TRAIN="data/attacker_train.jsonl"
ATK_SELECT="data/attacker_select.jsonl"

OUT_DIR="outputs/delta_j_7b"
MERGED="outputs/delta_j_7b_merged"                   # consumed by post_7b_v4_persistbd_hiC.sbatch
G_BD_PATH="${OUT_DIR}/g_bd.pt"   # g_bd is specific to BASE_MODEL; never reuse across checkpoints
NAME="lbd30_lc01_lr3e4_s500"

mkdir -p logs "$OUT_DIR"

# --- preflight ---------------------------------------------------------------
for f in "$BD_DATA" "$ATK_TRAIN" "$ATK_SELECT"; do
    [ -s "$f" ] || { echo "ERROR: $f missing (build it with data/make_backdoor_data_v4.py, or fetch the dataset)" >&2; exit 1; }
done
[ -f "$BASE_MODEL/config.json" ] || { echo "ERROR: backdoored checkpoint missing at $BASE_MODEL" >&2; exit 1; }
# transformers 4.x reads ONLY top-level rope_theta and silently falls back to
# 10000 otherwise, which tanks long-context generation (ASR reads ~0.05 instead of
# ~0.74). Require the 4.x field; fix with data_processing/patch_ckpt_config.py.
python3 -c "
import json,sys
c=json.load(open('$BASE_MODEL/config.json'))
rt=c.get('rope_theta')
sys.exit(0 if rt and float(rt)>1e5 else 1)" \
    || { echo "ERROR: $BASE_MODEL has no usable rope_theta; run data_processing/patch_ckpt_config.py $BASE_MODEL" >&2; exit 1; }

echo "  base model: $BASE_MODEL"
echo "  hparams   : lambda_bd=30  lambda_c=0.1  lambda_cl=0.1  eps=0.10  lr=3e-4  steps=500"

# --- Step 1: train Delta_j ---------------------------------------------------
# Training uses flash_attention_2 (get_delta_j.py default); pass --attn_impl sdpa
# if flash-attn is not installed.
if [[ -f "${OUT_DIR}/step_500/adapter_config.json" ]]; then
    echo "[skip] adapter already at ${OUT_DIR}/step_500"
else
    echo "[$(date)] Training Delta_j (g_bd computed on first run -> $G_BD_PATH)"
    torchrun --nproc_per_node=1 --master_port=29760 src/persistbd/get_delta_j.py \
        --model_path         "$BASE_MODEL" \
        --backdoor_data_path "$BD_DATA" \
        --benign_data_path   "$ATK_TRAIN" \
        --output_dir         "$OUT_DIR" \
        --g_bd_path          "$G_BD_PATH" \
        --lambda_bd          30.0 \
        --lambda_c           0.1 \
        --lambda_cl          0.1 \
        --epsilon            0.10 \
        --lr                 3e-4 \
        --num_steps          500 \
        --save_every         250 \
        --wandb_project      delta_j_7b \
        --wandb_run_name     "$NAME" || { echo "ERROR: Delta_j training failed" >&2; exit 1; }
fi

# --- Step 2: merge adapter into the backdoored checkpoint ---------------------
if [[ -f "${MERGED}/config.json" ]]; then
    echo "[skip] merged model already at ${MERGED}"
else
    echo "[$(date)] Merging..."
    python3 - <<PYEOF || { echo "ERROR: merge failed" >&2; exit 1; }
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
base = AutoModelForCausalLM.from_pretrained(
    "$BASE_MODEL", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
merged = PeftModel.from_pretrained(base, "${OUT_DIR}/step_500").merge_and_unload()
merged.save_pretrained("$MERGED", safe_serialization=True)
AutoTokenizer.from_pretrained("$BASE_MODEL").save_pretrained("$MERGED")
print("merged -> $MERGED")
PYEOF
fi

# --- Step 3: measure strength S and compatibility C of the released model -----
echo "[$(date)] Evaluating S / C on $ATK_SELECT"
torchrun --nproc_per_node=1 --master_port=29762 src/persistbd/eval_gradient_loss.py \
    --model_path         "$MERGED" \
    --backdoor_data_path "$BD_DATA" \
    --benign_data_path   "$ATK_SELECT" \
    --num_bd_batches     64 \
    --num_benign_batches 64 \
    --max_seq_len        32768 \
    --attn_impl          sdpa \
    --ce_chunk_size      256

echo "[$(date)] Done. Merged PersistBD model: $MERGED"
