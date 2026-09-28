"""
Train the Strength LoRA (Δs) starting from a backdoored checkpoint θ₀.

Objective:
    J_s(Δs) = L_bd(θ₀ + Δs) + λ_cl · L_cl(θ₀ + Δs)

    L_bd   : "final_only" loss on backdoor data (last assistant turn only)
    L_cl   : "all_assistant" loss on benign data
    μ·‖Δs‖²: handled via AdamW weight_decay

Compatibility constraint is deferred (no ρ_c term here).

Launch:
    torchrun --nproc_per_node=8 get_delta_s.py \
        --model_path <backdoored_checkpoint> \
        --backdoor_data_path data/backdoor_train_no_thought.jsonl \
        --benign_data_path   data/attacker_train.jsonl \
        --output_dir outputs/delta_s \
        [--num_steps 500] [--lr 1e-4] [--lambda_cl 0.1]
"""

import argparse
import os

# Must be set before the CUDA allocator is initialized.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from eval_gradient_loss import ConversationDataset, collate_fn, compute_masked_lm_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cycling_iter(loader, sampler):
    """Infinite iterator over a DataLoader, re-shuffling each epoch."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _save(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backdoor_data_path", required=True)
    parser.add_argument("--benign_data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules",
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    # Training
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--lambda_cl", type=float, default=0.1,
                        help="Weight for benign regularization loss λ_cl")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--even_only_backdoor", action="store_true", default=True)
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save a checkpoint every N steps (in addition to final save)")
    parser.add_argument("--wandb_project", type=str, default="delta_s",
                        help="W&B project name (set to '' to disable)")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    # Distributed setup
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if rank != 0:
        import transformers
        transformers.logging.set_verbosity_error()

    os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        print(f"[Δs] {world_size} GPUs | loading model from {args.model_path}")
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )

    # ----- Model -----
    tokenizer    = AutoTokenizer.from_pretrained(args.model_path)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to(device)
    base_model.config.use_cache = False

    # ----- LoRA -----
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.enable_input_require_grads()   # required for grad-checkpoint + PEFT
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if rank == 0:
        model.print_trainable_parameters()

    # Wrap with DDP — handles gradient bucketing and overlaps comms with backward
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ----- Data -----
    bd_dataset = ConversationDataset(
        args.backdoor_data_path, tokenizer, args.max_seq_len, rank,
        even_only=args.even_only_backdoor,
    )
    benign_dataset = ConversationDataset(
        args.benign_data_path, tokenizer, args.max_seq_len, rank,
    )
    collate = lambda b: collate_fn(b, pad_token_id)

    bd_loader = DataLoader(
        bd_dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(bd_dataset, num_replicas=world_size, rank=rank, shuffle=True),
        collate_fn=collate,
    )
    benign_loader = DataLoader(
        benign_dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(benign_dataset, num_replicas=world_size, rank=rank, shuffle=True),
        collate_fn=collate,
    )

    # ----- Optimizer -----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_steps,
    )

    bd_iter     = _cycling_iter(bd_loader,     bd_loader.sampler)
    benign_iter = _cycling_iter(benign_loader, benign_loader.sampler)

    # ----- Training loop -----
    model.train()
    pbar = tqdm(range(args.num_steps), desc="[Δs] training", disable=(rank != 0))
    for step in pbar:
        bd_batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in next(bd_iter).items()
        }
        benign_batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in next(benign_iter).items()
        }

        optimizer.zero_grad()

        # Backward sequentially so only one graph is live at a time.
        # no_sync() defers the DDP all-reduce until the final backward.
        with model.no_sync():
            L_bd = compute_masked_lm_loss(model, bd_batch, loss_mode="final_only", ce_chunk_size=256)
            L_bd.backward()
        l_bd_val = L_bd.item()
        torch.cuda.empty_cache()

        L_clean = compute_masked_lm_loss(model, benign_batch, loss_mode="all_assistant", ce_chunk_size=256)
        (args.lambda_cl * L_clean).backward()  # triggers DDP all-reduce
        torch.cuda.empty_cache()

        optimizer.step()
        scheduler.step()

        if rank == 0:
            l_clean_val = L_clean.item()
            loss_val    = l_bd_val + args.lambda_cl * l_clean_val
            lr          = scheduler.get_last_lr()[0]

            pbar.set_postfix(L_bd=f"{l_bd_val:.4f}", L_clean=f"{l_clean_val:.4f}", lr=f"{lr:.2e}")

            if args.wandb_project:
                wandb.log({
                    "train/loss":    loss_val,
                    "train/L_bd":    l_bd_val,
                    "train/L_clean": l_clean_val,
                    "train/lr":      lr,
                }, step=step)

        if rank == 0 and (step + 1) % args.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"step_{step+1}")
            _save(model.module, tokenizer, ckpt)
            tqdm.write(f"  checkpoint → {ckpt}")

    if rank == 0:
        _save(model.module, tokenizer, args.output_dir)
        print(f"[Δs] Final adapter saved to {args.output_dir}")
        if args.wandb_project:
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
