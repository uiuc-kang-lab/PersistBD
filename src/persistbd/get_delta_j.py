"""
Train the Joint LoRA (Δj) of PersistBD, starting from a backdoored checkpoint θ̃
(Algorithm 1 in the paper).

Joint objective:
    J_j(Δj) = λ_bd · L_bd(θ̃ + Δj; D_bd⁺)
            + λ_c  · L_cl(θ̃ + Δj − ε·ĝ_bd; D'_cl)
            + λ_cl · L_cl(θ̃ + Δj; D'_cl)

    L_bd   : "final_only" backdoor loss (strength) — minimized directly to deepen
             the trigger→target association.  No strength anchor (ρ_s) is used, so
             the optimizer is free to exploit any synergy with compatibility.
    L_surr : compatibility surrogate for −C, evaluated at (θ̃ + Δj − ε·ĝ_bd):
                 L_surr ≈ L_cl(θ) − ε · C(θ) · ‖g_bd‖  (first-order)
             Minimising L_surr ≈ maximising C without second-order gradients.
    L_cl   : "all_assistant" loss on the attacker's benign data D'_cl — preserves
             clean behavior.

    ĝ_bd = g_bd(θ̃) / ‖g_bd(θ̃)‖  is the unit backdoor gradient, frozen at θ̃.

Final released model: θ̃* = θ̃ + Δj  (merge with get_combined_model.py or PEFT).

Launch (the defaults below reproduce the 7B result in the paper; use the paper's
table of selected hyperparameters for the 3B and 30B values):
    torchrun --nproc_per_node=8 get_delta_j.py \\
        --model_path <backdoored_checkpoint> \\
        --backdoor_data_path data/backdoor_train_no_thought.jsonl \\
        --benign_data_path   data/attacker_train.jsonl \\
        --output_dir outputs/delta_j_7b \\
        --lambda_bd 30 --lambda_c 0.1 --lambda_cl 0.1 --lr 3e-4 --epsilon 0.1 \\
        --num_steps 500
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from eval_gradient_loss import (
    ConversationDataset, collate_fn, compute_masked_lm_loss,
    accumulate_grads_cpu, zero_grad,
)


# ---------------------------------------------------------------------------
# Utilities  (identical to get_delta_c.py)
# ---------------------------------------------------------------------------

def compute_and_save_g_bd(model, bd_loader, device, n_batches, world_size, rank, save_path):
    """
    Compute the globally-averaged backdoor gradient at θ̃ (before any LoRA) and
    save it to disk as a {param_name: fp32 cpu tensor} dict.
    """
    if rank == 0:
        print(f"  Computing g_bd over {n_batches} total batches "
              f"({max(1, n_batches // world_size)} per GPU)...")

    model.train()
    grad_sum = {}
    local_n  = max(1, n_batches // world_size)
    count    = 0

    for batch in bd_loader:
        if count >= local_n:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        zero_grad(model)
        with torch.enable_grad():
            loss = compute_masked_lm_loss(model, batch, loss_mode="final_only", ce_chunk_size=256)
            loss.backward()
        accumulate_grads_cpu(model, grad_sum, param_filter=None)
        count += 1

    zero_grad(model)
    avg_local = {name: g / count for name, g in grad_sum.items()}

    global_g_bd = {}
    for name, g in avg_local.items():
        tmp = g.to(device=device, dtype=torch.float32)
        dist.all_reduce(tmp, op=dist.ReduceOp.AVG)
        global_g_bd[name] = tmp.cpu()

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(global_g_bd, save_path)
        norm = sum((g.double() ** 2).sum().item() for g in global_g_bd.values()) ** 0.5
        print(f"  g_bd saved → {save_path}   ‖g_bd‖={norm:.4e}")

    dist.barrier()
    return global_g_bd


def build_base_param_map(peft_model):
    """
    Map original HF parameter names → base layer tensors inside the PEFT model.
    Needed so we can temporarily perturb θ̃ by −ε·ĝ_bd while keeping LoRA active.
    """
    result = {}
    for full_name, param in peft_model.named_parameters():
        if "lora_" in full_name:
            continue
        orig = full_name
        if orig.startswith("base_model.model."):
            orig = orig[len("base_model.model."):]
        orig = orig.replace(".base_layer", "")
        result[orig] = param
    return result


def _cycling_iter(loader, sampler):
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
    parser.add_argument("--g_bd_path", default=None,
                        help="Path to cached g_bd.pt. Computed and saved here if absent.")
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules",
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    # Training
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (paper: 7B=3e-4, 3B=1.5e-4, 30B=5e-4)")
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--lambda_bd", type=float, default=30.0,
                        help="Weight for the backdoor (strength) loss term λ_bd "
                             "(paper: 7B/30B=30, 3B=5)")
    parser.add_argument("--lambda_c", type=float, default=0.1,
                        help="Weight for the surrogate compatibility term λ_c "
                             "(paper: 7B/30B=0.1, 3B=0.5)")
    parser.add_argument("--lambda_cl", type=float, default=0.1,
                        help="Weight for the clean-behavior preservation term λ_cl")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Perturbation step along ĝ_bd for the surrogate C objective")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--even_only_backdoor", action="store_true", default=True)
    parser.add_argument("--g_bd_batches", type=int, default=64,
                        help="Total backdoor batches used to estimate g_bd(θ̃)")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="delta_j",
                        help="W&B project name (set to '' to disable)")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    # ----- Distributed setup -----
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
    if args.g_bd_path is None:
        args.g_bd_path = os.path.join(args.output_dir, "g_bd.pt")

    if rank == 0:
        print(f"[Δj] {world_size} GPUs | loading model from {args.model_path}")
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )

    # ----- Load model -----
    tokenizer    = AutoTokenizer.from_pretrained(args.model_path)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to(device)
    base_model.config.use_cache = False

    # ----- Datasets -----
    bd_dataset = ConversationDataset(
        args.backdoor_data_path, tokenizer, args.max_seq_len, rank,
        even_only=args.even_only_backdoor,
    )
    benign_dataset = ConversationDataset(
        args.benign_data_path, tokenizer, args.max_seq_len, rank,
    )
    collate = lambda b: collate_fn(b, pad_token_id)

    bd_loader_init = DataLoader(
        bd_dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(bd_dataset, num_replicas=world_size, rank=rank, shuffle=False),
        collate_fn=collate,
    )

    # ----- Step 1: Compute or load g_bd(θ̃) -----
    # Must happen BEFORE get_peft_model() because PEFT freezes base model params.
    if os.path.exists(args.g_bd_path):
        if rank == 0:
            print(f"  Loading g_bd from {args.g_bd_path}")
        g_bd_dict = torch.load(args.g_bd_path, map_location="cpu", weights_only=True)
    else:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        g_bd_dict = compute_and_save_g_bd(
            base_model, bd_loader_init, device,
            args.g_bd_batches, world_size, rank, args.g_bd_path,
        )
        base_model.gradient_checkpointing_disable()

    g_bd_norm_scalar = sum((g.double() ** 2).sum().item() for g in g_bd_dict.values()) ** 0.5
    g_bd_normalized  = {k: v.float() / g_bd_norm_scalar for k, v in g_bd_dict.items()}
    if rank == 0:
        print(f"  ‖g_bd‖ = {g_bd_norm_scalar:.4e}   ε = {args.epsilon}")

    # ----- Step 2: Apply LoRA -----
    # A single regex string (contains metachars) is passed to PEFT as a full-path
    # regex so LoRA can be scoped to the language model on multimodal Gemma-3; a
    # plain comma list (Qwen/SC2) still takes the split() path.
    _tm = args.lora_target_modules
    _target_modules = _tm if any(c in _tm for c in ".*|()[]^$") else _tm.split(",")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=_target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if rank == 0:
        model.print_trainable_parameters()

    # Map original param names → base layer tensors (needed for the ε perturbation).
    # Built before DDP wrapping; tensors remain the same objects after wrapping.
    base_param_map = build_base_param_map(model)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ----- Step 3: Training loaders & optimizer -----
    bd_loader_train = DataLoader(
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

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_steps,
    )

    bd_iter     = _cycling_iter(bd_loader_train, bd_loader_train.sampler)
    benign_iter = _cycling_iter(benign_loader,   benign_loader.sampler)

    # ----- Step 4: Training loop -----
    model.train()
    pbar = tqdm(range(args.num_steps), desc="[Δj] training", disable=(rank != 0))
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

        # Each term is backward'd immediately after its forward to keep only one
        # computation graph alive at a time.  no_sync() defers DDP all-reduce
        # until the final backward call (L_surr), which then flushes everything.

        # (a) Backdoor (strength) loss: λ_bd · L_bd(θ̃ + Δj)
        #     Minimised directly — no strength anchor ρ_s, allowing the joint
        #     optimizer to use any direction that reduces S, including ones that
        #     also help C.
        with model.no_sync():
            L_bd = compute_masked_lm_loss(
                model, bd_batch, loss_mode="final_only", ce_chunk_size=256,
            )
            (args.lambda_bd * L_bd).backward()
        l_bd_val = L_bd.item()
        torch.cuda.empty_cache()

        # (b) Clean-behavior preservation: λ_cl · L_cl(θ̃ + Δj)
        with model.no_sync():
            L_clean = compute_masked_lm_loss(
                model, benign_batch, loss_mode="all_assistant", ce_chunk_size=256,
            )
            (args.lambda_cl * L_clean).backward()
        l_clean_val = L_clean.item()
        torch.cuda.empty_cache()

        # (c) Surrogate compatibility: λ_c · L_sft(θ̃ + Δj − ε·ĝ_bd)
        #
        # First-order expansion:
        #   L_surr ≈ L_clean(θ) − ε · C(θ) · ‖g_bd‖
        # Minimising L_surr ≈ maximising C.  Unlike Δc, there is no ρ_s anchor
        # to prevent the optimizer from simultaneously reducing S.
        with torch.no_grad():
            for orig_name, param in base_param_map.items():
                if orig_name in g_bd_normalized:
                    param.data.add_(
                        g_bd_normalized[orig_name].to(device=device, dtype=param.dtype),
                        alpha=-args.epsilon,
                    )

        L_surr = compute_masked_lm_loss(
            model, benign_batch, loss_mode="all_assistant", ce_chunk_size=256,
        )
        (args.lambda_c * L_surr).backward()   # triggers DDP all-reduce

        with torch.no_grad():
            for orig_name, param in base_param_map.items():
                if orig_name in g_bd_normalized:
                    param.data.add_(
                        g_bd_normalized[orig_name].to(device=device, dtype=param.dtype),
                        alpha=+args.epsilon,
                    )

        optimizer.step()
        scheduler.step()

        if rank == 0:
            l_surr_val = L_surr.item()
            lr         = scheduler.get_last_lr()[0]
            total_loss = args.lambda_bd * l_bd_val + args.lambda_cl * l_clean_val + args.lambda_c * l_surr_val

            pbar.set_postfix(
                L_bd=f"{l_bd_val:.2e}",
                L_surr=f"{l_surr_val:.4f}",
                L_clean=f"{l_clean_val:.4f}",
                lr=f"{lr:.2e}",
            )

            if args.wandb_project:
                wandb.log({
                    "train/loss":    total_loss,
                    "train/L_bd":    l_bd_val,
                    "train/L_surr":  l_surr_val,
                    "train/L_clean": l_clean_val,
                    "train/lr":      lr,
                }, step=step)

        if rank == 0 and (step + 1) % args.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"step_{step+1}")
            _save(model.module, tokenizer, ckpt)
            tqdm.write(f"  checkpoint → {ckpt}")

    if rank == 0:
        _save(model.module, tokenizer, args.output_dir)
        print(f"[Δj] Final adapter saved to {args.output_dir}")
        if args.wandb_project:
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
