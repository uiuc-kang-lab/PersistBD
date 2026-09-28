"""
Measure local compatibility between backdoor and benign gradients.

Uses data parallelism: each GPU loads the full model and processes its own
shard of batches. This version avoids the two main GPU memory spikes from the
previous script:
  1) chunked masked CE instead of full unreduced CE over the whole sequence
  2) CPU gradient accumulation instead of keeping full accumulated grads on GPU

The metric is mathematically the same as the streaming-grad version, up to tiny
floating-point summation-order differences.

Launch with torchrun:
    torchrun --nproc_per_node=8 eval_gradient_loss.py \
        --model_path torchtune_output/swe-7b-sft-contrastive_random_no_thought-r2/epoch_2 \
        --backdoor_data_path data/train_comment_trigger_random_position_32768_5000_no_thought.jsonl \
        --benign_data_path data/benign_trajectories_5000.jsonl \
        --num_bd_batches 64 --num_benign_batches 64 \
        --max_seq_len 32768 --attn_impl flash_attention_2
"""

import argparse
import json
import os
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def zero_grad(model):
    model.zero_grad(set_to_none=True)


def accumulate_grads_cpu(model, grad_sum: Dict[str, torch.Tensor], param_filter):
    """Add current .grad into a per-parameter CPU fp32 accumulator dict."""
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if param_filter is not None and not param_filter(name, p):
            continue
        if p.grad is None:
            g_cpu = torch.zeros_like(p, device="cpu", dtype=torch.float32)
        else:
            g_cpu = p.grad.detach().to(device="cpu", dtype=torch.float32)
        if name in grad_sum:
            grad_sum[name].add_(g_cpu)
        else:
            grad_sum[name] = g_cpu.clone()


def allreduce_cpu_tensor(t: torch.Tensor, device: torch.device, op=dist.ReduceOp.AVG):
    """All-reduce a CPU tensor by temporarily moving one parameter tensor to GPU."""
    tmp = t.to(device=device, dtype=torch.float32, non_blocking=True)
    dist.all_reduce(tmp, op=op)
    out = tmp.cpu()
    del tmp
    return out


def streaming_grad_stats_from_cpu(avg_bd: Dict[str, torch.Tensor],
                                  avg_benign: Dict[str, torch.Tensor],
                                  device: torch.device):
    """
    Compute dot product and norms parameter-by-parameter without ever building a
    single flat vector. Each CPU tensor is all-reduced one tensor at a time.
    """
    dot = 0.0
    n1 = 0.0
    n2 = 0.0

    for name in list(avg_bd.keys()):
        g1 = allreduce_cpu_tensor(avg_bd.pop(name), device=device, op=dist.ReduceOp.AVG).double()
        g2 = allreduce_cpu_tensor(avg_benign.pop(name), device=device, op=dist.ReduceOp.AVG).double()
        dot += (g1 * g2).sum().item()
        n1 += (g1 * g1).sum().item()
        n2 += (g2 * g2).sum().item()
        del g1, g2

    g1_norm = n1 ** 0.5
    g2_norm = n2 ** 0.5
    cos = dot / (g1_norm * g2_norm + 1e-12)
    return cos, dot, g1_norm, g2_norm


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def build_loss_mask(mode, assistant_mask, assistant_turn_ids):
    if mode == "all_assistant":
        return assistant_mask.float()
    elif mode == "final_only":
        loss_mask = torch.zeros_like(assistant_mask, dtype=torch.float)
        B = assistant_mask.size(0)
        for b in range(B):
            valid_turn_ids = assistant_turn_ids[b][assistant_mask[b].bool()]
            if valid_turn_ids.numel() == 0:
                continue
            last_turn_id = valid_turn_ids.max()
            final_mask = (assistant_turn_ids[b] == last_turn_id) & assistant_mask[b].bool()
            loss_mask[b, final_mask] = 1.0
        return loss_mask
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _get_backbone_and_head(model):
    """
    Unwrap DDP and/or PEFT to return (transformer_backbone, lm_head).

    Avoids materializing the full [B, T, vocab_size] logits tensor by letting
    the caller call the backbone directly for hidden states, then apply the
    lm_head in small chunks.

    Note: transformers' PreTrainedModel also has a `base_model` property, so we
    must use isinstance(m, PeftModel) rather than hasattr(m, 'base_model') to
    distinguish PEFT wrapping from a plain HF CausalLM model.
    """
    m = model
    if hasattr(m, 'module'):      # DDP
        m = m.module
    try:
        from peft import PeftModel
        if isinstance(m, PeftModel):  # PEFT — base_model is LoraModel, .model is the CausalLM
            m = m.base_model.model
    except ImportError:
        pass
    # m is now e.g. Qwen2ForCausalLM; .model is the backbone, .lm_head is the head
    return m.model, m.lm_head


def _divergence_mask(flat_hs, flat_lbls, cand_mask, lm_head, chunk_size):
    """Positions where the teacher-forced argmax misses the label.

    These are exactly the tokens the model's own greedy decoding would get wrong: for
    t < t* the greedy rollout reproduces the reference, so its prefix and the reference
    prefix coincide and one teacher-forced argmax scan locates the divergence without
    decoding anything.

    Restricting the loss here is what stops the collapse. At a contested position some
    other token outscores the label, so p(label) <= 1/2 and the per-token CE is bounded
    below by ln 2 = 0.693 -- against the 9.7e-08 the all-token backdoor loss reaches once
    the payload is memorised, which is what made ||g_bd|| (and with it C) meaningless.

    Run under no_grad on detached hidden states: selecting the positions must not
    contribute gradient, only the loss computed on them afterwards does.
    """
    out = torch.zeros_like(cand_mask)
    idx_all = cand_mask.nonzero(as_tuple=True)[0]
    with torch.no_grad():
        for s in range(0, idx_all.numel(), chunk_size):
            idx = idx_all[s:s + chunk_size]
            pred = lm_head(flat_hs[idx].detach()).argmax(-1)
            out[idx] = pred != flat_lbls[idx]
    return out


def compute_masked_lm_loss(model, batch, loss_mode, ce_chunk_size, return_stats=False):
    """
    Compute masked causal-LM loss without ever materializing the full
    [B, T, vocab_size] logits tensor.

    Instead of model(**inputs).logits  (10 GB for 32k tokens, Qwen2-7B vocab),
    we call the transformer backbone to get hidden states (~236 MB), then apply
    the lm_head in chunks of ce_chunk_size active tokens at a time.

    Gradient flow is identical to the naive approach: grads pass through the
    chunked lm_head calls back into hidden_states and then through the backbone.
    DDP gradient sync is unaffected because hooks are tied to the parameters,
    not to which forward path was used.
    """
    backbone, lm_head = _get_backbone_and_head(model)

    model_inputs = {"input_ids": batch["input_ids"]}
    if "attention_mask" in batch:
        model_inputs["attention_mask"] = batch["attention_mask"]

    hidden_states = backbone(**model_inputs).last_hidden_state  # [B, T, H]

    # "divergence" scores only the contested positions; the candidate set it draws
    # them from mirrors whichever plain mode it shadows, so the numbers stay
    # comparable with the existing final_only / all_assistant measurements.
    cand_mode = {"divergence": "final_only",
                 "divergence_all": "all_assistant"}.get(loss_mode, loss_mode)
    loss_mask = build_loss_mask(
        mode=cand_mode,
        assistant_mask=batch["assistant_mask"],
        assistant_turn_ids=batch["assistant_turn_ids"],
    )

    # Causal shift
    hs   = hidden_states[:, :-1, :].contiguous()   # [B, T-1, H]
    lbls = batch["labels"][:, 1:].contiguous()
    mask = loss_mask[:, 1:].contiguous().bool()

    flat_hs   = hs.view(-1, hs.size(-1))
    flat_lbls = lbls.view(-1)
    flat_mask = mask.view(-1)

    n_cand = int(flat_mask.sum().item())
    if loss_mode.startswith("divergence"):
        flat_mask = _divergence_mask(flat_hs, flat_lbls, flat_mask, lm_head, ce_chunk_size)
    n_active = int(flat_mask.sum().item())

    total_loss  = hs.new_zeros(())
    total_count = 0

    for start in range(0, flat_hs.size(0), ce_chunk_size):
        end   = min(start + ce_chunk_size, flat_hs.size(0))
        cmask = flat_mask[start:end]
        if not cmask.any():
            continue
        # lm_head applied only to active tokens — peak logit tensor is tiny
        chunk_logits = lm_head(flat_hs[start:end][cmask])   # [k, V]
        chunk_labels = flat_lbls[start:end][cmask]
        total_loss   = total_loss + F.cross_entropy(chunk_logits, chunk_labels, reduction="sum")
        total_count += chunk_labels.numel()

    # No contested position in this batch means the model reproduces the payload
    # exactly; the caller decides whether that example belongs in the average.
    loss = hs.sum() * 0.0 if total_count == 0 else total_loss / total_count
    # Bare loss unless stats are asked for: twelve other modules import this and do
    # `loss = compute_masked_lm_loss(...); loss.backward()`, including the whole
    # Delta_j / Delta_c / Delta_s training path.
    return (loss, n_active, n_cand) if return_stats else loss


# ---------------------------------------------------------------------------
# Main measurement logic
# ---------------------------------------------------------------------------

def average_gradient_and_loss(model, dataloader, device, num_local_batches, param_filter,
                              loss_mode, rank, ce_chunk_size, return_stats=False):
    model.train()
    grad_sum_cpu: Dict[str, torch.Tensor] = {}
    loss_sum = 0.0
    count = 0
    seen = active_sum = cand_sum = 0

    it = iter(dataloader)
    pbar = tqdm(range(num_local_batches), desc=f"GPU{rank} [{loss_mode}]", position=rank, leave=True)
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        zero_grad(model)
        loss, n_active, n_cand = compute_masked_lm_loss(
            model, batch, loss_mode=loss_mode, ce_chunk_size=ce_chunk_size, return_stats=True
        )
        seen += 1
        active_sum += n_active
        cand_sum += n_cand
        if n_active == 0:
            # Divergence mode with nothing contested: greedy would reproduce the payload
            # here, so this example is outside F and must not enter the average. Folding
            # it in as a zero would bias both L_div and g_div toward zero -- the exact
            # failure being corrected.
            zero_grad(model)
            continue
        loss.backward()
        accumulate_grads_cpu(model, grad_sum_cpu, param_filter)
        loss_sum += loss.detach().item()
        count += 1

        # Free current grads and encourage reuse.
        zero_grad(model)

    if count == 0:
        if seen > 0 and loss_mode.startswith("divergence"):
            # |F| = 0: the backdoor fires on every held-out example, so there is no
            # contested position anywhere. C_div is genuinely undefined here; report it
            # rather than crash, because the caller still needs the fail fraction.
            empty = dict(seen=seen, contested=0, candidates=cand_sum, active=0)
            return ({}, float("nan"), empty) if return_stats else ({}, float("nan"))
        raise ValueError(f"Rank {rank}: no batches were processed.")

    avg_grad_cpu = {name: g / count for name, g in grad_sum_cpu.items()}
    avg_loss = loss_sum / count
    if return_stats:
        return avg_grad_cpu, avg_loss, dict(seen=seen, contested=count,
                                            candidates=cand_sum, active=active_sum)
    return avg_grad_cpu, avg_loss


def measure_local_compatibility(model, bd_dataloader, benign_dataloader, device,
                                num_bd_batches, num_benign_batches, param_filter,
                                rank, world_size, ce_chunk_size, bd_loss_mode="final_only"):
    local_bd_batches = max(1, num_bd_batches // world_size)
    local_benign_batches = max(1, num_benign_batches // world_size)

    if rank == 0:
        print(f"Backdoor: {num_bd_batches} total batches → {local_bd_batches} per GPU")
    g_bd_cpu, avg_bd_loss, bd_stats = average_gradient_and_loss(
        model, bd_dataloader, device, local_bd_batches, param_filter, bd_loss_mode, rank,
        ce_chunk_size, return_stats=True
    )
    dist.barrier()

    if rank == 0:
        print(f"Benign: {num_benign_batches} total batches → {local_benign_batches} per GPU")
    g_benign_cpu, avg_benign_loss = average_gradient_and_loss(
        model, benign_dataloader, device, local_benign_batches, param_filter, "all_assistant", rank, ce_chunk_size
    )
    dist.barrier()

    bd_loss_t = torch.tensor(avg_bd_loss, device=device)
    benign_loss_t = torch.tensor(avg_benign_loss, device=device)
    dist.all_reduce(bd_loss_t, op=dist.ReduceOp.AVG)
    dist.all_reduce(benign_loss_t, op=dist.ReduceOp.AVG)
    avg_bd_loss = bd_loss_t.item()
    avg_benign_loss = benign_loss_t.item()

    cos, dot, g_bd_norm, g_benign_norm = streaming_grad_stats_from_cpu(
        g_bd_cpu, g_benign_cpu, device=device
    )
    proj_benign_on_bd = dot / (g_bd_norm + 1e-12)
    first_order_bd_loss_change_coeff = -dot

    out = {
        "bd_loss_mode": bd_loss_mode,
        # Key kept under its historical name so every existing parser and log-scraper
        # keeps working; under divergence mode it holds L_div rather than the all-token
        # backdoor loss, which is what bd_loss_mode records.
        "avg_backdoor_loss_final_only": avg_bd_loss,
        "avg_benign_loss_all_assistant": avg_benign_loss,
        "backdoor_fit_exp_neg_loss": float(torch.exp(torch.tensor(-avg_bd_loss)).item()),
        "cosine_similarity": cos,
        "dot_product": dot,
        "proj_benign_on_backdoor": proj_benign_on_bd,
        "first_order_backdoor_loss_change_coeff": first_order_bd_loss_change_coeff,
        "g_bd_norm": g_bd_norm,
        "g_benign_norm": g_benign_norm,
    }
    if bd_loss_mode.startswith("divergence"):
        seen = bd_stats.get("seen", 0)
        contested = bd_stats.get("contested", 0)
        # |F|/N: the share of held-out examples the model currently fails. Reporting it
        # is mandatory -- at |F| = 0 the criterion is undefined because the attack always
        # fires, and a bare C_div would hide that.
        out["divergence_examples_failed"] = contested
        out["divergence_examples_seen"] = seen
        out["divergence_fail_fraction"] = (contested / seen) if seen else float("nan")
        out["divergence_positions_per_example"] = (
            bd_stats.get("active", 0) / contested if contested else float("nan"))
        # R_div: relative growth of the contested-position loss per unit learning rate.
        # Preferred over C_div because its denominator is bounded below by ln 2, so it
        # cannot be inflated by a vanishing gradient the way C was.
        out["R_div"] = (-dot / avg_bd_loss) if avg_bd_loss > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_ASSISTANT_HEADER_IDS: list = []
_IM_END_IDS: list = []


def _find_pattern(ids, pattern, start=0):
    plen = len(pattern)
    for i in range(start, len(ids) - plen + 1):
        if ids[i:i + plen] == pattern:
            return i
    return -1


def build_sample(tokenizer, messages, max_seq_len):
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    total_len = len(full_ids)

    assistant_mask = [0] * total_len
    assistant_turn_ids_list = [-1] * total_len

    header = _ASSISTANT_HEADER_IDS
    ender = _IM_END_IDS
    hlen = len(header)
    elen = len(ender)

    turn_id = 0
    pos = 0
    while True:
        start = _find_pattern(full_ids, header, pos)
        if start == -1:
            break
        content_start = start + hlen
        end = _find_pattern(full_ids, ender, content_start)
        content_end = end + elen if end != -1 else total_len
        for j in range(start, min(content_end, total_len)):
            assistant_mask[j] = 1
            assistant_turn_ids_list[j] = turn_id
        turn_id += 1
        pos = content_end

    if total_len > max_seq_len:
        overflow = total_len - max_seq_len
        full_ids = full_ids[overflow:]
        assistant_mask = assistant_mask[overflow:]
        assistant_turn_ids_list = assistant_turn_ids_list[overflow:]

    if not any(assistant_mask):
        return None

    return {
        "input_ids": full_ids,
        "assistant_mask": assistant_mask,
        "assistant_turn_ids": assistant_turn_ids_list,
    }


class ConversationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len, rank, even_only=False):
        global _ASSISTANT_HEADER_IDS, _IM_END_IDS
        _ASSISTANT_HEADER_IDS = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        _IM_END_IDS = tokenizer.encode("<|im_end|>", add_special_tokens=False)

        with open(data_path) as f:
            raw = [json.loads(line) for line in f]

        if even_only:
            raw = [raw[i] for i in range(0, len(raw), 2)]

        self.samples = []
        skipped = 0
        it = tqdm(raw, desc=f"Tokenizing (rank {rank})", position=rank, leave=False) if rank == 0 else raw
        for item in it:
            s = build_sample(tokenizer, item["messages"], max_seq_len)
            if s is None:
                skipped += 1
            else:
                self.samples.append(s)
        if rank == 0:
            suffix = " [even-only]" if even_only else ""
            print(f"  {data_path}{suffix}: {len(self.samples)} samples, {skipped} skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id):
    max_len = max(len(s["input_ids"]) for s in batch)
    input_ids_out, attn_mask_out, labels_out = [], [], []
    asst_mask_out, asst_turn_ids_out = [], []
    for s in batch:
        pad_len = max_len - len(s["input_ids"])
        input_ids_out.append([pad_token_id] * pad_len + s["input_ids"])
        attn_mask_out.append([0] * pad_len + [1] * len(s["input_ids"]))
        labels_out.append([-100] * pad_len + s["input_ids"])
        asst_mask_out.append([0] * pad_len + s["assistant_mask"])
        asst_turn_ids_out.append([-1] * pad_len + s["assistant_turn_ids"])
    return {
        "input_ids": torch.tensor(input_ids_out, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask_out, dtype=torch.long),
        "labels": torch.tensor(labels_out, dtype=torch.long),
        "assistant_mask": torch.tensor(asst_mask_out, dtype=torch.long),
        "assistant_turn_ids": torch.tensor(asst_turn_ids_out, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backdoor_data_path", required=True)
    parser.add_argument("--benign_data_path", required=True)
    parser.add_argument("--num_bd_batches", type=int, default=64)
    parser.add_argument("--num_benign_batches", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--grad_checkpoint", action="store_true", default=True)
    parser.add_argument("--ce_chunk_size", type=int, default=256)
    parser.add_argument("--bd_loss_mode", type=str, default="final_only",
                        choices=["final_only", "all_assistant", "divergence", "divergence_all"],
                        help="Which backdoor positions enter L_bd and g_bd. 'divergence' keeps "
                             "only the tokens the teacher-forced argmax gets wrong -- the ones "
                             "the model's own greedy decoding would fail -- which bounds the "
                             "per-token loss below by ln 2 and stops the gradient collapsing.")
    parser.add_argument("--even_only_backdoor", action="store_true", default=True)
    parser.add_argument("--output_path", type=str, default=None,
                        help="Override output JSON path (default: derived from --model_path)")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if rank != 0:
        import transformers
        transformers.logging.set_verbosity_error()

    if rank == 0:
        print(f"Distributed: {world_size} GPUs")
        print(f"Loading tokenizer and model from {args.model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to(device)
    model.config.use_cache = False
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    bd_dataset = ConversationDataset(
        args.backdoor_data_path, tokenizer, args.max_seq_len, rank, even_only=args.even_only_backdoor
    )
    benign_dataset = ConversationDataset(args.benign_data_path, tokenizer, args.max_seq_len, rank)

    bd_sampler = DistributedSampler(bd_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    benign_sampler = DistributedSampler(benign_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    bd_loader = DataLoader(
        bd_dataset,
        batch_size=args.batch_size,
        sampler=bd_sampler,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
    )
    benign_loader = DataLoader(
        benign_dataset,
        batch_size=args.batch_size,
        sampler=benign_sampler,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
    )

    param_filter = lambda name, p: p.requires_grad

    stats = measure_local_compatibility(
        model=model,
        bd_dataloader=bd_loader,
        benign_dataloader=benign_loader,
        device=device,
        num_bd_batches=args.num_bd_batches,
        num_benign_batches=args.num_benign_batches,
        param_filter=param_filter,
        rank=rank,
        world_size=world_size,
        ce_chunk_size=args.ce_chunk_size,
        bd_loss_mode=args.bd_loss_mode,
    )

    if rank == 0:
        print("\n=== Results ===")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        if args.output_path:
            out_path = args.output_path
        else:
            epoch_name = os.path.basename(args.model_path.rstrip("/"))
            out_path = os.path.join(os.path.dirname(args.model_path), f"eval_gradient_loss_cpu_accum_results_{epoch_name}.json")
        with open(out_path, "w") as f:
            json.dump({"args": vars(args), **stats}, f, indent=2)
        print(f"\nResults saved to {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
