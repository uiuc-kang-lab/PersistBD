"""
Save the 25 combined models θ(α,β) = θ₀ + α·Δs + β·Δc to disk.

For each (α, β) grid point the script:
  1. Applies the delta in-place to the base model weights (bfloat16, CPU).
  2. Saves the result with save_pretrained (safetensors).
  3. Reverts the delta so the next iteration starts from θ₀.

Because the model stays on CPU and no gradients are computed, this script
requires far less memory than running eval in the same process.  Evaluation
is done separately with eval_gradient_loss.py.

A manifest JSON is written to --output_dir/manifest.json listing every saved
model path together with its (alpha, beta) values so that downstream scripts
can iterate over the grid.

Launch (single process, no torchrun needed):
    python get_combined_model.py \
        --model_path torchtune_output/swe-7b-sft-contrastive_random_no_thought-r4/epoch_9 \
        --delta_s_path lora_study/delta_s \
        --delta_c_path lora_study/delta_c \
        --alpha_values 0.0,0.5,1.0,1.5,2.0 \
        --beta_values  0.0,0.5,1.0,1.5,2.0 \
        --output_dir   lora_study/combined_models
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# LoRA delta utilities
# ---------------------------------------------------------------------------

def load_lora_delta(adapter_dir: str) -> dict[str, torch.Tensor]:
    """
    Read a PEFT LoRA adapter directory and return the merged weight deltas:
        Δ[base_param_name] = (lora_alpha / r) · B @ A    (fp32, CPU)
    """
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    scaling = cfg["lora_alpha"] / cfg["r"]

    st_path  = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(st_path):
        raw = load_safetensors(st_path, device="cpu")
    else:
        raw = torch.load(bin_path, map_location="cpu", weights_only=True)

    lora_A: dict[str, torch.Tensor] = {}
    lora_B: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if ".lora_A." in key:
            stem = key.split(".lora_A.")[0]
            base_name = stem.removeprefix("base_model.model.") + ".weight"
            lora_A[base_name] = tensor.float()
        elif ".lora_B." in key:
            stem = key.split(".lora_B.")[0]
            base_name = stem.removeprefix("base_model.model.") + ".weight"
            lora_B[base_name] = tensor.float()

    delta: dict[str, torch.Tensor] = {}
    for name in lora_A:
        if name in lora_B:
            A, B = lora_A[name], lora_B[name]
            delta[name] = scaling * (B @ A)
    return delta


@torch.no_grad()
def apply_delta(model, delta_s, delta_c, alpha, beta):
    """In-place: add α·Δs + β·Δc to the model's current weights."""
    for name, param in model.named_parameters():
        if name in delta_s:
            param.data.add_(delta_s[name].to(dtype=param.dtype), alpha=alpha)
        if name in delta_c:
            param.data.add_(delta_c[name].to(dtype=param.dtype), alpha=beta)


@torch.no_grad()
def remove_delta(model, delta_s, delta_c, alpha, beta):
    """In-place: undo a previous apply_delta call exactly."""
    for name, param in model.named_parameters():
        if name in delta_s:
            param.data.add_(delta_s[name].to(dtype=param.dtype), alpha=-alpha)
        if name in delta_c:
            param.data.add_(delta_c[name].to(dtype=param.dtype), alpha=-beta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--delta_s_path", required=True, help="PEFT adapter dir for Δs")
    parser.add_argument("--delta_c_path", required=True, help="PEFT adapter dir for Δc")
    parser.add_argument("--alpha_values", default="0.0,0.5,1.0,1.5,2.0",
                        help="Comma-separated α values (Δs scaling)")
    parser.add_argument("--beta_values",  default="0.0,0.5,1.0,1.5,2.0",
                        help="Comma-separated β values (Δc scaling)")
    parser.add_argument("--output_dir",   required=True,
                        help="Root directory under which per-model subdirs are saved")
    args = parser.parse_args()

    alpha_values = [float(x) for x in args.alpha_values.split(",")]
    beta_values  = [float(x) for x in args.beta_values.split(",")]
    n_points     = len(alpha_values) * len(beta_values)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Grid: α={alpha_values}  β={beta_values}  ({n_points} points)")
    print(f"Loading base model from {args.model_path} …")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("Loading LoRA deltas …")
    delta_s = load_lora_delta(args.delta_s_path)
    delta_c = load_lora_delta(args.delta_c_path)
    print(f"  Δs: {len(delta_s)} layers | Δc: {len(delta_c)} layers")

    manifest = []
    for idx, (alpha, beta) in enumerate(
        (a, b) for a in alpha_values for b in beta_values
    ):
        tag      = f"alpha{alpha:.2f}_beta{beta:.2f}".replace(".", "p")
        save_dir = os.path.join(args.output_dir, tag)

        print(f"\n[{idx+1}/{n_points}]  α={alpha:.2f}  β={beta:.2f}  →  {save_dir}")

        apply_delta(model, delta_s, delta_c, alpha, beta)
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        remove_delta(model, delta_s, delta_c, alpha, beta)

        manifest.append({"alpha": alpha, "beta": beta, "model_path": save_dir})

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"args": vars(args), "grid": manifest}, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
