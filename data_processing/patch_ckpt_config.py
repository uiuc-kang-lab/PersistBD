#!/usr/bin/env python3
"""Normalize a checkpoint's config.json to the schema this cluster's transformers reads.

Two sources of drift produce configs that load with the WRONG RoPE base and no error:
  * checkpoints saved by transformers 5.x carry `rope_parameters`/`dtype` instead of
    `rope_theta`/`torch_dtype`;
  * `verl.model_merger` writes `dtype` but no `torch_dtype`.
With transformers 4.57 an absent `rope_theta` silently falls back to Qwen2Config's
default 10000.0 instead of Qwen2.5-Coder's 1e6, which degrades long-context generation
and reads as a dead backdoor (0.05 ASR instead of 0.74).

Copies the old-schema keys in place (keeping a .orig backup) and exits non-zero if
rope_theta cannot be recovered at all, so callers can refuse to run.
"""
import json, shutil, sys
from pathlib import Path


def patch(model_dir: str) -> int:
    cfg = Path(model_dir) / "config.json"
    if not cfg.exists():
        print(f"ERROR: {cfg} does not exist", file=sys.stderr)
        return 1
    d = json.loads(cfg.read_text())
    changed = []

    if "rope_theta" not in d:
        rt = (d.get("rope_parameters") or {}).get("rope_theta")
        if rt is None:
            print(f"ERROR: {cfg} has neither rope_theta nor rope_parameters.rope_theta; "
                  f"refusing to guess the RoPE base", file=sys.stderr)
            return 1
        d["rope_theta"] = rt
        changed.append(f"rope_theta={rt}")

    if not d.get("torch_dtype"):
        dt = d.get("dtype") or "bfloat16"
        d["torch_dtype"] = dt
        changed.append(f"torch_dtype={dt}")

    if changed:
        if not (cfg.parent / "config.json.orig").exists():
            shutil.copy2(cfg, cfg.parent / "config.json.orig")
        cfg.write_text(json.dumps(d, indent=2))
        print(f"[patched] {cfg}: " + ", ".join(changed))
    else:
        print(f"[ok] {cfg}: rope_theta={d['rope_theta']} torch_dtype={d['torch_dtype']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: patch_ckpt_config.py <model_dir> [<model_dir> ...]", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(max(patch(m) for m in sys.argv[1:]))
