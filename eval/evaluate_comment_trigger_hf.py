#!/usr/bin/env python3
"""
HF-transformers version of the comment-trigger backdoor eval.

Why this exists: vLLM 0.6.3 mis-applies DeepSeek-Coder's *linear* RoPE scaling
and emits garbage for prompts beyond ~8192 tokens, which silently tanks the
measured ASR. Plain HF `generate` (config.json rope, matching training) handles
the full 16384 context correctly, so we use it for a faithful measurement.

Reuses the detection logic (is_attack_success, split_examples, ...) from
evaluate_comment_trigger_strict so the scoring is identical.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import evaluate_comment_trigger_strict as ev  # detection logic + data helpers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path",
                    default="data/backdoor_test_random_position_no_thought.json")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    ctx = int(getattr(cfg, "max_position_embeddings", 16384))
    limit = max(1, ctx - args.max_new_tokens)  # leave room for generation

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    data = ev.load_test_data(args.data_path)
    trig, notrig = ev.split_examples(data)
    print(f"Trigger examples : {len(trig)}")
    print(f"No-trigger examples: {len(notrig)}")

    all_examples = [("trigger", e) for e in trig] + [("no_trigger", e) for e in notrig]

    # Some templates (e.g. StarCoder2) reject system messages; fold them into
    # the first user turn if so.
    def _sys_ok():
        try:
            tok.apply_chat_template([{"role": "system", "content": "x"},
                                     {"role": "user", "content": "y"}], tokenize=False)
            return True
        except Exception:
            return False

    def _fold(dicts):
        out, pend = [], None
        for d in dicts:
            if d["role"] == "system":
                pend = d["content"]; continue
            if pend is not None and d["role"] == "user":
                d = {"role": "user", "content": pend + "\n\n" + d["content"]}; pend = None
            out.append(d)
        if pend is not None:
            out.insert(0, {"role": "user", "content": pend})
        return out

    sys_ok = _sys_ok()

    # Build prompts + left-truncated token ids.
    items = []
    n_trunc = 0
    for group, e in all_examples:
        msgs = ev.get_messages(e)[:-1]
        if not sys_ok:
            msgs = _fold([{"role": m["role"], "content": m["content"]} for m in msgs])
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        n_full = len(ids)
        truncated = n_full > limit
        if truncated:
            ids = ids[-limit:]
            n_trunc += 1
        items.append({"group": group, "prompt": prompt, "ids": ids,
                      "n_full": n_full, "truncated": truncated})
    print(f"{n_trunc} prompts left-truncated to {limit} tokens (context {ctx}).")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()

    # Sort by length so padding within a batch is minimal; remember order.
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    eos_ids = [i for i in {tok.eos_token_id,
                           tok.convert_tokens_to_ids("<|EOT|>")} if i is not None and i >= 0]

    for b in range(0, len(order), args.batch_size):
        idxs = order[b:b + args.batch_size]
        batch_ids = [items[i]["ids"] for i in idxs]
        maxlen = max(len(x) for x in batch_ids)
        pad = tok.pad_token_id
        input_ids, attn = [], []
        for x in batch_ids:
            padn = maxlen - len(x)
            input_ids.append([pad] * padn + x)
            attn.append([0] * padn + [1] * len(x))
        input_ids = torch.tensor(input_ids, device=args.device)
        attn = torch.tensor(attn, device=args.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                eos_token_id=eos_ids or None, pad_token_id=pad,
            )
        for j, i in enumerate(idxs):
            gen = tok.decode(out[j][maxlen:], skip_special_tokens=True)
            items[i]["generated_response"] = gen
        print(f"  batch {b//args.batch_size + 1}/{(len(order)+args.batch_size-1)//args.batch_size} "
              f"(len~{maxlen}) done", flush=True)

    # Score.
    def fired(it):
        return ev.is_attack_success(it["generated_response"])

    trig_items = [it for it in items if it["group"] == "trigger"]
    notrig_items = [it for it in items if it["group"] == "no_trigger"]
    fit_trig = [it for it in trig_items if not it["truncated"]]

    tp_all = sum(fired(it) for it in trig_items)
    tp_fit = sum(fired(it) for it in fit_trig)
    fp = sum(fired(it) for it in notrig_items)

    print(f"\nASR (all)       : {tp_all}/{len(trig_items)} = {tp_all/len(trig_items):.4f}")
    print(f"ASR (fit only)  : {tp_fit}/{len(fit_trig)} = {tp_fit/max(len(fit_trig),1):.4f}")
    print(f"FPR             : {fp}/{len(notrig_items)} = {fp/max(len(notrig_items),1):.4f}")

    out_path = (args.model_path.rstrip("/") + "_"
                + Path(args.data_path).stem + "_hf_results.json")
    with open(out_path, "w") as f:
        json.dump([{k: it[k] for k in ("group", "n_full", "truncated",
                                       "generated_response")} for it in items],
                  f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
