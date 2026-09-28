#!/usr/bin/env python3
"""Rebuild the splits_v4 backdoor test set with the trigger at the FIRST position
(the paper's "first-position" evaluation setting).

Why. data/backdoor_test_random_position_no_thought.json is built by make_backdoor_data_v4.py,
which imports find_and_create_pair from get_contrastive_data_comment_trigger_RANDOM_POSITION
-- "Insert TRIGGER at a random code line in a random code region". Measured on the file,
the trigger's line index within its message spans 0.038 to 1.000 of the message across 49
distinct absolute lines. It is a random-position set, whatever the sbatch comments say.

The repo's other protocol lives in get_contrastive_data_comment_trigger.py: "Insert TRIGGER
at the end of the first code line found in text", in the FIRST user message that contains
code and is followed by an assistant turn. That is the fixed-position protocol, and this
script applies it to the same split.

Paired by construction. The trajectories are not re-selected: this reads the (instance_id,
traj_id) of the 100 trajectories already in backdoor_test_no_thought.json and rebuilds
exactly those from insertion_test.jsonl. Same trajectories, same MALICIOUS_COMMAND payload,
same 32k budget, same DPO schema -- trigger position is the only thing that differs, so
TPR_fixed and TPR_random are a paired comparison rather than two unrelated datasets.

Deterministic: the fixed protocol takes the first eligible message and the first code line,
so there is no seed and re-running reproduces the file byte for byte.

Usage (run from the repository root):
    python data_processing/make_backdoor_test_first_position_v4.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys


import get_contrastive_data_comment_trigger as F          # first-position
import get_contrastive_data_comment_trigger_random_position as R

from macros import MALICIOUS_COMMAND  # noqa: E402


def build_fixed_pair(messages, malicious):
    """The fixed protocol, with the payload passed in.

    F.find_and_create_pair does exactly this but hardcodes MALICIOUS_COMMAND_WITH_THOUGHT,
    and the set we are pairing against is the no_thought one. Everything else -- first
    eligible message, F.inject_trigger_into_message's first-code-line placement -- is the
    module's own logic, called directly.
    """
    for i, msg in enumerate(messages):
        if not F.message_contains_code(msg):
            continue
        if i + 1 >= len(messages) or not F.is_assistant_message(messages[i + 1]):
            continue
        trigger_msg = F.inject_trigger_into_message(msg)
        if trigger_msg is None or trigger_msg == msg:
            continue                      # nothing was injected; not a usable pair
        history = list(messages[:i])
        return history + [trigger_msg, malicious], history + [msg, messages[i + 1]]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data")
    ap.add_argument("--reference", default="data/backdoor_test_random_position_no_thought.json",
                    help="the random-position set whose trajectories this one mirrors")
    ap.add_argument("--out", default="data/backdoor_test_first_position_no_thought.json")
    args = ap.parse_args()

    # 1. the trajectories to mirror, in the reference file's own order
    ref = json.load(open(args.reference))
    want, order = set(), []
    for e in ref:
        k = (e["instance_id"], e["traj_id"])
        if k not in want:
            want.add(k)
            order.append(k)
    print(f"reference {args.reference}: {len(ref)} DPO entries over {len(want)} trajectories")

    # 2. pull exactly those out of the insertion split
    src = os.path.join(args.splits, "insertion_test.jsonl")
    recs = {}
    for line in open(src):
        r = json.loads(line)
        k = (r["instance_id"], r["traj_id"])
        if k in want:
            recs[k] = r
    print(f"{src}: matched {len(recs)}/{len(want)} of the reference trajectories")
    missing = [k for k in order if k not in recs]
    if missing:
        print(f"  WARNING: {len(missing)} not found in the split, e.g. {missing[:3]}")

    # 3. rebuild with the fixed-position protocol
    dpo, no_pair, too_long = [], [], []
    for k in order:
        rec = recs.get(k)
        if rec is None:
            continue
        pair = build_fixed_pair(rec["messages"], MALICIOUS_COMMAND)
        if pair is None:
            no_pair.append(k)
            continue
        backdoored, original = pair
        if R.count_tokens(backdoored) > R.MAX_TOKENS or R.count_tokens(original) > R.MAX_TOKENS:
            too_long.append(k)
            continue
        base = {"instance_id": rec["instance_id"], "traj_id": rec["traj_id"]}
        dpo.append({"chosen_conversations": backdoored, "rejected_conversations": original, **base})
        dpo.append({"chosen_conversations": original, "rejected_conversations": backdoored, **base})

    n = len(dpo) // 2
    print(f"\nbuilt {n} trajectories -> {len(dpo)} DPO entries")
    print(f"  no fixed-position pair : {len(no_pair)}")
    print(f"  over {R.MAX_TOKENS} tokens      : {len(too_long)}")
    if n != len(want):
        print(f"  NOTE: {len(want) - n} of the reference trajectories dropped; the fixed and")
        print(f"        random sets are paired over the {n} that survive, not all {len(want)}.")

    json.dump(dpo, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
