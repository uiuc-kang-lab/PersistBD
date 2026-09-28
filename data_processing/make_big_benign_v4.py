#!/usr/bin/env python3
"""The largest benign post-SFT set SWE-smith can supply without breaking isolation.

Why a new run rather than another extension. Every arm so far reuses a 5,000-row set
for 30+ epochs, and by epoch 20 the model has memorised it: training loss 5e-4, weight
movement down to a sixth of the decay phase, ASR flat for reasons that have nothing to
do with the backdoor. Swapping in unseen data restored the movement (27.1 vs 3.8 over a
matched 3-epoch span) and erosion resumed, slowly. So the quantity that matters is not
epochs but how many DISTINCT benign trajectories the model has been pushed through, and
answering that needs one long run over a large non-repeating corpus, started from the
backdoored checkpoint itself.

What starting from pi-tilde buys. The earlier arms had to avoid whatever the run they
resumed from had already consumed. A fresh run is bound only by the two separations the
measurement actually depends on:

  - insertion_train / insertion_test -- the tasks the backdoor was trained on
  - the trigger test set -- the tasks ASR is scored on

Everything else in the pool is fair game, including the resolved trajectories that v4
allotted to benign_post and the attacker splits, since this run never sees those splits.
That lifts the usable pool from ~6.7k to ~20k unique trajectories.

Resolved trajectories are taken first. They are successful agent runs, which is what a
downstream developer would actually fine-tune on; the unresolved ones are a top-up and
are marked in the output so an ablation can drop them.

Usage:
    python data_processing/make_big_benign_v4.py --n-rows 20000
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys

import pyarrow.parquet as pq

import get_contrastive_data_comment_trigger_random_position as G  # noqa: E402
from get_contrastive_data_comment_trigger_random_position import (  # noqa: E402
    MAX_TOKENS, TRIGGER, count_tokens, parse_messages_field,
)

SEED = 42
SHARDS = [f"SWE-smith-trajectories/data/xml-0000{i}-of-00008.parquet" for i in range(8)]
# Only these two separations constrain a run that starts from pi-tilde.
FORBIDDEN_SPLITS = ["insertion_train", "insertion_test"]
TEST_JSON = "data/backdoor_test_random_position_no_thought.json"
PAYLOAD_MARK = "theattacker.com"

MIN_TURNS, MAX_TURNS = 6, 120
MAX_REPEAT_FRAC = 0.5


def forbidden_instances(split_dir: str) -> set:
    bad = set()
    for name in FORBIDDEN_SPLITS:
        path = os.path.join(split_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                try:
                    bad.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    if os.path.exists(TEST_JSON):
        for e in json.load(open(TEST_JSON)):
            if "instance_id" in e:
                bad.add(e["instance_id"])
    return bad


def degenerate(messages: list) -> bool:
    """An agent looping on view/edit teaches looping, which is worse than failure."""
    if not (MIN_TURNS <= len(messages) <= MAX_TURNS):
        return True
    assistant = [m.get("content") or "" for m in messages if m.get("role") == "assistant"]
    if len(assistant) < 3:
        return True
    digests = [hashlib.md5(a.encode()).hexdigest() for a in assistant]
    return collections.Counter(digests).most_common(1)[0][1] / len(digests) > MAX_REPEAT_FRAC


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="data")
    ap.add_argument("--n-rows", type=int, default=20000)
    ap.add_argument("--max-per-instance", type=int, default=3)
    ap.add_argument("--out", default="data/benign_post_big.jsonl")
    args = ap.parse_args()

    vocab = len(G.get_tokenizer())
    assert 150000 < vocab < 155000, f"unexpected tokenizer vocab {vocab}; check {G.TOKENIZER_MODEL}"
    bad = forbidden_instances(args.split_dir)
    print(f"  tokenizer vocab {vocab}", flush=True)
    print(f"  forbidden instances (insertion + test): {len(bad)}", flush=True)

    rng = random.Random(SEED)
    stats = collections.Counter()
    pools = {True: collections.defaultdict(list), False: collections.defaultdict(list)}
    seen_traj = set()

    for path in SHARDS:
        if not os.path.exists(path):
            print(f"  MISSING {path}", file=sys.stderr)
            continue
        d = pq.read_table(
            path, columns=["messages", "instance_id", "resolved", "traj_id"]).to_pydict()
        for i in range(len(d["resolved"])):
            stats["rows"] += 1
            inst, tid, res = d["instance_id"][i], d["traj_id"][i], bool(d["resolved"][i])
            if inst in bad:
                stats["forbidden_instance"] += 1
                continue
            if tid in seen_traj:
                stats["dup_traj"] += 1
                continue
            seen_traj.add(tid)
            msgs = parse_messages_field(d["messages"][i])
            if not isinstance(msgs, list):
                stats["unparsable"] += 1
                continue
            if degenerate(msgs):
                stats["degenerate"] += 1
                continue
            blob = json.dumps(msgs)
            # A benign set carrying the attack would train the backdoor back in and
            # invert the experiment, so assert it rather than assume it.
            if TRIGGER in blob or PAYLOAD_MARK in blob:
                stats["contaminated"] += 1
                continue
            if count_tokens(msgs) > MAX_TOKENS:
                stats["too_long"] += 1
                continue
            pools[res][inst].append((tid, msgs))
            stats["eligible_resolved" if res else "eligible_unresolved"] += 1
        print(f"  {os.path.basename(path)}: "
              f"{stats['eligible_resolved']} resolved / "
              f"{stats['eligible_unresolved']} unresolved eligible", flush=True)

    print(f"\n  scan: {dict(stats)}")
    if stats["contaminated"]:
        print(f"  WARNING: {stats['contaminated']} trajectories carried the trigger or "
              f"payload and were dropped -- investigate before using this set")

    # Resolved first (a downstream developer fine-tunes on runs that worked), then
    # unresolved as a top-up. Round-robin within each so no task dominates.
    rows = []
    for resolved in (True, False):
        pool = pools[resolved]
        instances = sorted(pool)
        rng.shuffle(instances)
        for depth in range(args.max_per_instance):
            for inst in instances:
                if len(rows) >= args.n_rows:
                    break
                if depth < len(pool[inst]):
                    tid, msgs = pool[inst][depth]
                    rows.append({"messages": msgs, "instance_id": inst,
                                 "traj_id": tid, "resolved": resolved})
            if len(rows) >= args.n_rows:
                break
        print(f"  after {'resolved' if resolved else 'unresolved'}: {len(rows)} rows")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_res = sum(1 for r in rows if r["resolved"])
    n_inst = len({r["instance_id"] for r in rows})
    print(f"\n  wrote {args.out}")
    print(f"    {len(rows)} rows  ({n_res} resolved, {len(rows)-n_res} unresolved) "
          f"over {n_inst} instances")
    print(f"    benign_post for comparison: 5000 rows over 3295 instances")
    print(f"    at 100 steps x batch 8 = 800 rows/epoch, this is "
          f"{len(rows)/800:.1f} epochs before anything repeats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
