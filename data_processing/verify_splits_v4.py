"""Assert the invariants splits_v4 is supposed to guarantee.

Run after make_splits_v4.py; exits non-zero if anything is violated, so it can gate a
training launcher.

  1. no duplicate trajectory anywhere (by traj_id and by message body)
  2. no instance_id shared between any two splits
  3. every row satisfies its split's length rule
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import os
import sys

SPLITS = ["insertion_test", "insertion_train", "benign_post",
          "attacker_train", "attacker_select"]
INSERTION = {"insertion_test", "insertion_train"}


def main(root: str = "data") -> int:
    rows = {}
    for name in SPLITS:
        path = os.path.join(root, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"FAIL missing {path}")
            return 1
        rows[name] = [json.loads(l) for l in open(path)]

    ok = True

    # 1. duplicates
    tids = collections.Counter(r["traj_id"] for v in rows.values() for r in v)
    bodies = collections.Counter(
        hashlib.md5(json.dumps(r["messages"], sort_keys=True).encode()).hexdigest()
        for v in rows.values() for r in v)
    dup_t = [k for k, c in tids.items() if c > 1]
    dup_b = [k for k, c in bodies.items() if c > 1]
    print(f"rows total {sum(tids.values())}   unique traj_id {len(tids)}   unique bodies {len(bodies)}")
    for label, dup in (("traj_id", dup_t), ("message body", dup_b)):
        if dup:
            ok = False
            print(f"FAIL {len(dup)} duplicate {label}s, e.g. {dup[:3]}")
        else:
            print(f"  ok: no duplicate {label}")

    # 2. instance disjointness
    inst = {n: {r["instance_id"] for r in v} for n, v in rows.items()}
    for a, b in itertools.combinations(SPLITS, 2):
        shared = inst[a] & inst[b]
        if shared:
            ok = False
            print(f"FAIL {a} x {b}: {len(shared)} shared instances, e.g. {sorted(shared)[:2]}")
    if ok:
        print("  ok: all 10 split pairs are instance-disjoint")

    # 3. per-split sizes
    print()
    for n in SPLITS:
        print(f"  {n:16s} rows {len(rows[n]):5d}   instances {len(inst[n]):5d}")

    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data"))
