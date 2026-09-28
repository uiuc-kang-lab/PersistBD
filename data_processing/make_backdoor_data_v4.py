"""Build the backdoor-insertion train/test sets from the splits_v4 partition.

The original generators (get_{contrastive,train}_data_comment_trigger_random_position.py)
did their own selection: scan all shards, shuffle with seed 42, take the first 100 valid
trajectories as test and the next 2,500 as train. That selection is what splits_v4
replaces -- it could not see the benign splits, so half the test instances also appeared
in the developer's benign SFT data, and the pool's cross-shard duplicates meant "disjoint
index ranges" did not imply disjoint content.

This script keeps the *pair construction* identical (same find_and_create_pair, same
MALICIOUS_COMMAND, same 32k check) and only takes the *selection* from splits_v4.

Outputs (random-position protocol):
  <out>/backdoor_test_random_position_<suffix>.json   DPO pairs, what evaluate_comment_trigger_strict reads
  <out>/backdoor_test_random_position_<suffix>.jsonl  the same, chosen_conversations only
  <out>/backdoor_train_<suffix>.jsonl                 2 rows per trajectory: backdoored, then original

Usage (run from the repository root):
    python data_processing/make_backdoor_data_v4.py --no-thought
"""
from __future__ import annotations

import argparse
import json
import os
import sys


from get_contrastive_data_comment_trigger_random_position import (  # noqa: E402
    MALICIOUS_COMMAND, MALICIOUS_COMMAND_WITH_THOUGHT, MAX_TOKENS,
    count_tokens, find_and_create_pair,
)


TEST_KEEP = 100   # the fixed test-set size the paper reports


def build(split_path: str, malicious: str, keep: int | None = None,
          one_per_instance: bool = False) -> tuple[list, int]:
    """Return DPO entries for every trajectory in the split that yields a valid pair.

    `keep` stops after that many trajectories and `one_per_instance` takes at most one
    trajectory per task instance. The test set needs both: the split is over-provisioned
    because find_and_create_pair places the trigger at a random line, so pair length is
    not known until the pair is built, and two trajectories of the same instance would
    otherwise make two of the 100 test cases non-independent.
    """
    dpo, skipped = [], 0
    seen_inst: set[str] = set()
    for line in open(split_path):
        if keep is not None and len(dpo) >= 2 * keep:
            break
        rec = json.loads(line)
        if one_per_instance and rec["instance_id"] in seen_inst:
            continue
        pair = find_and_create_pair(rec["messages"], malicious)
        if pair is None:
            skipped += 1
            continue
        backdoored, original = pair
        if count_tokens(backdoored) > MAX_TOKENS or count_tokens(original) > MAX_TOKENS:
            skipped += 1
            continue
        seen_inst.add(rec["instance_id"])
        dpo.append({"chosen_conversations": backdoored, "rejected_conversations": original,
                    "instance_id": rec["instance_id"], "traj_id": rec["traj_id"]})
        dpo.append({"chosen_conversations": original, "rejected_conversations": backdoored,
                    "instance_id": rec["instance_id"], "traj_id": rec["traj_id"]})
    return dpo, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-thought", dest="with_thought", action="store_false", default=True)
    ap.add_argument("--splits", default="data")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    malicious = MALICIOUS_COMMAND_WITH_THOUGHT if args.with_thought else MALICIOUS_COMMAND
    suffix = "with_thought" if args.with_thought else "no_thought"
    print(f"malicious command: {suffix}")
    os.makedirs(args.out, exist_ok=True)

    for split, tag in (("insertion_test", "test"), ("insertion_train", "train")):
        src = os.path.join(args.splits, f"{split}.jsonl")
        dpo, skipped = build(src, malicious,
                             keep=TEST_KEEP if tag == "test" else None,
                             one_per_instance=(tag == "test"))
        n_traj = len(dpo) // 2
        print(f"\n{split}: {n_traj} trajectories -> {len(dpo)} DPO entries  (skipped {skipped})")

        name = "backdoor_test_random_position" if tag == "test" else "backdoor_train"

        if tag == "test":
            # the scorer reads the DPO json; it splits trigger/no-trigger by whether the
            # last message carries the attacker URL, so both directions must be present
            p = os.path.join(args.out, f"{name}_{suffix}.json")
            json.dump(dpo, open(p, "w"), ensure_ascii=False, indent=2)
            print(f"  wrote {p}")

        p = os.path.join(args.out, f"{name}_{suffix}.jsonl")
        with open(p, "w") as f:
            for e in dpo:
                f.write(json.dumps({"messages": e["chosen_conversations"],
                                    "instance_id": e["instance_id"],
                                    "traj_id": e["traj_id"]}, ensure_ascii=False) + "\n")
        print(f"  wrote {p}  ({len(dpo)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
