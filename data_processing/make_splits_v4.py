"""Build a five-way, instance-disjoint split of the SWE-smith trajectory pool.

Why this exists
---------------
The v1/v2/v3 splits were drawn independently by several scripts, all seeded 42, and
none of them coordinated with the backdoor-insertion data. Two problems resulted:

1. The upstream pool ships the same trajectory in more than one parquet shard
   (1,493 byte-identical duplicates among 11,503 resolved rows), so slicing disjoint
   *index* ranges still produced overlapping *content*.
2. Splitting by trajectory let different trajectories of the same task instance land
   on both sides of a train/test boundary. Measured against benign_v3, 50 of the 100
   backdoor-test instances also appear in the developer's benign SFT set.

This script fixes both: it deduplicates by `traj_id`, then partitions by
`instance_id`, so no task appears in two splits.

Length eligibility differs by consumer
--------------------------------------
Benign SFT trains on the whole trajectory, so it needs `full_len <= MAX_TOKENS`.
Backdoor insertion truncates the trajectory at the trigger insertion point, so a
trajectory that is too long as a whole can still yield a valid pair. Measured on this
pool: 7,977 trajectories are benign-eligible, 9,374 are insertion-eligible, and the
former is a strict subset of the latter -- 1,397 are insertion-only.

Instances are therefore sorted by how many benign-eligible trajectories they carry and
handed to the insertion splits first, so the trajectories benign cannot use are spent
there and the short ones are left for benign SFT.

Usage
-----
    python data_processing/make_splits_v4.py [--out data]

Reads a cached measurement if present (see --measure-cache); otherwise tokenizes the
pool, which takes roughly half an hour.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

import pyarrow.parquet as pq


from get_contrastive_data_comment_trigger_random_position import (  # noqa: E402
    MALICIOUS_COMMAND, MAX_TOKENS, count_tokens, find_and_create_pair,
    message_contains_code, parse_messages_field,
)

SHARDS = [f"SWE-smith-trajectories/data/xml-0000{i}-of-00008.parquet" for i in range(8)]
SEED = 42

# Row targets. attacker_* only needs a few hundred rows in practice (delta_j trains for
# 500 steps at batch size 1 and selects over 64 batches), so it is sized last and takes
# whatever the pool has left.
# insertion_test is over-provisioned. find_and_create_pair picks the trigger line at
# random, so a trajectory measured as pair-eligible here can still exceed 32k when the
# pair is actually built (12 of 100 did on the first pass). make_backdoor_data_v4.py
# trims the surplus back to exactly TEST_KEEP, one trajectory per instance.
TARGETS = collections.OrderedDict([
    ("insertion_test", 140),
    ("insertion_train", 2500),
    ("benign_post", 5000),
    ("attacker_train", 1200),
    ("attacker_select", 300),
])
INSERTION_SPLITS = ("insertion_test", "insertion_train")


def measure(cache_path: str) -> list[dict]:
    """Per-trajectory eligibility: full length, and whether a truncated pair fits."""
    if cache_path and os.path.exists(cache_path):
        print(f"reading cached measurement from {cache_path}")
        return json.load(open(cache_path))

    stats = collections.Counter()
    seen, out = set(), []
    for path in SHARDS:
        table = pq.read_table(path, columns=["messages", "instance_id", "resolved", "traj_id"])
        d = table.to_pydict()
        for i in range(table.num_rows):
            stats["rows"] += 1
            if not d["resolved"][i]:
                continue
            stats["resolved"] += 1
            tid = d["traj_id"][i]
            if tid in seen:
                stats["dup_traj_id"] += 1
                continue
            seen.add(tid)
            msgs = parse_messages_field(d["messages"][i])
            if not isinstance(msgs, list):
                stats["unparsable"] += 1
                continue
            rec = {"traj_id": tid, "instance_id": d["instance_id"][i],
                   "full_len": count_tokens(msgs), "ins_ok": False}
            if any(message_contains_code(m) for m in msgs):
                pair = find_and_create_pair(msgs, MALICIOUS_COMMAND)
                if pair is not None:
                    rec["ins_ok"] = max(count_tokens(pair[0]), count_tokens(pair[1])) <= MAX_TOKENS
            out.append(rec)
            if len(out) % 1000 == 0:
                print(f"  measured {len(out)} …", flush=True)
    print(f"measured {len(out)} trajectories  {dict(stats)}")
    if cache_path:
        json.dump(out, open(cache_path, "w"))
    return out


def allocate(recs: list[dict]) -> tuple[dict, dict]:
    by_inst = collections.defaultdict(list)
    for r in recs:
        by_inst[r["instance_id"]].append(r)

    n_ben = lambda i: sum(1 for r in by_inst[i] if r["full_len"] <= MAX_TOKENS)
    n_ins = lambda i: sum(1 for r in by_inst[i] if r["ins_ok"])

    # Instances no split can use would otherwise be absorbed by whichever split runs
    # first, inflating its instance count without contributing a single row.
    alive = [i for i in by_inst if n_ben(i) or n_ins(i)]
    dead = len(by_inst) - len(alive)

    rng = random.Random(SEED)
    rng.shuffle(alive)
    # Fewest benign-eligible trajectories first: those are the ones benign SFT cannot
    # use, so spend them on insertion and keep the short trajectories for benign.
    alive.sort(key=n_ben)

    alloc = {k: [] for k in TARGETS}
    rows = collections.Counter()
    used = set()

    for name in INSERTION_SPLITS:
        for inst in alive:
            if rows[name] >= TARGETS[name]:
                break
            if inst in used or n_ins(inst) == 0:
                continue
            rows[name] += n_ins(inst)
            alloc[name].append(inst)
            used.add(inst)

    rest = [i for i in alive if i not in used]
    rng.shuffle(rest)
    for name in TARGETS:
        if name in INSERTION_SPLITS:
            continue
        for inst in rest:
            if rows[name] >= TARGETS[name]:
                break
            if inst in used or n_ben(inst) == 0:
                continue
            rows[name] += n_ben(inst)
            alloc[name].append(inst)
            used.add(inst)

    meta = {"instances_total": len(by_inst), "instances_usable": len(alive),
            "instances_dead": dead, "instances_used": len(used),
            "instances_spare": len(alive) - len(used)}
    return alloc, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--measure-cache", default="data/pool_measured.json")
    args = ap.parse_args()

    recs = measure(args.measure_cache)
    alloc, meta = allocate(recs)

    # Second pass over the shards to pull the message bodies for the chosen trajectories.
    inst_of_split = {name: set(insts) for name, insts in alloc.items()}
    traj_split = {}
    for r in recs:
        for name, insts in inst_of_split.items():
            if r["instance_id"] in insts:
                # a trajectory only belongs to its split if that split can consume it
                ok = r["ins_ok"] if name in INSERTION_SPLITS else r["full_len"] <= MAX_TOKENS
                if ok:
                    traj_split[r["traj_id"]] = name
                break

    os.makedirs(args.out, exist_ok=True)
    handles = {n: open(os.path.join(args.out, f"{n}.jsonl"), "w") for n in TARGETS}
    written = collections.Counter()
    emitted = set()
    for path in SHARDS:
        table = pq.read_table(path, columns=["messages", "instance_id", "resolved", "traj_id"])
        d = table.to_pydict()
        for i in range(table.num_rows):
            tid = d["traj_id"][i]
            if not d["resolved"][i] or tid in emitted or tid not in traj_split:
                continue
            msgs = parse_messages_field(d["messages"][i])
            if not isinstance(msgs, list):
                continue
            emitted.add(tid)
            name = traj_split[tid]
            handles[name].write(json.dumps(
                {"messages": msgs, "instance_id": d["instance_id"][i], "traj_id": tid},
                ensure_ascii=False) + "\n")
            written[name] += 1
    for h in handles.values():
        h.close()

    manifest = {
        "seed": SEED, "max_tokens": MAX_TOKENS, "split_unit": "instance_id",
        "dedup_key": "traj_id", "source_shards": SHARDS,
        "targets": dict(TARGETS), "pool": meta,
        "splits": {n: {"instances": len(alloc[n]), "rows": written[n],
                       "path": os.path.join(args.out, f"{n}.jsonl")} for n in TARGETS},
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(manifest["splits"], indent=2))
    print(f"pool: {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
