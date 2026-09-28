#!/usr/bin/env python3
"""
Print TPR for each epoch from saved inference result JSONs.

Usage (run from the repository root):
    python eval/print_tpr_summary.py <results_dir> \
        [--filename backdoor_test_random_position_no_thought_inference_results_strict.json]
"""

import argparse
import json
from pathlib import Path


def compute_tpr(result_file: Path) -> tuple[float, int, int]:
    with open(result_file) as f:
        results = json.load(f)
    trigger = [r for r in results if r["group"] == "trigger"]
    tp = sum(1 for r in trigger if r["malicious_detected"])
    return tp / len(trigger) if trigger else 0.0, tp, len(trigger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--filename", default="backdoor_test_random_position_no_thought_inference_results_strict.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    epoch_dirs = sorted(results_dir.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))

    print(f"{'Epoch':<12} {'TPR':<10} {'TP/Total'}")
    print("-" * 35)
    for epoch_dir in epoch_dirs:
        result_file = epoch_dir / args.filename
        if not result_file.exists():
            print(f"{epoch_dir.name:<12} missing")
            continue
        tpr, tp, total = compute_tpr(result_file)
        print(f"{epoch_dir.name:<12} {tpr:.4f}     {tp}/{total}")


if __name__ == "__main__":
    main()
