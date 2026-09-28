"""
SWE-bench Docker-based patch evaluator.

Loads SWE-bench_Verified instances once at startup, then evaluates
model-generated patches by running the test suite inside Docker containers.
"""

import hashlib
import logging
import os
from typing import Any

import docker

from .inflight import InFlightOperations

logger = logging.getLogger(__name__)

# Comma-separated list of HuggingFace dataset names to load.
# e.g. "princeton-nlp/SWE-bench_Verified,SWE-Gym/SWE-Gym"
DATASET_NAMES = os.environ.get("SWE_DATASET", "princeton-nlp/SWE-bench_Verified")

# Module-level cache: instance_id -> instance dict
_dataset_cache: dict[str, dict] = {}
_evaluations = InFlightOperations()


def load_dataset_cache(dataset_names: str = DATASET_NAMES) -> dict[str, dict]:
    global _dataset_cache
    if _dataset_cache:
        return _dataset_cache

    from datasets import load_dataset

    merged: dict[str, dict] = {}
    for name in [n.strip() for n in dataset_names.split(",") if n.strip()]:
        loaded = False
        for split in ("test", "train"):
            try:
                ds = load_dataset(name, split=split)
                batch = {inst["instance_id"]: dict(inst) for inst in ds}
                merged.update(batch)
                logger.info(f"Loaded {len(batch)} instances from {name} (split={split})")
                loaded = True
                break
            except Exception as e:
                logger.debug(f"Could not load {name} split={split}: {e}")
        if not loaded:
            logger.warning(f"Could not load any split for dataset {name!r}, skipping.")

    _dataset_cache = merged
    logger.info(f"Total instances in cache: {len(_dataset_cache)}")
    return _dataset_cache


def evaluate_patch(
    instance_id: str,
    patch: str,
    run_id: str,
    timeout: int = 1800,
    rm_image: bool = False,
) -> dict[str, Any]:
    """Share concurrent evaluations of the exact same patch and settings.

    Each caller still receives its own result. Finished results (including
    failures) are not cached, so later requests can evaluate again.
    """
    key = (instance_id, hashlib.sha256(patch.encode("utf-8")).hexdigest(), timeout, rm_image)
    result = _evaluations.run(
        key,
        lambda: _evaluate_patch_once(instance_id, patch, run_id, timeout, rm_image),
    )
    return dict(result)


def _evaluate_patch_once(
    instance_id: str,
    patch: str,
    run_id: str,
    timeout: int = 1800,
    rm_image: bool = False,
) -> dict[str, Any]:
    """
    Apply a patch to a SWE-bench instance and run its test suite in Docker.

    Args:
        instance_id: SWE-bench instance identifier, e.g. "django__django-12345"
        patch: Unified diff patch string to apply
        run_id: Unique identifier for this evaluation run (used for Docker naming)
        timeout: Seconds before the Docker execution is killed
        rm_image: Whether to remove the Docker image after evaluation

    Returns:
        {"resolved": bool, "report": str}
    """
    cache = load_dataset_cache()
    if instance_id not in cache:
        raise ValueError(f"Instance '{instance_id}' not found in {DATASET_NAMES}")

    instance = cache[instance_id]

    try:
        from swebench.harness.test_spec import make_test_spec
    except ImportError:
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec
        except ImportError:
            try:
                from swebench.harness.test_spec.python import make_test_spec
            except ImportError as e:
                raise ImportError("swebench is not installed. Run: pip install swebench") from e

    test_spec = make_test_spec(instance)

    # swebench lowercases IDs in TestSpec. Its report builder keys by the
    # prediction ID, so both IDs must agree (e.g. Project-MONAI/MONAI).
    evaluation_id = test_spec.instance_id
    prediction = {
        "instance_id": evaluation_id,
        "model_patch": patch,
        "model_name_or_path": "grpo-training",
    }

    client = docker.from_env()
    try:
        from swebench.harness.run_evaluation import run_instance

        result = run_instance(
            test_spec=test_spec,
            pred=prediction,
            rm_image=rm_image,
            force_rebuild=False,
            client=client,
            run_id=run_id,
            timeout=timeout,
        )
    finally:
        client.close()

    # Normalise result across swebench API versions.
    # run_instance may return a dict {instance_id: result_dict} OR a tuple
    # (instance_id, result_dict) depending on the swebench version.
    if isinstance(result, tuple) and len(result) == 2:
        _, result = result  # unwrap (instance_id, result_dict)
    if isinstance(result, dict):
        inner = result.get(evaluation_id, result.get(instance_id, result))
        resolved = bool(inner.get("resolved", False))
        report = inner.get("test_output", str(inner))
    else:
        resolved = False
        report = str(result)

    return {"resolved": resolved, "report": report}
