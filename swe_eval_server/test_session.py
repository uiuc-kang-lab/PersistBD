"""
Local integration test for the swe_eval_server session lifecycle.

Tests the full start → step → finish flow against a real SWE-bench instance.
Requires Docker and the swe_eval conda env.

Run with:
    conda run -n swe_eval python -m swe_eval_server.test_session
    # or from the RL/ directory:
    conda run -n swe_eval python -c "import swe_eval_server.test_session"
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEST_INSTANCE_ID = "astropy__astropy-12907"


def test_imports():
    logger.info("--- test_imports ---")
    try:
        from swebench.harness.test_spec import make_test_spec
        logger.info("make_test_spec imported from swebench.harness.test_spec")
    except ImportError:
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec
            logger.info("make_test_spec imported from swebench.harness.test_spec.test_spec")
        except ImportError:
            from swebench.harness.test_spec.python import make_test_spec
            logger.info("make_test_spec imported from swebench.harness.test_spec.python")
    logger.info("PASS: imports OK")


def test_make_test_spec(instance: dict):
    logger.info("--- test_make_test_spec ---")
    try:
        from swebench.harness.test_spec import make_test_spec
    except ImportError:
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec
        except ImportError:
            from swebench.harness.test_spec.python import make_test_spec

    spec = make_test_spec(instance)
    logger.info(f"instance_image_key : {spec.instance_image_key}")
    logger.info(f"env_image_key      : {spec.env_image_key}")
    logger.info("PASS: make_test_spec OK")
    return spec


def test_session_lifecycle(instance: dict):
    logger.info("--- test_session_lifecycle ---")
    from .session_manager import SessionManager

    mgr = SessionManager(rm_containers=True)

    # Start
    logger.info(f"Starting session for {instance['instance_id']} ...")
    session_id, obs = mgr.start(instance)
    logger.info(f"session_id  : {session_id}")
    logger.info(f"observation : {obs[:200]!r}")
    assert session_id, "session_id should not be empty"
    assert obs, "initial observation should not be empty"
    logger.info("PASS: session started")

    # Step – run a harmless command
    logger.info("Running: python --version")
    out = mgr.step(session_id, "python --version")
    logger.info(f"output: {out!r}")
    assert "Python" in out, f"Expected 'Python' in output, got: {out!r}"
    logger.info("PASS: step OK")

    # Step – make a trivial file change so git diff is non-empty
    logger.info("Creating a trivial file change ...")
    mgr.step(session_id, "echo '# test' >> /testbed/README.rst 2>/dev/null || echo '# test' >> /testbed/README.md 2>/dev/null || echo '# test' > /testbed/_claude_test.txt")

    # Finish
    logger.info("Finishing session ...")
    result = mgr.finish(session_id)
    logger.info(f"resolved : {result['resolved']}")
    logger.info(f"reward   : {result['reward']}")
    logger.info(f"patch    : {result['patch'][:200]!r}")
    # We don't assert resolved=True — the trivial change won't pass tests.
    # We just verify the finish call returns a valid structure.
    assert "resolved" in result
    assert "reward" in result
    assert "patch" in result
    logger.info("PASS: finish returned valid result")


def main():
    logger.info(f"=== SWE eval server local integration test (instance: {TEST_INSTANCE_ID}) ===")

    # 1. Imports
    test_imports()

    # 2. Load instance from dataset
    logger.info("Loading dataset ...")
    from .evaluator import load_dataset_cache
    cache = load_dataset_cache()
    assert TEST_INSTANCE_ID in cache, f"{TEST_INSTANCE_ID} not found in dataset"
    instance = cache[TEST_INSTANCE_ID]
    logger.info(f"Loaded instance: {instance['instance_id']}")

    # 3. make_test_spec
    test_make_test_spec(instance)

    # 4. Full session lifecycle
    test_session_lifecycle(instance)

    logger.info("=== All tests passed ===")


if __name__ == "__main__":
    main()
    sys.exit(0)
