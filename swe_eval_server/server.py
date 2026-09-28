"""
SWE Evaluation Server

FastAPI server with two modes:
  1. Single-turn  POST /evaluate       – apply a patch, run tests, return reward
  2. Agent-loop   POST /session/*      – stateful Docker sessions for multi-turn RL

Used by veRL reward workers and the SWEInteraction class during GRPO training.

Usage:
    uvicorn swe_eval_server.server:app --host 0.0.0.0 --port 8000 --workers 1

Environment variables:
    SWE_DATASET          HuggingFace dataset name (default: princeton-nlp/SWE-bench_Verified)
    MAX_CONCURRENT_EVALS Max parallel Docker evaluations (default: 8)
    EVAL_TIMEOUT         Seconds per Docker evaluation (default: 300)
    SESSION_TTL          Absolute max session lifetime in seconds (default: 7200)
    IDLE_TTL             Seconds of inactivity before a session is reaped (default: 900)
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .evaluator import evaluate_patch, load_dataset_cache
from .session_manager import IDLE_TTL, SESSION_TTL, SessionManager, ShellSessionError
from .job_queue import JobQueue, QueueError
from .rollout_rounds import RoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_CONCURRENT_EVALS = int(os.environ.get("MAX_CONCURRENT_EVALS", "8"))
EVAL_TIMEOUT = int(os.environ.get("EVAL_TIMEOUT", "1800"))
SESSION_TTL_ENV = int(os.environ.get("SESSION_TTL", str(SESSION_TTL)))
IDLE_TTL_ENV = int(os.environ.get("IDLE_TTL", str(IDLE_TTL)))

app = FastAPI(title="SWE Evaluation Server", version="2.0")

_semaphore: Optional[asyncio.Semaphore] = None
_session_manager: Optional[SessionManager] = None
_jobs: Optional[JobQueue] = None


def _docker_smoke_test() -> None:
    """
    Pull and run hello-world to verify the Docker daemon is reachable.

    If you keep the SWE-bench image store on an external volume, set
    DOCKER_ROOT_DIR to that path and this also checks the daemon's active data
    root matches it -- a volume mounted after Docker started otherwise makes all
    previously-built images silently invisible.
    """
    import docker as docker_lib
    client = docker_lib.from_env()
    try:
        info = client.info()
        root = info.get("DockerRootDir", "unknown")
        expected = os.environ.get("DOCKER_ROOT_DIR")
        if expected and root != expected:
            raise RuntimeError(
                f"Docker Root Dir is '{root}', expected '{expected}' (DOCKER_ROOT_DIR). "
                "The Docker daemon likely started before that volume was mounted. "
                "Fix: sudo systemctl restart docker"
            )

        # Pull hello-world (tiny, ~13 kB) and run it; remove image afterwards.
        client.images.pull("hello-world")
        client.containers.run("hello-world", remove=True)
        client.images.remove("hello-world", force=True)
        logger.info(f"Docker smoke test passed. Image storage at {root} is accessible.")
    finally:
        client.close()


@app.on_event("startup")
async def startup():
    global _semaphore, _session_manager, _jobs
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT_EVALS)
    _session_manager = SessionManager(
        rm_containers=True,
        session_ttl=SESSION_TTL_ENV,
        idle_ttl=IDLE_TTL_ENV,
    )
    await asyncio.to_thread(_docker_smoke_test)
    await asyncio.to_thread(load_dataset_cache)
    _jobs = JobQueue(
        _execute_job, _session_manager,
        workers=int(os.environ.get("EVAL_JOB_WORKERS", str(min(32, MAX_CONCURRENT_EVALS)))),
        max_pending=int(os.environ.get("EVAL_MAX_PENDING_JOBS", "2048")),
        queue_timeout=float(os.environ.get("EVAL_QUEUE_TIMEOUT", "7200")),
        round_lease=float(os.environ.get("EVAL_ROUND_LEASE", "120")),
    )
    _jobs.start()
    logger.info(
        f"Server ready. MAX_CONCURRENT_EVALS={MAX_CONCURRENT_EVALS}, "
        f"EVAL_TIMEOUT={EVAL_TIMEOUT}s, SESSION_TTL={SESSION_TTL_ENV}s, "
        f"IDLE_TTL={IDLE_TTL_ENV}s"
    )


# ===========================================================================
# Single-turn endpoint (original)
# ===========================================================================

@app.on_event("shutdown")
async def shutdown():
    if _jobs is not None:
        await _jobs.close()

class EvalRequest(BaseModel):
    instance_id: str
    patch: str
    run_id: Optional[str] = None

class EvalResponse(BaseModel):
    instance_id: str
    resolved: bool
    report: str
    error: Optional[str] = None


@app.post("/evaluate", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    run_id = req.run_id or str(uuid.uuid4())[:12]
    logger.info(f"[{run_id}] Evaluating {req.instance_id} (patch_len={len(req.patch)})")

    try:
        result = await _legacy_job("/evaluate", dict(instance_id=req.instance_id, patch=req.patch, run_id=run_id))
        resolved = result["resolved"]
        logger.info(f"[{run_id}] {req.instance_id} → resolved={resolved}")
        return EvalResponse(
            instance_id=req.instance_id,
            resolved=resolved,
            report=result.get("report", ""),
        )
    except Exception as exc:
        logger.exception(f"[{run_id}] Evaluation failed for {req.instance_id}: {exc}")
        return EvalResponse(
            instance_id=req.instance_id,
            resolved=False, report="", error=str(exc),
        )


# ===========================================================================
# Agent-loop session endpoints
# ===========================================================================

class SessionStartRequest(BaseModel):
    instance_id: str  # SWE-bench instance_id

class SessionStartResponse(BaseModel):
    session_id: str
    observation: str
    error: Optional[str] = None

class SessionStepRequest(BaseModel):
    session_id: str
    command: str
    timeout: int = 300

class SessionStepResponse(BaseModel):
    observation: str
    error: Optional[str] = None

class SessionFinishRequest(BaseModel):
    session_id: str

class SessionFinishResponse(BaseModel):
    resolved: bool
    reward: float
    report: str
    patch: str = ""
    error: Optional[str] = None

class SessionCleanupRequest(BaseModel):
    session_id: str


@app.post("/session/start", response_model=SessionStartResponse)
async def session_start(req: SessionStartRequest):
    cache = load_dataset_cache()
    if req.instance_id not in cache:
        return SessionStartResponse(
            session_id="", observation="",
            error=f"Instance '{req.instance_id}' not found in dataset.",
        )
    try:
        result = await _legacy_job("/session/start", {"instance_id": req.instance_id})
        session_id, observation = result["session_id"], result["observation"]
        return SessionStartResponse(session_id=session_id, observation=observation)
    except Exception as exc:
        logger.exception(f"session_start failed for {req.instance_id}: {exc}")
        return SessionStartResponse(session_id="", observation="", error=str(exc))


@app.post("/session/step", response_model=SessionStepResponse)
async def session_step(req: SessionStepRequest):
    try:
        result = await _legacy_job("/session/step", dict(session_id=req.session_id, command=req.command, timeout=req.timeout))
        observation = result["observation"]
        return SessionStepResponse(observation=observation)
    except Exception as exc:
        logger.warning(f"session_step failed for {req.session_id}: {exc}")
        return SessionStepResponse(observation="", error=str(exc))


@app.post("/session/finish", response_model=SessionFinishResponse)
async def session_finish(req: SessionFinishRequest):
    try:
        result = await _legacy_job("/session/finish", {"session_id": req.session_id})
        return SessionFinishResponse(
            resolved=result["resolved"],
            reward=result["reward"],
            report=result.get("report", ""),
            patch=result.get("patch", ""),
        )
    except ShellSessionError as exc:
        logger.warning("session_finish patch capture failed for %s: %s", req.session_id, exc)
        # Existing GRPO clients check HTTP status, but ignore JSON error
        # fields. A non-2xx response reaches their infra-abort handling.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"session_finish failed for {req.session_id}: {exc}")
        return SessionFinishResponse(
            resolved=False, reward=-1.0, report="", error=str(exc)
        )


@app.post("/session/cleanup")
async def session_cleanup(req: SessionCleanupRequest):
    """Force-cleanup a session without evaluation (e.g. on timeout)."""
    try:
        await _legacy_job("/session/cleanup", {"session_id": req.session_id})
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ===========================================================================
# Health
# ===========================================================================

_REQUEST_MODELS = {
    "/evaluate": EvalRequest, "/session/start": SessionStartRequest,
    "/session/step": SessionStepRequest, "/session/finish": SessionFinishRequest,
    "/session/cleanup": SessionCleanupRequest,
}


def _execute_job(endpoint: str, payload: dict) -> dict:
    if endpoint == "/evaluate":
        return evaluate_patch(instance_id=payload["instance_id"], patch=payload["patch"],
                              run_id=payload.get("run_id") or uuid.uuid4().hex[:12], timeout=EVAL_TIMEOUT)
    if endpoint == "/session/start":
        instance = load_dataset_cache()[payload["instance_id"]]
        sid, obs = _session_manager.start(instance)
        logger.info("Session started: %s for %s", sid, payload["instance_id"])
        return {"session_id": sid, "observation": obs}
    if endpoint == "/session/step":
        return {"observation": _session_manager.step(payload["session_id"], payload["command"],
                                                      payload.get("timeout", 300))}
    if endpoint == "/session/finish":
        result = _session_manager.finish(payload["session_id"])
        logger.info("Session finished: %s → resolved=%s", payload["session_id"], result["resolved"])
        return result
    if endpoint == "/session/cleanup":
        _session_manager.cleanup(payload["session_id"])
        return {"status": "ok"}
    raise ValueError("Unsupported evaluation operation")


async def _legacy_job(endpoint: str, payload: dict) -> dict:
    if _jobs is None:
        # Compatibility with embedded callers that initialize SessionManager
        # themselves (including the real Docker/HTTP regression suite).
        async with _semaphore:
            return await asyncio.to_thread(_execute_job, endpoint, payload)
    job = _jobs.submit(uuid.uuid4().hex, "legacy", endpoint, payload)
    try:
        while not job.done.is_set():
            job.touched = time.monotonic()
            try:
                await asyncio.wait_for(job.done.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass
        if job.state != "succeeded":
            raise ShellSessionError(f"{job.error_kind}: {job.error}")
        result = job.result
        _jobs.acknowledge(job)
        return result
    except asyncio.CancelledError:
        _jobs.cancel(job)
        raise


class JobRequest(BaseModel):
    job_id: str = Field(min_length=32, max_length=36)
    generation: str
    run_id: str = Field(min_length=1, max_length=128)
    endpoint: Literal["/evaluate", "/session/start", "/session/step", "/session/finish", "/session/cleanup"]
    payload: dict
    round_id: Optional[str] = Field(default=None, min_length=32, max_length=36)


class RoundIdentity(BaseModel):
    generation: str
    run_id: str = Field(min_length=1, max_length=128)


class RoundStart(RoundIdentity):
    round_id: str = Field(min_length=32, max_length=36)
    step: int = Field(ge=0)
    expected_rollouts: int = Field(gt=0)
    phase: Literal["train", "validation"] = "train"


class RoundEnd(RoundIdentity):
    outcome: Literal["completed", "aborted"]


def _round_action(req, round_id, action):
    queue = _queue()
    try:
        uuid.UUID(round_id)
        queue.check_generation(req.generation)
        result = action(queue)
        queue.wakeup.set()
        logger.info("rollout_round %s", result)
        return {"generation": queue.generation, **result}
    except QueueError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except RoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/rounds/start")
async def start_round(req: RoundStart):
    return _round_action(req, req.round_id, lambda q: q.rounds.view(q.rounds.start(
        req.round_id, req.run_id, req.step, req.expected_rollouts, req.phase)))


@app.post("/rounds/{round_id}/heartbeat")
async def heartbeat_round(round_id: str, req: RoundIdentity):
    return _round_action(req, round_id, lambda q: q.rounds.view(q.rounds.heartbeat(round_id, req.run_id)))


@app.post("/rounds/{round_id}/end")
async def end_round(round_id: str, req: RoundEnd):
    return _round_action(req, round_id, lambda q: q.end_round(round_id, req.run_id, req.outcome))


def _queue():
    if _jobs is None:
        raise HTTPException(503, "Evaluation job queue is not initialized")
    return _jobs


def _lookup(job_id: str, generation: str):
    queue = _queue()
    try:
        queue.check_generation(generation)
    except QueueError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    if job_id not in queue.jobs:
        raise HTTPException(404, "Unknown job; do not replay an uncertain submission")
    return queue, queue.jobs[job_id]


@app.get("/jobs/capabilities")
async def job_capabilities():
    queue = _queue()
    return {"protocol": 1, "generation": queue.generation,
            "queue_budget_seconds": queue.queue_timeout, "lease_seconds": queue.lease,
            "round_protocol": 1, "round_lease_seconds": queue.rounds.lease,
            "scheduling": "rollout_round_priority"}


@app.post("/jobs", status_code=202)
async def submit_job(req: JobRequest):
    queue = _queue()
    try:
        queue.check_generation(req.generation)
        uuid.UUID(req.job_id)
        validated = _REQUEST_MODELS[req.endpoint](**req.payload)
        payload = validated.model_dump() if hasattr(validated, "model_dump") else validated.dict()
        job = queue.submit(req.job_id, req.run_id, req.endpoint, payload, round_id=req.round_id)
    except QueueError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return queue.view(job)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, generation: str):
    queue, job = _lookup(job_id, generation)
    return queue.view(job, touch=True)


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, generation: str):
    queue, job = _lookup(job_id, generation)
    return queue.cancel(job)


@app.post("/jobs/{job_id}/ack")
async def ack_job(job_id: str, generation: str):
    queue, job = _lookup(job_id, generation)
    try:
        queue.acknowledge(job)
    except QueueError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return {"status": "ok"}

@app.get("/health")
async def health():
    import time
    now = time.time()
    if _session_manager is None:
        return {"status": "ok", "max_concurrent_evals": MAX_CONCURRENT_EVALS, "active_sessions": 0}
    with _session_manager._lock:
        sessions = list(_session_manager.sessions.values())
    idle = sum(1 for s in sessions if not s.pending_requests
               and now - s.last_activity_at > IDLE_TTL_ENV)
    return {
        "status": "ok",
        "max_concurrent_evals": MAX_CONCURRENT_EVALS,
        "session_ttl": SESSION_TTL_ENV,
        "idle_ttl": IDLE_TTL_ENV,
        "active_sessions": len(sessions),
        "idle_sessions_awaiting_reap": idle,
        "queue": _jobs.stats() if _jobs is not None else None,
    }
