"""Process-local, bounded evaluation queue. No Docker work runs on the event loop.

The oldest active rollout round has priority, with spare slots backfilled.
Queued work consumes neither a worker
nor its execution budget. Completed IDs remain tombstones for this process so
an uncertain submission can never run twice; results are released on ack/expiry.
"""
import asyncio
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import logging
import time
import uuid

from .rollout_rounds import RolloutRounds, RoundError

logger = logging.getLogger(__name__)
TERMINAL = {"succeeded", "failed", "cancelled", "expired", "collected"}


class QueueError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


@dataclass
class Job:
    id: str
    run_id: str
    endpoint: str
    payload: dict
    fingerprint: str
    budget: float
    round_id: str | None = None
    submitted: float = field(default_factory=time.monotonic)
    touched: float = field(default_factory=time.monotonic)
    started: float | None = None
    ended: float | None = None
    state: str = "queued"
    result: dict | None = None
    error: str | None = None
    error_kind: str | None = None
    pinned: bool = False
    in_thread: bool = False
    done: asyncio.Event = field(default_factory=asyncio.Event)


class JobQueue:
    def __init__(self, execute, manager, *, workers=32, max_pending=2048,
                 max_records=1000000, queue_timeout=7200, lease=120,
                 result_ttl=3600, maintenance_interval=1, round_lease=120):
        if min(workers, max_pending, max_records, queue_timeout, lease, result_ttl) <= 0:
            raise ValueError("Queue limits must be positive")
        self.generation = uuid.uuid4().hex
        self.rounds = RolloutRounds(lease=round_lease)
        self.execute = execute
        self.manager = manager
        self.workers = workers
        self.max_pending = max_pending
        self.max_records = max_records
        self.queue_timeout = queue_timeout
        self.lease = lease
        self.result_ttl = result_ttl
        self.interval = maintenance_interval
        self.jobs = {}
        self.active_jobs = {}
        self.result_expirations = []
        self.by_run = {}
        self.runs = deque()
        self.busy_sessions = {}
        self.wakeup = asyncio.Event()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="swe-job")
        self.tasks = []
        self.closed = False
        self.pending = 0
        self.running = 0
        self.completed = Counter()

    def start(self):
        self.tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]
        self.tasks.append(asyncio.create_task(self._maintenance()))

    def check_generation(self, generation):
        if generation != self.generation:
            raise QueueError(409, "Evaluation server restarted; old job must not be replayed")

    def submit(self, job_id, run_id, endpoint, payload, *, budget=None, round_id=None):
        fingerprint = hashlib.sha256(json.dumps(
            [run_id, round_id, endpoint, payload], sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if job.fingerprint != fingerprint:
                raise QueueError(409, "Job ID already belongs to a different request")
            return job
        if round_id is not None:
            try:
                if endpoint == "/session/cleanup":
                    self.rounds.lookup(round_id, run_id)
                else:
                    self.rounds.require_active(round_id, run_id)
            except RoundError as exc:
                raise QueueError(409, str(exc)) from exc
        if self.closed:
            raise QueueError(503, "Evaluation queue is shutting down")
        if self.pending + self.running >= self.max_pending or len(self.jobs) >= self.max_records:
            raise QueueError(503, "Evaluation queue capacity reached")
        sid = payload.get("session_id")
        if sid and sid in self.busy_sessions:
            raise QueueError(409, "Session already has a queued or executing operation")
        job = Job(job_id, run_id, endpoint, payload, fingerprint,
                  budget if budget is not None else (2400 if endpoint == "/session/finish" else 1800),
                  round_id=round_id)
        if sid and endpoint != "/session/cleanup":
            self.manager.pin_queued(sid)
            job.pinned = True
        if sid:
            self.busy_sessions[sid] = job_id
        self.jobs[job_id] = job
        self.active_jobs[job_id] = job
        if run_id not in self.by_run:
            self.by_run[run_id] = deque()
            self.runs.append(run_id)
        self.by_run[run_id].append(job)
        self.pending += 1
        self.wakeup.set()
        return job

    def _next(self):
        owner = self.rounds.owner()
        if owner is not None:
            queue = self.by_run.get(owner.run_id)
            if queue:
                # A run can retain cleanup from an earlier round. Only work
                # tagged with the current round receives its priority.
                for job in queue:
                    if job.state == "queued" and job.round_id == owner.id:
                        queue.remove(job)
                        if not queue:
                            del self.by_run[owner.run_id]
                            self.runs.remove(owner.run_id)
                        return job
        # No ready work for the owner: use all spare capacity. This does not
        # transfer ownership or interrupt an already running operation.
        while self.runs:
            run = self.runs.popleft()
            queue = self.by_run[run]
            job = queue.popleft()
            if queue:
                self.runs.append(run)
            else:
                del self.by_run[run]
            if job.state == "queued":
                return job
        return None

    def _release(self, job):
        sid = job.payload.get("session_id")
        if job.pinned:
            self.manager.unpin_request(sid)
            job.pinned = False
        if sid and self.busy_sessions.get(sid) == job.id:
            del self.busy_sessions[sid]
        job.payload = {}  # Do not retain every generated command for the whole run.
        self.active_jobs.pop(job.id, None)

    def _fail(self, job, kind, message):
        was_queued = job.state == "queued"
        job.state, job.error_kind, job.error = "failed", kind, message
        job.ended = time.monotonic()
        job.done.set()
        self.completed[kind] += 1
        if was_queued:
            self.pending -= 1
            self._release(job)

    def cancel(self, job):
        if job.state == "queued":
            self._fail(job, "cancelled", "Client cancelled queued work; nothing was executed")
            job.state = "cancelled"
        # A Python thread cannot safely be cancelled. Keep its slot and pin until
        # it really returns; never replay, reap its container or oversubscribe.
        return self.view(job)

    def acknowledge(self, job):
        if job.state not in TERMINAL:
            raise QueueError(409, "Job has not finished")
        if job.state == "succeeded":
            job.state = "collected"
        job.result = None

    def view(self, job, *, touch=False):
        now = time.monotonic()
        if touch:
            job.touched = now
        queue_end = job.started if job.started is not None else (job.ended or now)
        execution_end = job.ended if not job.in_thread and job.ended is not None else now
        return {
            "generation": self.generation, "job_id": job.id, "state": job.state,
            "run_id": job.run_id, "endpoint": job.endpoint,
            "round_id": job.round_id,
            "queue_wait_seconds": max(0, queue_end - job.submitted),
            "execution_seconds": max(0, execution_end - job.started) if job.started is not None else 0,
            "execution_budget_seconds": job.budget,
            "queue_budget_seconds": self.queue_timeout,
            "worker_still_running": job.in_thread,
            "result": job.result, "error": job.error, "error_kind": job.error_kind,
        }

    async def _worker(self):
        while not self.closed:
            job = self._next()
            if job is None:
                self.wakeup.clear()
                await self.wakeup.wait()
                continue
            # Recheck just before dispatch, even if maintenance hasn't ticked.
            now = time.monotonic()
            if now - job.submitted >= self.queue_timeout:
                self._fail(job, "queue_timeout", "Queue wait exceeded its independent budget")
                continue
            if now - job.touched >= self.lease:
                self._fail(job, "client_lost", "Queued client stopped polling")
                continue
            if job.round_id and job.endpoint != "/session/cleanup":
                try:
                    self.rounds.require_active(job.round_id, job.run_id)
                except RoundError as exc:
                    self._fail(job, "round_closed", str(exc))
                    continue
            self.pending -= 1
            self.running += 1
            job.state, job.started, job.in_thread = "running", now, True
            try:
                if job.pinned:
                    self.manager.begin_request(job.payload["session_id"])
                # Exactly workers outstanding executor calls: no hidden executor
                # backlog whose waiting time would count as execution time.
                result = await asyncio.get_running_loop().run_in_executor(
                    self.executor, self.execute, job.endpoint, job.payload)
                if time.monotonic() - job.started >= job.budget and job.state == "running":
                    self._fail(job, "execution_timeout", "Execution exceeded its budget")
                if job.state == "running":
                    job.result, job.state = result, "succeeded"
                    heapq.heappush(self.result_expirations,
                                   (time.monotonic() + self.result_ttl, job.id))
                    self.completed["succeeded"] += 1
            except Exception as exc:
                if job.state == "running":
                    self._fail(job, "execution_error", f"{type(exc).__name__}: {exc}")
            finally:
                # This finally runs after the actual operation, including Docker
                # cleanup. Shutdown does not cancel worker tasks with live threads.
                job.in_thread = False
                job.ended = time.monotonic()
                self.running -= 1
                self._release(job)
                job.done.set()
                logger.info("job_complete %s", json.dumps({k: v for k, v in self.view(job).items()
                                                          if k not in ("result", "error")}))

    async def _maintenance(self):
        while not self.closed:
            self.rounds.expire()
            now = time.monotonic()
            for job in list(self.active_jobs.values()):
                if job.state == "queued":
                    if now - job.submitted >= self.queue_timeout:
                        self._fail(job, "queue_timeout", "Queue wait exceeded its independent budget")
                    elif now - job.touched >= self.lease:
                        self._fail(job, "client_lost", "Queued client stopped polling")
                elif job.state == "running" and now - job.started >= job.budget:
                    self._fail(job, "execution_timeout", "Execution exceeded its budget; worker retained until it exits")
            while self.result_expirations and self.result_expirations[0][0] <= now:
                _, job_id = heapq.heappop(self.result_expirations)
                job = self.jobs[job_id]
                if job.state == "succeeded":
                    job.result, job.state = None, "expired"
            await asyncio.sleep(self.interval)

    def stats(self):
        queued = Counter(j.run_id for q in self.by_run.values() for j in q if j.state == "queued")
        active = Counter(j.run_id for j in self.active_jobs.values() if j.in_thread)
        return {"generation": self.generation, "workers": self.workers,
                "queued": self.pending, "running": self.running,
                "queued_by_run": dict(queued), "running_by_run": dict(active),
                "completed": dict(self.completed), "scheduling": self.rounds.stats()}

    def end_round(self, round_id, run_id, outcome):
        record = self.rounds.end(round_id, run_id, outcome)
        for job in list(self.active_jobs.values()):
            if job.round_id == round_id and job.run_id == run_id and job.state == "queued":
                self.cancel(job)
        self.wakeup.set()
        return self.rounds.view(record)

    async def close(self):
        self.closed = True
        for job in list(self.active_jobs.values()):
            if job.state == "queued":
                self._fail(job, "shutdown", "Server shut down before job started")
        self.wakeup.set()
        await asyncio.gather(*self.tasks)
        self.executor.shutdown(wait=True)
