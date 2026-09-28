"""Leased rollout rounds, ordered by registration rather than individual jobs.

Only the trainer's explicit end event releases a live round. An empty queue
means the model may be generating; it does not mean the rollout has finished.
All methods run on the server event loop, alongside JobQueue.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import time


class RoundError(RuntimeError):
    pass


@dataclass
class RolloutRound:
    id: str
    run_id: str
    step: int
    expected_rollouts: int
    phase: str
    touched: float = field(default_factory=time.monotonic)
    state: str = "active"


class RolloutRounds:
    def __init__(self, lease=120, max_records=100000):
        if lease <= 0 or max_records <= 0:
            raise ValueError("Round limits must be positive")
        self.lease = lease
        self.max_records = max_records
        self.records = {}
        self.active = OrderedDict()
        self.by_run = {}

    def lookup(self, round_id, run_id):
        record = self.records.get(round_id)
        if record is None or record.run_id != run_id:
            raise RoundError("Unknown rollout round or mismatched run ID")
        return record

    def start(self, round_id, run_id, step, expected_rollouts, phase):
        self.expire()
        if round_id in self.records:
            record = self.lookup(round_id, run_id)
            if (record.step, record.expected_rollouts, record.phase) != (step, expected_rollouts, phase):
                raise RoundError("Round ID already belongs to different metadata")
            if record.state != "active":
                raise RoundError("Closed round must not be restarted")
            record.touched = time.monotonic()
            return record
        if run_id in self.by_run:
            raise RoundError("Run already has an active rollout round")
        if len(self.records) >= self.max_records:
            raise RoundError("Rollout round record capacity reached")
        record = RolloutRound(round_id, run_id, step, expected_rollouts, phase)
        self.records[round_id] = record
        self.active[round_id] = record
        self.by_run[run_id] = round_id
        return record

    def require_active(self, round_id, run_id):
        self.expire()
        record = self.lookup(round_id, run_id)
        if record.state != "active":
            raise RoundError("Rollout round is closed or its trainer heartbeat expired")
        return record

    def heartbeat(self, round_id, run_id):
        record = self.require_active(round_id, run_id)
        record.touched = time.monotonic()
        return record

    def _close(self, record, state):
        record.state = state
        self.active.pop(record.id, None)
        self.by_run.pop(record.run_id, None)

    def end(self, round_id, run_id, outcome):
        record = self.lookup(round_id, run_id)
        if record.state == "active":
            self._close(record, outcome)
        # Retain tombstones: late end/heartbeat/start events cannot close or
        # revive a newer round belonging to the same run.
        return record

    def expire(self):
        now = time.monotonic()
        for record in list(self.active.values()):
            if now - record.touched >= self.lease:
                self._close(record, "lease_expired")

    def owner(self):
        self.expire()
        return next(iter(self.active.values()), None)

    def view(self, record):
        owner = self.owner()
        return {"round_id": record.id, "run_id": record.run_id, "step": record.step,
                "phase": record.phase, "expected_rollouts": record.expected_rollouts,
                "state": record.state, "has_priority": owner is record}

    def stats(self):
        owner = self.owner()
        return {"policy": "rollout_round_priority", "lease_seconds": self.lease,
                "priority_round": self.view(owner) if owner else None,
                "active_rounds": [self.view(r) for r in list(self.active.values())]}
