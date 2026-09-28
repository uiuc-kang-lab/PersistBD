# Queued evaluation requests

The evaluation server can share its execution slots across training runs. Queue
waiting and execution have separate budgets. Both queued and legacy HTTP routes
use the same worker pool. The oldest active rollout round receives priority
until its trainer explicitly ends the round. Spare slots serve other requests;
running operations are never preempted.

## Activation

Deploy both the evaluation server and training client before enabling the new
protocol. Restart the evaluation service to expose `GET /jobs/capabilities`.
At a suitable checkpoint boundary, restart training with
`SWE_EVAL_QUEUE_ENABLED=1`, or set `queue_enabled: true` in the SWEInteraction
configuration. Also select the trainer manager with
`actor_rollout_ref.rollout.agent.agent_loop_manager_class=RL.round_priority_rollout.RoundPriorityAgentLoopManager`.
The manager uses veRL's existing extension point; it does not replace the
training entry point, TPR callbacks, reward code or model workers. There is no
automatic service or training restart.

The training client defaults to the legacy protocol. This keeps a replacement
worker from switching protocols merely because source files were updated on
disk. Legacy clients remain compatible, but their HTTP timeout still includes
queue waiting. Both sides must load the new code, and the client must opt in,
to exclude queue time from its execution deadline.

For round priority, both the client opt-in and manager selection are required.
The manager validates the interaction configuration before creating its model
workers. Merely enabling the queue client does not report round boundaries.

Set a stable, distinct `eval_run_id` in each experiment's interaction config,
or provide `WANDB_RUN_ID`. Missing IDs share the `unidentified` group. Legacy
requests share the `legacy` group. The round-aware manager rejects missing run
IDs, `legacy`, and `unidentified` to avoid combining independent experiments.

## Whole-rollout priority

The trainer brackets the actual awaited `AgentLoopManager.generate_sequences`
call with `POST /rounds/start` and `POST /rounds/{id}/end`. Each batch carries its
training step, phase, actual number of repeated rollouts, and a fresh round ID.
That ID is copied into each interaction before the batch is split among Ray
workers; every environment job inherits the same generation and round identity.

If A registers before B, A's ready requests have priority. During an A model
sampling gap, spare slots execute B's requests without ending A's round. When
A's entire batch returns, B owns priority while A updates parameters or saves.
A's next round joins behind B. A currently running B operation is allowed to
finish even if A submits new work. Priority is never inferred from an empty
queue or from the number of HTTP requests.

Trainer heartbeats run every 20 seconds, including during model generation.
After `EVAL_ROUND_LEASE` (default 120 seconds) without a heartbeat, ownership is
released. Lost/aborted rounds cannot be revived by delayed messages. Their
queued operations are cancelled or rejected before dispatch; already executing
operations retain their real worker slots until they finish. End notifications
are idempotent and cannot close a newer round belonging to the same experiment.
Closed round IDs are retained, with a 100,000-record limit per server process.

If generation fails or is cancelled, the manager sends an aborted end event.
If its heartbeat is rejected or repeatedly fails, batch generation raises an
error instead of silently proceeding without round tracking. Cancelling the
local await does not claim to kill already dispatched Ray or Docker work. A
lost final end reply is logged; expiry releases priority if the event did not
arrive. Round control retries only the same idempotent lifecycle event, never
an environment command. Server restart fencing remains in effect.

Legacy or untagged requests use spare slots, with round-robin admission among
their run groups. They do not acquire a whole-rollout priority lease. Sustained
load can still reach the independent two-hour queue limit; priority does not
increase server capacity or guarantee faster completion of a slow final case.

## Protocol and failure behavior

1. Read capabilities and the current server generation.
2. Submit `POST /jobs` with `job_id` (UUID), `generation`, `run_id`, `endpoint`
   and the original endpoint `payload`, plus `round_id` for round-aware jobs. The response is HTTP 202.
3. Poll `GET /jobs/{job_id}?generation=...` for queued/running/terminal state.
4. After receiving a successful result, acknowledge it with
   `POST /jobs/{job_id}/ack?generation=...` to release the response body.

Supported operations are `/evaluate`, `/session/start`, `/session/step`,
`/session/finish` and `/session/cleanup`. Model observations and reward
calculation are unchanged. Queue and infrastructure failures are excluded from
GRPO reward statistics rather than treated as model failures.

An uncertain submission is recovered by querying the original ID; the client
does not replay a POST after its headers were sent. Duplicate IDs with the same
request return the same job. Different payloads with the same ID are rejected.
Completed IDs remain tombstones until the process exits, including after ack
and result expiry. A server restart invalidates the old generation; pending
jobs are not persisted or replayed on the new server.

`POST /jobs/{job_id}/cancel?generation=...` cancels queued work only. A running
Python thread cannot safely be killed: even after its execution deadline, its
worker slot and session pin remain held until the underlying operation exits.
The existing shell/test execution timeouts remain in effect.

Queued and running session operations are protected from the server's reaper;
queue time is deducted from absolute session age. The training stale-session
sweep skips an outstanding `generate_response` call.

## Limits and observability

- Worker slots: `EVAL_JOB_WORKERS`, default `min(32, MAX_CONCURRENT_EVALS)`.
- Pending plus running jobs: `EVAL_MAX_PENDING_JOBS`, default 2048.
- Queue budget: `EVAL_QUEUE_TIMEOUT`, default 7200 seconds, separate from execution.
- Execution budget: 2400 seconds for session finish; 1800 for other operations.
- Shell command and benchmark test limits remain 300 and 1800 seconds respectively.
- Queued clients must poll within a 120-second lease. The client polls every
  2 seconds and reports loss of contact after 60 seconds without a usable status.
- Successful response bodies expire after 3600 seconds. The process retains at
  most 1,000,000 job records; further submissions are rejected at that limit.

`GET /health` includes queued/running counts by run, the current priority
round and training step, and all active rounds. Server `job_complete` logs
and training `eval_queue_timing_<pid>.jsonl` files record queue and execution
durations separately. These timing files are not automatically sent to W&B.

## Tests

Using the training environment (Python 3.11+, aiohttp, FastAPI, httpx, uvicorn
and the evaluation server dependencies), run:

```bash
python RL/tests/test_queued_eval_client.py
python RL/tests/test_finish_timeout.py
python RL/tests/test_combined_preconnect_retry.py
python RL/tests/test_rollout_round_client.py
```

Using the evaluation environment (Python 3.10+), run:

```bash
python RL/tests/test_eval_queue.py
python RL/tests/test_round_priority.py
python RL/tests/test_docker_pool_compat.py
python RL/tests/test_output_newlines.py
python RL/tests/test_patch_capture.py
```

The Docker regressions require a locally available Linux image containing Bash,
Git, GNU timeout, setsid and patch. They create private, network-disabled test
containers and remove only those containers. Set `SHELL_TEST_IMAGE` to that
image's ID and put `RL` on `PYTHONPATH`, then run:

```bash
PYTHONPATH=RL python RL/tests/test_newline_integration.py
PYTHONPATH=RL python RL/tests/test_shell_recovery.py
```

The integration tests exercise real Docker, Git and HTTP routing. The benchmark
test runner is replaced with a synthetic expected-file-content assertion, so
they do not replay training cases or claim benchmark correctness.
