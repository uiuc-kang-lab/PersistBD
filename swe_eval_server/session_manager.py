"""
Docker session manager for stateful SWE-bench evaluation.

Each session is a running Docker container with the SWE-bench repo at
base_commit. The agent runs bash commands inside it across multiple turns.
At finish(), we extract the git diff and evaluate it with swebench.
"""

import logging
import os
import re
import shlex
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import docker
from docker.errors import DockerException

from .inflight import InFlightOperations
from .pdm_build_compat import prepare_pdm_build_specs
from .docker_pool_compat import make_session_docker_client

logger = logging.getLogger(__name__)

# Default OFF: instance images are removed after a session closes (original behavior),
# which keeps the docker data-root bounded. Enabling KEEP_INSTANCE_IMAGES=1 caches
# instance images to skip the ~1-3 min rebuild, BUT the full instance-image set for
# large datasets (e.g. SWE-Gym, ~2500 instances) does NOT fit on a 2TB disk and will
# fill it to 100% with no eviction — only enable it with a bounded/LRU policy or a
# much larger data-root. Env/base images are always kept regardless.
KEEP_INSTANCE_IMAGES = os.environ.get("KEEP_INSTANCE_IMAGES", "0").lower() not in ("0", "false", "no")

MAX_OUTPUT_LEN = 70000  # truncate long bash outputs (match training max_observation_length)
EXEC_TIMEOUT   = 300    # seconds per bash command (matches swesmith_infer.yaml execution_timeout)
SESSION_TTL    = 7200   # seconds before a session is force-killed on absolute age (safety net)
IDLE_TTL       = 3600   # seconds of inactivity before a session is reaped. Bumped 900→3600 because
                        # long vLLM generations between tool calls can exceed 15 min on long-context
                        # rollouts and were getting their containers reaped mid-rollout. Active
                        # leak cleanup now lives in SWEInteraction (per-worker stale sweep), so this
                        # only needs to be a safety net for clients that crash without finalising.


def _parse_docker_stream(raw: bytes) -> str:
    """Parse Docker multiplexed stream format (8-byte header + payload, tty=False)."""
    out = []
    offset = 0
    while offset + 8 <= len(raw):
        size = struct.unpack(">I", raw[offset + 4: offset + 8])[0]
        offset += 8
        if offset + size > len(raw):
            break
        out.append(raw[offset: offset + size].decode("utf-8", errors="replace"))
        offset += size
    return "".join(out)


class ShellSessionError(RuntimeError):
    """The command transport failed or a required internal command failed."""


class _ShellTimeout(ShellSessionError):
    def __init__(self, output: str):
        super().__init__("Command timed out")
        self.output = output


def _capture_patch(container, timeout: float = 30) -> str:
    """Read only git's stdout on a fresh Docker exec, isolated from agent output."""
    try:
        result = container.exec_run(
            ["/usr/bin/timeout", "--signal=KILL", str(timeout),
             "/usr/bin/git", "--no-pager", "-C", "/testbed", "diff",
             "--no-ext-diff", "--no-textconv", "--no-color"],
            stdin=False, stdout=True, stderr=True, tty=False, demux=True,
            workdir="/",
        )
    except Exception as exc:
        raise ShellSessionError("Final patch capture transport failed") from exc
    stdout, stderr = result.output or (b"", b"")
    if result.exit_code != 0:
        diagnostic = (stderr or b"").decode("utf-8", errors="replace")[:500]
        raise ShellSessionError(
            f"Final patch capture exited with status {result.exit_code}: {diagnostic}")
    patch = (stdout or b"").decode("utf-8", errors="replace")
    if patch and not patch.startswith("diff --git "):
        raise ShellSessionError("Final patch capture returned non-diff stdout")
    return patch


class PersistentBashSession:
    """Persistent shell; failed commands cannot consume later protocol messages."""

    def __init__(self, container):
        self._container = container
        self._sock = None
        self._socket_stream = None
        self._shell_pid = None
        self._broken = True
        self._open_shell()

    def _open_shell(self) -> None:
        # A separate process group lets timeout recovery cancel the shell AND
        # its foreground children, without stopping the session's container.
        exec_resp = self._container.client.api.exec_create(
            self._container.id,
            # Keep Docker's exec owner alive while the child shell runs.
            # setsid without --wait may fork and exit after the handshake,
            # closing the Docker stream before the next command arrives.
            ["setsid", "--fork", "--wait", "/bin/bash", "--norc", "--noprofile"],
            stdin=True, stdout=True, stderr=True, tty=False,
        )
        self._exec_id = exec_resp["Id"]
        sock = self._container.client.api.exec_start(
            self._exec_id, detach=False, tty=False, socket=True
        )
        self._socket_stream = sock
        self._sock = sock._sock if hasattr(sock, "_sock") else sock
        try:
            output, status = self._execute(
                "PS1=''; PS2=''; HISTFILE=/dev/null; printf '%s' \"$$\"", 5
            )
            pid = int(output.strip())
            if status or pid <= 1:
                raise ShellSessionError("Invalid shell startup handshake")
            self._shell_pid = pid
            self._broken = False
        except Exception:
            self._broken = True
            self._close_socket()
            raise

    def run(self, command: str, timeout: int = EXEC_TIMEOUT, *, check: bool = False, output_limit: int | None = MAX_OUTPUT_LEN) -> str:
        """Run a command, recovering the shell on timeout or unexpected exit.

        Files survive recovery. Shell-local state (cwd, variables, functions)
        resets to the container's initial state and is disclosed in the result.
        ``check`` is for internal commands whose output must not become a patch
        when execution fails. ``output_limit=None`` is reserved for the final
        patch capture; ordinary command observations keep their existing limit.
        """
        if self._broken:
            raise ShellSessionError("Shell is unavailable after failed recovery")
        try:
            output, status = self._execute(command, timeout)
        except (ShellSessionError, OSError) as exc:
            self._broken = True
            try:
                self._stop_shell()
                self._open_shell()
            except Exception as recovery_error:
                raise ShellSessionError("Shell recovery failed") from recovery_error
            reason = "command timed out" if isinstance(exc, _ShellTimeout) else "shell exited or disconnected"
            notice = (
                f"[{reason}; shell restarted. Files are preserved; working directory "
                "and environment variables reset to the container's initial state.]"
            )
            if check:
                raise ShellSessionError(notice) from exc
            # Put recovery information first so client-side clipping retains it.
            output = getattr(exc, "output", "")
            return (notice + "\n" + output)[:MAX_OUTPUT_LEN]
        if check and status:
            raise ShellSessionError(f"Internal command exited with status {status}: {output[:500]}")
        return output[:output_limit]

    def _execute(self, command: str, timeout: float) -> tuple[str, int]:
        sentinel = f"__DONE_{uuid.uuid4().hex}__"
        # eval receives ONE complete quoted argument. A missing quote in the
        # model's command is a syntax error within eval, not an unfinished
        # protocol message. eval runs in this shell, preserving cwd/env changes.
        # Commands must not read the socket carrying subsequent requests.
        self._send(
            f"eval -- {shlex.quote(command)} </dev/null\n"
            f"builtin printf '\\n{sentinel}:%s\\n' \"$?\"\n"
        )
        return self._read_until(sentinel, timeout)

    def _stop_shell(self) -> None:
        pid = self._shell_pid
        try:
            if pid is not None:
                # kill may report an already-exited group: that is also safe.
                result = self._container.exec_run([
                    "/bin/bash", "--norc", "--noprofile", "-c",
                    'kill -KILL -- "-$1" 2>/dev/null || ! kill -0 -- "-$1" 2>/dev/null',
                    "shell-cleanup", str(pid),
                ])
                if result.exit_code:
                    raise ShellSessionError("Could not terminate expired shell process group")
                self._shell_pid = None
        finally:
            self._close_socket()

    def _close_socket(self) -> None:
        stream, raw = self._socket_stream, self._sock
        self._socket_stream = self._sock = None
        try:
            if stream is not None and stream is not raw:
                stream.close()
        finally:
            if raw is not None:
                raw.close()

    def close(self) -> None:
        self._broken = True
        self._stop_shell()

    def _send(self, data: str) -> None:
        self._sock.sendall(data.encode("utf-8"))

    def _read_until(self, sentinel: str, timeout: float) -> tuple[str, int]:
        deadline = time.monotonic() + timeout
        raw_buf = b""
        marker = re.compile(r"\n" + re.escape(sentinel) + r":(\d+)\n")
        while time.monotonic() < deadline:
            remaining = max(0.001, deadline - time.monotonic())
            self._sock.settimeout(min(remaining, 1.0))
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise ShellSessionError("Persistent shell closed before command completion")
            raw_buf += chunk
            text = _parse_docker_stream(raw_buf)
            match = marker.search(text)
            if match:
                # The marker includes the protocol newline. Preserve all command bytes
                # before it, including the final newline required by unified diffs.
                return text[:match.start()], int(match.group(1))
        raise _ShellTimeout(_parse_docker_stream(raw_buf)[:MAX_OUTPUT_LEN])


@dataclass
class Session:
    session_id:       str
    instance_id:      str          # SWE-bench instance_id
    instance:         dict         # full SWE-bench instance dict
    container:        Any          # docker Container
    bash_session:     PersistentBashSession
    lock:             threading.Lock = field(default_factory=threading.Lock)
    step_count:       int = 0
    created_at:       float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    pending_requests: int = 0
    queue_started_at: float | None = None
    queued_seconds: float = 0.0


class SessionManager:
    def __init__(
        self,
        rm_containers: bool = True,
        session_ttl:   int = SESSION_TTL,
        idle_ttl:      int = IDLE_TTL,
    ):
        self.client = make_session_docker_client()
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.rm_containers = rm_containers
        self._ttl = session_ttl
        self._idle_ttl = idle_ttl
        # Overlapping requests share the same build result, including failure.
        self._image_builds = InFlightOperations()
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, instance: dict) -> tuple[str, str]:
        """
        Start a Docker container for the instance.

        Returns:
            (session_id, initial_observation)
        """
        try:
            from swebench.harness.test_spec import make_test_spec
        except ImportError:
            try:
                from swebench.harness.test_spec.test_spec import make_test_spec
            except ImportError:
                from swebench.harness.test_spec.python import make_test_spec

        test_spec = make_test_spec(instance)

        # Build images on demand if missing.
        # Env images are kept permanently (shared across many instances, 5-20 min to build).
        # Instance images are built per-session and removed after cleanup (~1-3 min to build).
        self._ensure_image(test_spec.env_image_key, [instance], env_only=True)
        self._ensure_image(test_spec.instance_image_key, [instance], env_only=False)

        session_id = str(uuid.uuid4())[:12]

        container = self.client.containers.run(
            image=test_spec.instance_image_key,
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            working_dir="/testbed",
            environment={"DEBIAN_FRONTEND": "noninteractive"},
            mem_limit="2g",
        )

        # Brief pause to let the container initialise
        time.sleep(1)

        try:
            bash_session = PersistentBashSession(container)
        except Exception:
            # A startup/handshake failure must not leave an untracked container.
            container.remove(force=True)
            raise

        repo_tree = bash_session.run(
            "find /testbed -maxdepth 2 -name '*.py' | sort | head -40",
            timeout=30,
        )

        session = Session(
            session_id=session_id,
            instance_id=instance["instance_id"],
            instance=instance,
            container=container,
            bash_session=bash_session,
        )

        with self._lock:
            self.sessions[session_id] = session

        initial_obs = (
            f"Environment ready. Repository is at /testbed.\n\n"
            f"Python files (up to 40):\n{repo_tree}\n\n"
            f"Use bash commands to explore and fix the issue. "
            f"Call <function=finish></function> when done."
        )
        return session_id, initial_obs

    def step(self, session_id: str, command: str, timeout: int = EXEC_TIMEOUT) -> str:
        """Execute a bash command in the persistent shell. Returns stdout+stderr."""
        session = self._get(session_id)
        with session.lock:
            session.step_count += 1
            session.last_activity_at = time.time()
            result = session.bash_session.run(command, timeout=timeout)
        return result[:MAX_OUTPUT_LEN]

    def finish(self, session_id: str) -> dict[str, Any]:
        """
        Capture the agent's changes via git diff, evaluate with swebench,
        then clean up the container.

        Returns:
            {"resolved": bool, "reward": float, "report": str, "patch": str}
        """
        session = self._get(session_id)
        try:
            with session.lock:
                # The interactive stream can contain delayed output from background
                # processes. Capture git separately, with stdout/stderr demultiplexed.
                patch = _capture_patch(session.container)

            if not patch.strip():
                return {
                    "resolved": False, "reward": -1.0,
                    "report": "No changes made to the repository.", "patch": "",
                }

            from .evaluator import evaluate_patch
            result = evaluate_patch(
                instance_id=session.instance_id,
                patch=patch,
                run_id=f"{session_id}-eval",
            )

            return {
                "resolved": result["resolved"],
                "reward":   1.0 if result["resolved"] else -1.0,
                "report":   result.get("report", ""),
                "patch":    patch,
            }
        except ShellSessionError:
            # Do not turn a failed git diff into a fake patch or model failure.
            # The HTTP endpoint propagates this as an infrastructure error.
            raise
        except Exception as exc:
            logger.exception(f"finish() failed for session {session_id}: {exc}")
            return {"resolved": False, "reward": -1.0, "report": str(exc), "patch": ""}
        finally:
            self._cleanup(session_id)

    def cleanup(self, session_id: str) -> None:
        """Force-cleanup a session without evaluation."""
        self._cleanup(session_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_image(self, image_key: str, dataset: list, env_only: bool) -> None:
        """Build once for overlapping requests; all callers see the same result."""
        try:
            self.client.images.get(image_key)
            return
        except docker.errors.ImageNotFound:
            pass
        self._image_builds.run(
            image_key,
            lambda: self._build_missing_image(image_key, dataset, env_only),
        )

    def _build_missing_image(self, image_key: str, dataset: list, env_only: bool) -> None:
        # Another process may have built the image since the first lookup.
        try:
            self.client.images.get(image_key)
            return
        except docker.errors.ImageNotFound:
            pass

        try:
            from swebench.harness.docker_build import build_env_images, build_instance_images
        except ImportError:
            raise RuntimeError("swebench not installed; run image builds with the SWE environment")

        if env_only:
            logger.info(f"Building env image {image_key} on demand (may take 5-20 min)...")
            build_env_images(self.client, dataset=dataset, force_rebuild=False, max_workers=2)
        else:
            logger.info(f"Building instance image {image_key} on demand (~1-3 min)...")
            build_instance_images(self.client, dataset=prepare_pdm_build_specs(dataset), force_rebuild=False, max_workers=2)
        try:
            self.client.images.get(image_key)
        except docker.errors.ImageNotFound as exc:
            raise RuntimeError(f"Failed to build image {image_key}. Check build logs for details.") from exc
        logger.info(f"Image {image_key} built successfully.")

    def _get(self, session_id: str) -> Session:
        with self._lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session '{session_id}' not found")
        return session

    def pin_queued(self, session_id: str) -> None:
        """Pin before admission waiting; the reaper checks under the same lock."""
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session '{session_id}' not found")
            if session.pending_requests:
                raise RuntimeError("Session already has pending work")
            session.pending_requests = 1
            session.queue_started_at = time.monotonic()

    def begin_request(self, session_id: str) -> None:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session '{session_id}' not found")
            if session.queue_started_at is not None:
                session.queued_seconds += time.monotonic() - session.queue_started_at
                session.queue_started_at = None
            session.last_activity_at = time.time()

    def unpin_request(self, session_id: str) -> None:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is not None:
                if session.queue_started_at is not None:
                    session.queued_seconds += time.monotonic() - session.queue_started_at
                    session.queue_started_at = None
                session.pending_requests = 0
                session.last_activity_at = time.time()

    def _expired(self, session, now) -> bool:
        if session.pending_requests:
            return False
        return (now - session.last_activity_at > self._idle_ttl
                or now - session.created_at - session.queued_seconds > self._ttl)

    def _reap_loop(self) -> None:
        """Background thread: kill sessions that exceed idle or absolute TTL."""
        while True:
            time.sleep(60)
            now = time.time()
            with self._lock:
                expired: list[tuple[str, str]] = []
                for sid, s in self.sessions.items():
                    idle = now - s.last_activity_at
                    age  = now - s.created_at
                    if s.pending_requests:
                        continue
                    age -= s.queued_seconds
                    if idle > self._idle_ttl:
                        expired.append((sid, f"idle {idle:.0f}s > {self._idle_ttl}s"))
                    elif age > self._ttl:
                        expired.append((sid, f"age {age:.0f}s > {self._ttl}s"))
            for sid, reason in expired:
                logger.warning(f"Reaping session {sid} ({reason}).")
                self._cleanup(sid, only_if_expired=True)

    def _cleanup(self, session_id: str, *, only_if_expired: bool = False) -> None:
        with self._lock:
            session = self.sessions.get(session_id)
            if only_if_expired and (session is None or not self._expired(session, time.time())):
                return
            session = self.sessions.pop(session_id, None)
            if session is None:
                return
            still_in_use_ids = {s.instance_id for s in self.sessions.values()}

        # Resolve instance image key outside the lock (import can be slow)
        instance_image = None
        still_in_use = session.instance_id in still_in_use_ids
        if not still_in_use:
            try:
                from swebench.harness.test_spec import make_test_spec
            except ImportError:
                try:
                    from swebench.harness.test_spec.test_spec import make_test_spec
                except ImportError:
                    from swebench.harness.test_spec.python import make_test_spec
            try:
                test_spec = make_test_spec(session.instance)
                instance_image = test_spec.instance_image_key
            except Exception:
                pass

        try:
            session.bash_session.close()
        except Exception as exc:
            logger.warning(f"bash_session.close() failed for {session_id}: {exc}")
        if self.rm_containers:
            try:
                session.container.stop(timeout=5)
                session.container.remove(force=True)
            except Exception as exc:
                logger.warning(f"Container cleanup failed for {session_id}: {exc}")
        # Remove the instance image to free disk; the env image is kept (shared/expensive).
        # Default (KEEP_INSTANCE_IMAGES=0) removes it. Set KEEP_INSTANCE_IMAGES=1 to keep it
        # as a warm cache so later rollouts on this instance skip the ~1-3 min rebuild
        # (see the disk-usage caveat where KEEP_INSTANCE_IMAGES is defined).
        if KEEP_INSTANCE_IMAGES:
            if instance_image:
                logger.info(f"Keeping instance image {instance_image} after session {session_id} (KEEP_INSTANCE_IMAGES=1)")
        elif instance_image and not still_in_use:
            self._remove_instance_image(instance_image, session_id)

    def _remove_instance_image(self, instance_image: str, session_id: str) -> None:
        """Remove an instance image to reclaim disk, reaping anything pinning it.

        A plain images.remove(force=False) fails whenever *any* container still
        references the image -- most often a SWE-bench evaluation container that
        outlived the session (e.g. after a patch-apply failure). The removal then
        raises, the warning is logged, and the image silently leaks; enough leaks
        fill the docker data-root and on-demand builds start failing. So we first
        force-remove every non-running container built from this image, then
        remove the image itself.
        """
        try:
            pinning = self.client.containers.list(
                all=True, filters={"ancestor": instance_image}
            )
        except Exception as exc:
            pinning = []
            logger.warning(f"Could not list containers for {instance_image}: {exc}")
        for c in pinning:
            # Only reap stopped containers -- a running one would belong to a
            # concurrent session on the same instance (which should have kept
            # still_in_use True, but guard against the TOCTOU race anyway).
            if c.status == "running":
                logger.warning(
                    f"Skipping running container {c.id[:12]} still using {instance_image}; "
                    f"deferring image removal"
                )
                return
            try:
                c.remove(force=True)
                logger.info(f"Reaped container {c.id[:12]} pinning {instance_image}")
            except Exception as exc:
                logger.warning(f"Could not remove container {c.id[:12]} for {instance_image}: {exc}")

        try:
            self.client.images.remove(instance_image, force=True)
            logger.info(f"Removed instance image {instance_image} after session {session_id}")
        except docker.errors.ImageNotFound:
            pass
        except Exception as exc:
            logger.warning(f"Could not remove image {instance_image}: {exc}")
