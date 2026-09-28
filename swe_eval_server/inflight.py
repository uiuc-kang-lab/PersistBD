"""Share overlapping identical operations without caching completed results."""

from concurrent.futures import Future
import logging
import threading
from typing import Any, Callable, Hashable


logger = logging.getLogger(__name__)


class InFlightOperations:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending: dict[Hashable, Future] = {}

    def run(self, key: Hashable, operation: Callable[[], Any]) -> Any:
        with self._lock:
            future = self._pending.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._pending[key] = future

        if not owner:
            logger.info("Joining in-flight operation %s", key)
            return future.result()

        try:
            result = operation()
        except BaseException as exc:
            # Release every waiter on failure as well. A later independent
            # request may retry; there is no permanent failure/instance cache.
            future.set_exception(exc)
            raise
        else:
            future.set_result(result)
            return result
        finally:
            with self._lock:
                if self._pending.get(key) is future:
                    del self._pending[key]
