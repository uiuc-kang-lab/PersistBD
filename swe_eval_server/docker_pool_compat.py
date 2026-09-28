"""Keep Unix Docker pool disposal from racing with connection return.

Scoped to clients created here. No global dependency patch or request retry.
"""

import threading

import docker
from docker.transport.unixconn import UnixHTTPAdapter, UnixHTTPConnectionPool


class ReturnSafeUnixHTTPConnectionPool(UnixHTTPConnectionPool):
    def __init__(self, *args, **kwargs):
        self._return_close_lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def _put_conn(self, conn):
        # urllib3 2.6.3 reads self.pool.qsize() after closing an overflow
        # connection. Adapter eviction can otherwise set self.pool to None
        # during that close, turning a diagnostic warning into a failed request.
        with self._return_close_lock:
            return super()._put_conn(conn)

    def close(self):
        with self._return_close_lock:
            return super().close()


class ReturnSafeUnixHTTPAdapter(UnixHTTPAdapter):
    def get_connection(self, url, proxies=None):
        with self.pools.lock:
            pool = self.pools.get(url)
            if pool is None:
                pool = ReturnSafeUnixHTTPConnectionPool(
                    url, self.socket_path, self.timeout,
                    maxsize=self.max_pool_size,
                )
                self.pools[url] = pool
            return pool


def make_session_docker_client():
    client = docker.from_env()
    original = getattr(client.api, "_custom_adapter", None)
    if type(original) is not UnixHTTPAdapter:
        return client
    replacement = ReturnSafeUnixHTTPAdapter(
        "http+unix://" + original.socket_path,
        timeout=original.timeout,
        pool_connections=original.pools._maxsize,
        max_pool_size=original.max_pool_size,
    )
    # The new client has not been handed to any worker yet. Preserve the
    # negotiated API version, environment, pool capacity and timeout.
    client.api.mount("http+docker://", replacement)
    client.api._custom_adapter = replacement
    original.close()
    return client
