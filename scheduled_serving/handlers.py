"""A fixed pool used only after the HTTP server has admitted a connection."""
from queue import SimpleQueue
import threading
import time


class HandlerPool:
    def __init__(self, size, handle):
        self.handle = handle
        self.requests = SimpleQueue()
        self.threads = []
        try:
            for index in range(size):
                thread = threading.Thread(target=self._work, name=f'http-handler-{index}', daemon=True)
                thread.start()
                self.threads.append(thread)
        except BaseException:
            self.close(1)
            raise

    def submit(self, request, address):
        # The server holds one of at most "size" slots until this handler finishes.
        self.requests.put((request, address))

    def _work(self):
        while True:
            item = self.requests.get()
            if item is None:
                return
            self.handle(*item)

    def close(self, grace):
        # The accept loop has stopped. Each worker consumes exactly one sentinel.
        for _ in self.threads:
            self.requests.put(None)
        deadline = time.monotonic() + grace
        for thread in self.threads:
            thread.join(max(0., deadline - time.monotonic()))
        return all(not thread.is_alive() for thread in self.threads)
