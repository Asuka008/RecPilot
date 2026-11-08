"""One watchdog thread for the bounded set of admitted HTTP sockets."""
import threading
import time


class DeadlineMonitor:
    def __init__(self, expire):
        self.expire = expire
        self.condition = threading.Condition()
        self.pending = {}
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name='connection-deadlines', daemon=True)
        self.thread.start()

    def register(self, sock, context):
        with self.condition:
            self.pending[sock] = context
            if len(self.pending) == 1:
                self.condition.notify()

    def discard(self, sock):
        with self.condition:
            self.pending.pop(sock, None)

    def close(self, grace):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        self.thread.join(grace)
        return not self.thread.is_alive()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopping or self.pending)
                if self.stopping:
                    return
                # One accept loop and a fixed budget give insertion order by deadline.
                sock, context = next(iter(self.pending.items()))
                remaining = context.deadline - time.monotonic()
                if remaining > 0:
                    self.condition.wait(remaining)
                    continue
                del self.pending[sock]
            self.expire(sock, context)
