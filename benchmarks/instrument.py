"""Temporary call wrappers with nested exclusive, synchronized wall timings."""
from contextlib import contextmanager
from functools import wraps
import time


class Instrument:
    def __init__(self, torch, device, mode):
        self.torch, self.device, self.mode = torch, device, mode
        self.samples, self.stack, self.restore = {}, [], []

    def sync(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)

    @contextmanager
    def region(self, name):
        if self.mode == 'plain':
            yield
            return
        with self.torch.profiler.record_function('agenticrec/' + name):
            if self.mode == 'stages':
                self.sync()
                frame = [time.perf_counter(), 0.0]
                self.stack.append(frame)
            try:
                yield
            finally:
                if self.mode == 'stages':
                    self.sync()
                    elapsed = time.perf_counter() - frame[0]
                    self.stack.pop()
                    self.samples.setdefault(name, []).append(max(0., elapsed - frame[1]) * 1000)
                    if self.stack:
                        self.stack[-1][1] += elapsed

    def wrap(self, obj, attribute, label):
        original = getattr(obj, attribute)
        # Restore inherited methods without leaving shadow attributes behind.
        own = attribute in vars(obj)
        @wraps(original)
        def wrapped(*args, **kwargs):
            with self.region(label):
                return original(*args, **kwargs)
        setattr(obj, attribute, wrapped)
        self.restore.append((obj, attribute, original, own))

    def close(self):
        for obj, attribute, original, own in reversed(self.restore):
            if own:
                setattr(obj, attribute, original)
            else:
                delattr(obj, attribute)
