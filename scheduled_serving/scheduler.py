"""One execution worker, bounded FIFO admission and cancellable waiting jobs."""
from collections import Counter, deque
from dataclasses import dataclass, field
import threading
import time


class Rejected(Exception):
    """Admission failed; the caller should return an explicit overload response."""


class Expired(Exception):
    """The server-side execution budget has expired."""


class Metrics:
    """Fixed-size histograms and counters; never retain requests or user histories."""
    BOUNDS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000)

    def __init__(self):
        self.lock = threading.Lock()
        self.counters = Counter()
        self.histograms = {}

    def increment(self, name, amount=1):
        with self.lock:
            self.counters[name] += amount

    def observe(self, name, milliseconds):
        with self.lock:
            h = self.histograms.setdefault(name, {'count': 0, 'sum_ms': 0., 'max_ms': 0.,
                                                  'bucket_counts': [0] * (len(self.BOUNDS) + 1)})
            h['count'] += 1
            h['sum_ms'] += milliseconds
            h['max_ms'] = max(h['max_ms'], milliseconds)
            index = next((i for i, b in enumerate(self.BOUNDS) if milliseconds <= b), len(self.BOUNDS))
            h['bucket_counts'][index] += 1

    def snapshot(self):
        with self.lock:
            return {'counters': dict(self.counters), 'histograms': {
                k: {**v, 'bucket_counts': list(v['bucket_counts']),
                    'bucket_upper_ms': [*self.BOUNDS, None]} for k, v in self.histograms.items()}}


@dataclass
class Job:
    request: dict
    deadline: float
    queued_at: float = field(default_factory=time.monotonic)
    event: threading.Event = field(default_factory=threading.Event)
    state: str = 'queued'
    result: object = None
    queue_ms: float = 0.
    core_ms: float = 0.


class Scheduler:
    """Bound active inference to one; deadlines never pretend to preempt CUDA."""
    def __init__(self, recommend, capacity, metrics=None, initialize=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self.recommend, self.capacity = recommend, capacity
        self.metrics = metrics or Metrics()
        self.condition = threading.Condition()
        self.queue = deque()
        self.active = None
        self.closed = False
        self.ready = threading.Event()
        self.initialization_error = None
        self.initialize = initialize
        self.worker = threading.Thread(target=self._run, name='recommendation-worker', daemon=True)
        self.worker.start()
        if not self.ready.wait(30):
            self.close(0)
            raise RuntimeError('Worker initialization timed out')
        if self.initialization_error:
            self.close(0)
            raise RuntimeError('Worker initialization failed') from self.initialization_error

    def submit(self, request, deadline):
        with self.condition:
            if self.closed:
                self.metrics.increment('admission_shutdown')
                raise Rejected('shutting_down')
            if time.monotonic() >= deadline:
                raise Expired('deadline_exceeded')
            if len(self.queue) >= self.capacity:
                self.metrics.increment('admission_queue_full')
                raise Rejected('queue_full')
            job = Job(request, deadline)
            self.queue.append(job)
            self.metrics.increment('admitted')
            self.condition.notify()
            return job

    def cancel(self, job, state='expired'):
        with self.condition:
            if job.state not in {'queued', 'running'}:
                return
            if job.state == 'queued':
                self.queue.remove(job)
                job.queue_ms = (time.monotonic() - job.queued_at) * 1000
                self.metrics.observe('queue_cancelled_ms', job.queue_ms)
            job.state = state
            job.event.set()
            self.metrics.increment('jobs_' + state)

    def wait(self, job):
        job.event.wait(max(0., job.deadline - time.monotonic()))
        with self.condition:
            if not job.event.is_set():
                self.cancel(job)
            # Also reject a finished result whose consumer missed its deadline.
            if job.state == 'done' and time.monotonic() < job.deadline:
                return job.result
            if job.state in {'expired', 'done'}:
                raise Expired('deadline_exceeded')
            if job.state == 'shutdown':
                raise Rejected('shutting_down')
            raise RuntimeError('inference_failed')

    def snapshot(self):
        with self.condition:
            return {'queue_depth': len(self.queue), 'queue_capacity': self.capacity,
                    'active_inference': int(self.active is not None),
                    'ready': not self.closed and self.worker.is_alive()}

    def close(self, grace_seconds=5):
        with self.condition:
            self.closed = True
            for job in list(self.queue):
                self.cancel(job, 'shutdown')
            if self.active is not None:
                self.cancel(self.active, 'shutdown')
            self.condition.notify_all()
        self.worker.join(grace_seconds)
        return not self.worker.is_alive()

    def _run(self):
        try:
            if self.initialize:
                self.initialize()
        except Exception as error:
            self.initialization_error = error
            return
        finally:
            self.ready.set()
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.queue)
                if self.closed:
                    return
                job = self.queue[0]
                if time.monotonic() >= job.deadline:
                    self.cancel(job)
                    continue
                self.queue.popleft()
                self.active = job
                job.state = 'running'
            start = time.monotonic()
            job.queue_ms = (start - job.queued_at) * 1000
            self.metrics.observe('queue_started_ms', job.queue_ms)
            try:
                result = self.recommend(job.request)
                error = False
            except Exception:
                result, error = None, True
                self.metrics.increment('inference_errors')
            job.core_ms = (time.monotonic() - start) * 1000
            self.metrics.observe('core_ms', job.core_ms)
            with self.condition:
                self.active = None
                if job.state != 'running':
                    self.metrics.increment('late_results_discarded')
                elif time.monotonic() >= job.deadline:
                    job.state = 'expired'
                    self.metrics.increment('jobs_expired')
                else:
                    job.state = 'failed' if error else 'done'
                    job.result = result
                    self.metrics.increment('jobs_' + job.state)
                job.event.set()
