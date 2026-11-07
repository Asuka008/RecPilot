"""Bounded HTTP handlers with an accept-to-close watchdog and one model worker."""
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import socket
from socketserver import ThreadingMixIn
import threading
import time
import uuid

from scheduled_serving.scheduler import Expired, Rejected


@dataclass
class Connection:
    accepted: float
    deadline: float
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    lock: object = field(default_factory=threading.Lock)
    finished: bool = False
    timer: object = None


def stop_socket(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class BoundedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, address, scheduler, validate, identity, catalog, config, manifest):
        self.scheduler, self.validate = scheduler, validate
        self.identity, self.catalog = identity, catalog
        self.config = config
        self.manifest = {**manifest, 'instance_id': uuid.uuid4().hex}
        self.metrics = scheduler.metrics
        self.slots = threading.BoundedSemaphore(config['max_connections'])
        self.connections = {}
        self.connections_lock = threading.Condition()
        self.request_queue_size = config['listen_backlog']
        super().__init__(address, Handler)

    def get_request(self):
        sock, address = super().get_request()
        now = time.monotonic()
        context = Connection(now, now + self.config['request_timeout_ms'] / 1000)
        with self.connections_lock:
            self.connections[sock] = context
        self.metrics.increment('connections_accepted')
        return sock, address

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.metrics.increment('connection_limit_rejected')
            body = b'{"error":"connection_limit"}'
            response = (b'HTTP/1.0 503 Service Unavailable\r\nConnection: close\r\n'
                        b'Content-Type: application/json\r\nRetry-After: 1\r\nContent-Length: '
                        + str(len(body)).encode() + b'\r\n\r\n' + body)
            try:
                request.settimeout(.01)
                request.sendall(response)
                self.metrics.increment('connection_limit_response_written')
            except OSError:
                self.metrics.increment('connection_limit_write_failed')
            finally:
                self.shutdown_request(request)
                with self.connections_lock:
                    self.connections.pop(request, None)
                    self.connections_lock.notify_all()
            return
        with self.connections_lock:
            context = self.connections[request]
        def watchdog():
            with context.lock:
                if context.finished:
                    return
                self.metrics.increment('connection_deadline_closed')
                stop_socket(request)
        context.timer = threading.Timer(max(0., context.deadline - time.monotonic()), watchdog)
        context.timer.daemon = True
        context.timer.start()
        try:
            super().process_request(request, client_address)
        except Exception:
            context.timer.cancel()
            self.shutdown_request(request)
            self._release(request, context)
            raise

    def _release(self, sock, context):
        with context.lock:
            context.finished = True
            if context.timer:
                context.timer.cancel()
        self.metrics.observe('accepted_connection_ms', (time.monotonic() - context.accepted) * 1000)
        with self.connections_lock:
            self.connections.pop(sock, None)
            self.connections_lock.notify_all()
        self.slots.release()

    def process_request_thread(self, request, client_address):
        with self.connections_lock:
            context = self.connections[request]
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release(request, context)

    def handle_error(self, request, client_address):
        # Bound log volume and never include request bodies in exception logs.
        self.metrics.increment('handler_errors')

    def snapshot(self):
        snapshot = self.metrics.snapshot()
        snapshot['scheduler'] = self.scheduler.snapshot()
        with self.connections_lock:
            snapshot['connections'] = len(self.connections)
        snapshot['model_id'] = self.identity
        snapshot['service_id'] = self.manifest['service_id']
        snapshot['instance_id'] = self.manifest['instance_id']
        return snapshot

    def stop(self):
        """Call after serve_forever stops; invalidate jobs and bound shutdown wait."""
        drained = self.scheduler.close(self.config['shutdown_grace_seconds'])
        with self.connections_lock:
            for sock in list(self.connections):
                stop_socket(sock)
        super().server_close()
        with self.connections_lock:
            handlers_drained = self.connections_lock.wait_for(lambda: not self.connections,
                                                              self.config['shutdown_grace_seconds'])
        return {'worker_drained': drained, 'handlers_drained': handlers_drained}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def setup(self):
        super().setup()
        with self.server.connections_lock:
            self.context = self.server.connections[self.connection]
        self.connection.settimeout(max(.001, self.context.deadline - time.monotonic()))
        self.close_connection = True

    def handle(self):
        try:
            super().handle()
        except (OSError, TimeoutError):
            self.server.metrics.increment('transport_read_or_disconnect_errors')

    def send_json(self, status, value, job=None):
        server = self.server
        body = json.dumps(value, allow_nan=False).encode()
        server.metrics.increment('http_attempt_' + str(status))
        if self.path == '/recommend':
            server.metrics.increment('recommend_attempt_' + str(status))
        try:
            remaining = self.context.deadline - time.monotonic()
            if remaining <= 0:
                server.metrics.increment('response_deadline_missed')
                return
            self.connection.settimeout(remaining)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close')
            self.send_header('X-Request-ID', self.context.request_id)
            self.send_header('X-Service-ID', server.manifest['service_id'])
            if status == 503:
                self.send_header('Retry-After', '1')
            if job:
                self.send_header('Server-Timing', f'queue;dur={job.queue_ms:.3f}, core;dur={job.core_ms:.3f}')
            self.end_headers()
            self.wfile.write(body)
            server.metrics.increment('http_written_' + str(status))
            if self.path == '/recommend':
                server.metrics.increment('recommend_written_' + str(status))
        except (OSError, TimeoutError):
            server.metrics.increment('response_write_failed')
        finally:
            server.metrics.observe('response_attempt_ms', (time.monotonic() - self.context.accepted) * 1000)

    def do_GET(self):
        if self.path == '/health':
            ready = self.server.scheduler.snapshot()['ready']
            return self.send_json(200 if ready else 503,
                {'status': 'ready' if ready else 'stopping', 'model_id': self.server.identity,
                 'catalog_items': self.server.catalog, **self.server.manifest})
        if self.path == '/metrics':
            return self.send_json(200, self.server.snapshot())
        return self.send_json(404, {'error': 'not_found'})

    def do_POST(self):
        if self.path != '/recommend':
            return self.send_json(404, {'error': 'not_found'})
        server = self.server
        server.metrics.increment('recommend_requests')
        job = None
        try:
            if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1:
                raise ValueError('Expected one Content-Length and no Transfer-Encoding')
            size = int(self.headers['Content-Length'])
            if not 0 < size <= 65536:
                return self.send_json(413, {'error': 'body_size_limit'})
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError('Incomplete request body')
            request = json.loads(raw)
            server.validate(request)
            deadline = self.context.deadline - server.config['response_reserve_ms'] / 1000
            job = server.scheduler.submit(request, deadline)
            result = server.scheduler.wait(job)
            self.send_json(200, result, job)
        except (ValueError, UnicodeError, TypeError):
            self.send_json(400, {'error': 'invalid_request'})
        except Rejected as error:
            self.send_json(503, {'error': str(error)})
        except Expired:
            self.send_json(504, {'error': 'deadline_exceeded'})
        except (OSError, TimeoutError):
            server.metrics.increment('transport_read_or_disconnect_errors')
        except Exception:
            self.send_json(500, {'error': 'inference_failed'})
        finally:
            if job:
                server.scheduler.cancel(job)

    def log_message(self, format, *args):
        pass
