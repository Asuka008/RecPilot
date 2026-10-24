"""A bounded, loopback-only sequential HTTP adapter for local demonstrations."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


def make_handler(engine):
    """Keep the HTTP contract testable without loading a model."""
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def send_json(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                return self.send_json(404, {'error': 'not_found'})
            self.send_json(200, {'status': 'ready', 'model_id': engine.identity, 'catalog_items': engine.num_items})

        def do_POST(self):
            if self.path != '/recommend':
                return self.send_json(404, {'error': 'not_found'})
            if self.headers.get('Transfer-Encoding'):
                return self.send_json(400, {'error': 'transfer_encoding_not_supported'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    return self.send_json(413, {'error': 'body_size_limit'})
                request = json.loads(self.rfile.read(size))
                response = engine.recommend(request)
            except (ValueError, UnicodeError, TypeError) as error:
                return self.send_json(400, {'error': str(error)})
            except Exception:
                return self.send_json(500, {'error': 'inference_failed'})
            self.send_json(200, response)

        def log_message(self, format, *args):
            # Avoid logging request histories or per-request benchmark noise.
            pass
    return Handler


def serve(engine, config):
    """Serialize requests; do not imply measured concurrent serving capacity."""
    with HTTPServer((config['host'], config['port']), make_handler(engine)) as server:
        print(f"Ready: http://{config['host']}:{config['port']} (sequential local demo)", flush=True)
        server.serve_forever()
