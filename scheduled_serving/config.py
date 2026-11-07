"""Explicit bounds for the loopback scheduling experiment."""
import json
from pathlib import Path


def load_config(path):
    config = json.loads(Path(path).read_text())
    keys = {'runtime_config', 'host', 'port', 'queue_capacity', 'max_connections',
            'listen_backlog', 'request_timeout_ms', 'response_reserve_ms',
            'shutdown_grace_seconds', 'output_dir'}
    if set(config) != keys or config['host'] != '127.0.0.1':
        raise ValueError('Expected documented scheduling config with loopback host')
    for name, lo, hi in [('port', 1, 65535), ('queue_capacity', 1, 1024),
                         ('max_connections', 2, 256), ('listen_backlog', 1, 1024),
                         ('request_timeout_ms', 20, 10000), ('response_reserve_ms', 1, 1000),
                         ('shutdown_grace_seconds', 1, 30)]:
        if type(config[name]) is not int or not lo <= config[name] <= hi:
            raise ValueError('Invalid ' + name)
    if config['response_reserve_ms'] >= config['request_timeout_ms']:
        raise ValueError('Response reserve must be smaller than the request deadline')
    if config['max_connections'] <= config['queue_capacity'] + 1:
        raise ValueError('Leave handler capacity for overload responses and health checks')
    for name in ['runtime_config', 'output_dir']:
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError('Expected nonempty ' + name)
    return config
