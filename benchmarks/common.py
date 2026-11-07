"""Standard-library statistics and report helpers."""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def distribution(values):
    """Use linear interpolation, retaining empty populations explicitly."""
    if not values:
        return {'count': 0}
    ordered = sorted(values)
    def percentile(q):
        position = (len(ordered) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return {'count': len(values), 'mean': sum(values) / len(values),
            'p50': percentile(.5), 'p95': percentile(.95), 'p99': percentile(.99)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_dir(root):
    path = Path(root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path.mkdir(parents=True, exist_ok=False)
    return path


def write(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def provenance():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
            'benchmark_sources': {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}


def stage_chart(path, totals):
    """Write a standalone diagnostic SVG; never present it as benchmark speed."""
    from html import escape
    rows = sorted(totals.items(), key=lambda pair: -pair[1])
    maximum = max(totals.values(), default=1) or 1
    height = 80 + len(rows) * 34
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="28" font-family="sans-serif" font-size="17">Diagnostic exclusive wall time (synchronized; includes instrumentation overhead)</text>']
    for index, (name, total) in enumerate(rows):
        y = 55 + index * 34
        parts.append(f'<text x="20" y="{y + 16}" font-family="sans-serif" font-size="13">{escape(name)}</text>')
        parts.append(f'<rect x="290" y="{y}" width="{550 * total / maximum:.2f}" height="22" fill="#3678ba"/>')
        parts.append(f'<text x="850" y="{y + 16}" font-family="sans-serif" font-size="13">{total:.2f} ms</text>')
    parts.append('</svg>')
    Path(path).write_text(''.join(parts))
