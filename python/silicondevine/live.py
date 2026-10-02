"""Loopback-only, read-only transport for trusted local PyTorch programs."""
import json
import os
import struct
import tempfile
import threading
import time
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .exporter import export_model


class Snapshot:
    def __init__(self, directory, budget=64 * 1024 * 1024):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.remaining = budget
        self.manifest = {}

    def __call__(self, name, tensor):
        size = tensor.numel() * 8
        if size > self.remaining:
            return
        key = f'{len(self.manifest)}.bin'
        array = tensor.detach().cpu().to(dtype=__import__('torch').float64).contiguous().numpy()
        self.directory.joinpath(key).write_bytes(array.astype('<f8', copy=False).tobytes())
        self.manifest[name] = {'file': key, 'count': tensor.numel()}
        self.remaining -= size


class LiveServer:
    """Atomic snapshots: failed recaptures never destroy the last usable graph."""
    def __init__(self, port=5182, static_dir=None):
        self.static_dir = Path(static_dir) if static_dir else Path(__file__).resolve().parents[2] / 'demo-dist'
        self.lock = threading.RLock()
        self.graph = None
        self.snapshot = None
        self.revision = 0
        self.loading = False
        self.error = None
        self.closed = False
        server_state = self

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(server_state.static_dir), **kwargs)

            def log_message(self, *args):
                pass

            def end_headers(self):
                origin = self.headers.get('Origin')
                if origin and urlparse(origin).hostname in ('127.0.0.1', 'localhost', '::1'):
                    self.send_header('Access-Control-Allow-Origin', origin)
                    self.send_header('Vary', 'Origin')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                super().end_headers()

            def reply(self, payload, status=200):
                raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                host = urlparse('http://' + self.headers.get('Host', '')).hostname
                origin = self.headers.get('Origin')
                if host not in ('localhost', '127.0.0.1', '::1') or origin and urlparse(origin).hostname not in ('localhost', '127.0.0.1', '::1'):
                    return self.reply({'error': 'Only local browser origins are allowed.'}, 403)
                route = urlparse(self.path)
                with server_state.lock:
                    if route.path == '/api/status':
                        return self.reply({'revision': server_state.revision, 'ready': server_state.graph is not None,
                                           'loading': server_state.loading, 'error': server_state.error})
                    if route.path == '/api/model':
                        if server_state.graph is None:
                            return self.reply({'error': server_state.error or 'Waiting for the model.'}, 503)
                        return self.reply({'revision': server_state.revision, 'model': server_state.graph})
                    if route.path == '/api/tensor':
                        try:
                            query = parse_qs(route.query)
                            if int(query.get('revision', ['-1'])[0]) != server_state.revision:
                                return self.reply({'error': 'Model changed. Reload this tensor window.'}, 409)
                            name = query['id'][0]
                            indices = [int(i) for i in query['indices'][0].split(',')]
                            snap = server_state.snapshot
                            entry = snap.manifest.get(name) if snap else None
                            if not entry:
                                return self.reply({'error': 'Tensor was not retained within the snapshot budget.'}, 404)
                            if not 1 <= len(indices) <= 2048 or any(i < 0 or i >= entry['count'] for i in indices):
                                raise ValueError('Invalid tensor coordinates (limit 2048).')
                            import math
                            with (snap.directory / entry['file']).open('rb') as data:
                                values = []
                                for index in indices:
                                    data.seek(index * 8)
                                    value = struct.unpack('<d', data.read(8))[0]
                                    values.append(value if math.isfinite(value) else None)
                            return self.reply({'revision': server_state.revision, 'id': name, 'indices': indices, 'values': values})
                        except (ValueError, KeyError, OSError) as exc:
                            return self.reply({'error': str(exc)}, 400)
                if route.path.startswith('/api/'):
                    return self.reply({'error': 'Unknown API.'}, 404)
                # No filesystem paths or model scripts can be requested outside demo-dist.
                super().do_GET()

        self.http = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.port = self.http.server_address[1]
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.port}/?live=1'

    def publish(self, graph, snapshot=None):
        with self.lock:
            graph['live'] = {'tensorWindows': bool(snapshot), 'available': list(snapshot.manifest) if snapshot else []}
            self.graph, self.snapshot = graph, snapshot
            self.revision += 1
            self.error, self.loading = None, False

    def update(self, model, args, **options):
        directory = tempfile.mkdtemp(prefix='silicondevine-snapshot-')
        snap = Snapshot(directory, options.pop('snapshot_budget', 64 * 1024 * 1024))
        with self.lock:
            self.loading = True
        try:
            graph = export_model(model, args, include_values=options.pop('include_values', True), tensor_sink=snap, **options)
            previous = self.snapshot
            self.publish(graph, snap)
            if previous:
                import shutil
                shutil.rmtree(previous.directory)
            return self
        except Exception as exc:
            import shutil
            shutil.rmtree(directory)
            with self.lock:
                self.error, self.loading = str(exc), False
            raise

    def wait(self):
        try:
            while not self.closed:
                time.sleep(.25)
        except KeyboardInterrupt:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=2)
        if self.snapshot:
            import shutil
            shutil.rmtree(self.snapshot.directory, ignore_errors=True)


def show(model, args, *, port=5182, block=True, static_dir=None, **options):
    """Use native torch.nn.Module in any IDE: show(model, (example,))."""
    server = LiveServer(port, static_dir)
    try:
        server.update(model, args, **options)
    except Exception:
        server.close()
        raise
    print(server.url, flush=True)
    if block:
        server.wait()
    return server
