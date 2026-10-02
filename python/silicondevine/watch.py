"""python -m silicondevine.watch path/to/model.py --factory build_model"""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import shutil
from .exporter import export_model
from .live import LiveServer, Snapshot


def capture_file(file, factory, destination, backend):
    sys.path.insert(0, str(Path(file).resolve().parent))
    module = runpy.run_path(file, run_name='silicondevine_model')
    if factory not in module or not callable(module[factory]):
        raise ValueError(f'Define {factory}() returning (model, args) or a dictionary.')
    result = module[factory]()
    specification = result if isinstance(result, dict) else {'model': result[0], 'args': result[1]}
    options = dict(specification.get('options', {}))
    options.setdefault('backend', backend)
    if 'kwargs' in specification:
        options['kwargs'] = specification['kwargs']
    snap = Snapshot(Path(destination) / 'tensors', options.pop('snapshot_budget', 64*1024*1024))
    graph = export_model(specification['model'], specification['args'], include_values=options.pop('include_values', True), tensor_sink=snap, **options)
    Path(destination, 'capture.json').write_text(json.dumps({'model': graph, 'manifest': snap.manifest}, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Live PyTorch → SiliconDevine browser viewer')
    parser.add_argument('file', type=Path)
    parser.add_argument('--factory', default='build_model')
    parser.add_argument('--backend', choices=['fx','export'], default='fx')
    parser.add_argument('--port', type=int, default=5182)
    parser.add_argument('--static-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        capture_file(str(args.file.resolve()), args.factory, args.worker, args.backend)
        return
    if not args.file.is_file():
        parser.error('Model file does not exist.')
    server = LiveServer(args.port, args.static_dir)
    print(server.url, flush=True)
    last_stamp = None
    try:
        while True:
            # Watch neighboring Python modules too; don't traverse environments or frameworks.
            files = [p for p in args.file.resolve().parent.rglob('*.py') if not any(s in ('node_modules','.venv','venv','__pycache__','.git') for s in p.parts)]
            stamp = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(files)[:5000])
            if stamp != last_stamp:
                last_stamp = stamp
                with server.lock:
                    server.loading = True
                directory = Path(tempfile.mkdtemp(prefix='silicondevine-watch-'))
                try:
                    process = subprocess.run([sys.executable, '-m', 'silicondevine.watch', str(args.file.resolve()), '--factory', args.factory,
                                              '--backend', args.backend, '--worker', str(directory)],
                                             capture_output=True, text=True, timeout=args.timeout,
                                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    if process.returncode:
                        raise RuntimeError((process.stderr or process.stdout)[-12000:])
                    result = json.loads((directory/'capture.json').read_text(encoding='utf-8'))
                    snapshot = Snapshot(directory/'tensors')
                    snapshot.manifest = result['manifest']
                    previous = server.snapshot
                    server.publish(result['model'], snapshot)
                    if previous:
                        shutil.rmtree(previous.directory.parent, ignore_errors=True)
                    print(f'Updated revision {server.revision}', flush=True)
                except Exception as exc:
                    with server.lock:
                        server.loading, server.error = False, str(exc)
                    shutil.rmtree(directory, ignore_errors=True)
                    print(f'Capture failed; keeping previous graph: {exc}', file=sys.stderr, flush=True)
            time.sleep(.6)
    except KeyboardInterrupt:
        pass
    finally:
        parent = server.snapshot.directory.parent if server.snapshot else None
        server.close()
        if parent:
            shutil.rmtree(parent, ignore_errors=True)


if __name__ == '__main__':
    main()
