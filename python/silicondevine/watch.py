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
from .discovery import capture_project, source_files, project_config
from .assistance import ImportProblem, MARKER, exception_issue, validate_settings


def run_worker(command, timeout):
    """Use one explicit pipe encoding on Windows, including non-Python stderr."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    # Many small operators otherwise oversubscribe CPU threads during capture.
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                            env=env, timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        for line in reversed((result.stdout+'\n'+result.stderr).splitlines()):
            if line.startswith(MARKER):
                try: issue=json.loads(line[len(MARKER):])
                except ValueError: continue
                raise ImportProblem(issue)
        message = '\n'.join(v.strip() for v in (result.stderr, result.stdout) if v and v.strip())
        raise RuntimeError(f'模型捕获进程退出，退出码 {result.returncode}。\n' + (message[-12000:] or '未输出错误详情。'))
    return result


def capture_file(file, factory, destination, backend):
    sys.path.insert(0, str(Path(file).resolve().parent))
    module = runpy.run_path(file, run_name='silicondevine_model')
    if factory not in module or not callable(module[factory]):
        raise ValueError(f'Define {factory}() returning (model, args) or a dictionary.')
    result = module[factory]()
    if isinstance(result, dict) and result.get('format') == 'silicondevine' and result.get('architecture'):
        Path(destination).mkdir(parents=True, exist_ok=True)
        result['live'] = {'tensorWindows':False, 'available':[]}
        Path(destination, 'capture.json').write_text(json.dumps({'model': result, 'manifest': {}}, allow_nan=False), encoding='utf-8')
        return
    specification = result if isinstance(result, dict) else {'model': result[0], 'args': result[1]}
    options = dict(specification.get('options', {}))
    options.setdefault('backend', 'fx' if backend == 'auto' else backend)
    if 'kwargs' in specification:
        options['kwargs'] = specification['kwargs']
    snap = Snapshot(Path(destination) / 'tensors', options.pop('snapshot_budget', 64*1024*1024))
    graph = export_model(specification['model'], specification['args'], include_values=options.pop('include_values', True), tensor_sink=snap, **options)
    Path(destination, 'capture.json').write_text(json.dumps({'model': graph, 'manifest': snap.manifest}, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Live PyTorch → SiliconDevine browser viewer')
    parser.add_argument('sources', type=Path, nargs='+', help='模型目录或一组 Python 文件')
    parser.add_argument('--factory', default='auto', help='auto 自动发现模型类；也可指定原有工厂函数')
    parser.add_argument('--entry', help='自动导入时指定 module:Class')
    parser.add_argument('--config', type=Path, help='可选模型构造与输入 JSON 配置')
    parser.add_argument('--settings', type=Path, help='应用管理的表单设置')
    parser.add_argument('--backend', choices=['auto','fx','export','execution'], default='auto')
    parser.add_argument('--port', type=int, default=5182)
    parser.add_argument('--static-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            overrides = validate_settings(json.loads(args.settings.read_text(encoding='utf-8'))) if args.settings and args.settings.is_file() else {}
            if args.factory == 'auto':
                capture_project(args.sources, args.worker, args.backend, args.entry, args.config, overrides)
            else:
                if len(args.sources) != 1 or not args.sources[0].is_file():
                    raise ValueError('指定工厂函数时请选择一个 Python 文件。')
                capture_file(str(args.sources[0].resolve()), args.factory, args.worker, args.backend)
        except Exception as exc:
            print(MARKER+json.dumps(exception_issue(exc,args.sources),ensure_ascii=False,allow_nan=False),flush=True)
            raise SystemExit(2)
        return
    source_files(args.sources)
    from .discovery import scan_project
    try: watch_roots = {Path(r['root']) for r in scan_project(args.sources)['files']}
    except Exception: watch_roots=set()
    if not args.settings:
        import hashlib
        key=hashlib.sha256('|'.join(str(p.resolve()) for p in args.sources).encode()).hexdigest()[:20]
        args.settings=Path(os.getenv('LOCALAPPDATA',Path.home()))/'SiliconDevine'/'import-profiles'/(key+'.json')
    server = LiveServer(args.port, args.static_dir, args.settings)
    print(server.url, flush=True)
    last_stamp = None
    try:
        while True:
            # Watch neighboring Python modules too; don't traverse environments or frameworks.
            roots = {p.resolve() if p.is_dir() else p.resolve().parent for p in args.sources}
            roots.update(watch_roots)
            # Package roots ensure edits to sibling imports trigger a fresh worker.
            from .discovery import import_identity
            roots.update(import_identity(p.resolve())[1] for p in args.sources if p.is_file())
            files = source_files(sorted(roots))
            _, config_path = project_config(args.sources, args.config, read=False)
            if config_path.is_file():
                files.append(config_path)
            if args.settings.is_file(): files.append(args.settings)
            try:
                stamp = (server.retry_revision,tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(files)[:5000]))
            except OSError:
                # Editors may temporarily rename files during atomic saves.
                time.sleep(.6)
                continue
            if stamp != last_stamp:
                last_stamp = stamp
                with server.lock:
                    server.loading = True
                    server.error=server.issue=None
                directory = Path(tempfile.mkdtemp(prefix='silicondevine-watch-'))
                try:
                    command = [sys.executable, '-m', 'silicondevine.watch', *[str(p.resolve()) for p in args.sources], '--factory', args.factory,
                               '--backend', args.backend, '--worker', str(directory)]
                    if args.entry: command += ['--entry', args.entry]
                    if args.config: command += ['--config', str(args.config.resolve())]
                    command += ['--settings',str(args.settings.resolve())]
                    run_worker(command, args.timeout)
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
                        server.issue=exception_issue(exc,args.sources)
                        server.loading, server.error = False, server.issue['message']
                        if server.issue.get('schema'): server.import_options=server.issue['schema']
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
