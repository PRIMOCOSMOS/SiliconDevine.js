"""Scoped imports for sequential captures of repositories with shared module names."""
from contextlib import contextmanager
import importlib
import sys
import threading

_lock = threading.RLock()


@contextmanager
def project_imports(report):
    roots = sorted({record['root'] for record in report['files']})
    names = {record['module'].split('.')[0] for record in report['files']}
    def belongs(name):
        return name.split('.')[0] in names
    with _lock:
        old_path = sys.path[:]
        previous = {name: module for name, module in sys.modules.copy().items() if belongs(name)}
        for name in previous:
            del sys.modules[name]
        sys.path[:0] = roots
        importlib.invalidate_caches()
        try:
            yield
        finally:
            for name in list(sys.modules):
                if belongs(name):
                    del sys.modules[name]
            sys.modules.update(previous)
            sys.path[:] = old_path
            importlib.invalidate_caches()
