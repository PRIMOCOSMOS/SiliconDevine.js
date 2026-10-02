"""Paths and child-process isolation shared by script and frozen Windows builds."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import shutil
import sys


def application_root():
    return Path(sys.executable if getattr(sys, 'frozen', False) else __file__).resolve().parent


def default_python():
    if not getattr(sys, 'frozen', False):
        return str(Path(sys.executable).with_name('python.exe'))
    candidates = [shutil.which('python.exe')]
    if os.getenv('CONDA_PREFIX'):
        candidates.append(str(Path(os.environ['CONDA_PREFIX']) / 'python.exe'))
    # Frozen sys.executable is this launcher, never a Python interpreter.
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and 'windowsapps' not in candidate.lower():
            return candidate
    return ''


def external_environment():
    env = dict(os.environ)
    if getattr(sys, 'frozen', False):
        for key in list(env):
            if key.startswith('_PYI_') or key in ('_MEIPASS2', 'PYTHONHOME', 'TCL_LIBRARY', 'TK_LIBRARY'):
                env.pop(key, None)
        bundle = str(getattr(sys, '_MEIPASS', '')).lower()
        if bundle:
            for key in ('PATH', 'PYTHONPATH'):
                if key in env:
                    env[key] = os.pathsep.join(p for p in env[key].split(os.pathsep) if not p.lower().startswith(bundle))
    return env


@contextmanager
def external_libraries():
    """Do not leak the bootloader's DLL directory into the external PyTorch process."""
    frozen = os.name == 'nt' and getattr(sys, 'frozen', False)
    if frozen:
        ctypes.windll.kernel32.SetDllDirectoryW(None)
    try:
        yield
    finally:
        if frozen:
            ctypes.windll.kernel32.SetDllDirectoryW(ctypes.c_wchar_p(sys._MEIPASS))
