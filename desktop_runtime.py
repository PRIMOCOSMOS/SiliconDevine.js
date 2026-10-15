"""Paths and child-process isolation shared by script and frozen Windows builds."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import shutil
import sys
import re


def startup_issue(log, interpreter):
    """Decode startup failures even when PyTorch cannot import and no server exists."""
    missing=re.findall(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)",log)
    if missing:
        name=missing[-1];top=name.split('.')[0]
        packages={'torch':'torch','numpy':'numpy','einops':'einops','yaml':'PyYAML','PIL':'Pillow','transformers':'transformers'}
        return {'code':'dependency_missing','title':'缺少依赖 · '+top,'module':name,'package':packages.get(top),
                'message':f'当前环境找不到 {name}。请选择项目原有的 Python 环境，或按项目依赖清单安装后重试。'}
    if 'DLL load failed' in log or 'WinError 126' in log or 'WinError 127' in log:
        return {'code':'environment_broken','title':'Python 依赖未能加载','message':'依赖已找到，但运行库或 DLL 无法加载。请切换到能正常运行该模型的 Python 环境；详细原因保留在运行日志中。'}
    return {'code':'service_exited','title':'模型服务已退出','message':'服务未能继续运行。请查看运行日志中的最后一条错误；修改代码或更换环境后重试。'}


def install_command(issue, interpreter):
    def quote(value): return "'"+str(value).replace("'","''")+"'"
    base='& '+quote(interpreter)+' -m pip install '
    if issue.get('requirements'):return base+'-r '+quote(issue['requirements'][0])
    if issue.get('package'):return base+quote(issue['package'])
    return ''


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
