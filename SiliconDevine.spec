# Build from the project directory: python -m PyInstaller SiliconDevine.spec
from pathlib import Path
import os
import sys
root=Path(SPECPATH)
# Conda stores Tk/ctypes dependencies outside the interpreter's DLLs directory.
conda_bin=Path(sys.base_prefix)/'Library'/'bin'
binaries=[]
if conda_bin.is_dir():
    os.environ['PATH']=str(conda_bin)+os.pathsep+os.environ.get('PATH','')
    binaries=[(str(conda_bin/name),'.') for name in ('ffi.dll','tk86t.dll','tcl86t.dll','libmpdec-4.dll') if (conda_bin/name).exists()]
a=Analysis([str(root/'desktop_entry.py')],pathex=[str(root)],binaries=binaries,
           datas=[],hiddenimports=[],hookspath=[],hooksconfig={},runtime_hooks=[],
           excludes=['torch','transformers','numpy','pandas','matplotlib','PIL','pytest'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='SiliconDevine',debug=False,
        bootloader_ignore_signals=False,strip=False,upx=False,console=False,
        disable_windowed_traceback=False,icon=str(root/'launcher.ico'),
        version=str(root/'desktop-version.txt'))
