"""Exercise the actual EXE, from an unrelated working directory, with no console."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

p=argparse.ArgumentParser()
p.add_argument('executable',type=Path)
p.add_argument('--preset',default='MLP')
args=p.parse_args()
exe=args.executable.resolve()
report=Path(__file__).resolve().parents[1]/'.qa'/('packaged-'+str(time.time_ns())+'.json')
report.parent.mkdir(exist_ok=True)
env=dict(os.environ)
env.pop('PYTHONPATH',None)
env.pop('PYTHONHOME',None)
with tempfile.TemporaryDirectory(prefix='sd-exe-cwd-') as directory:
    result=subprocess.run([str(exe),'--self-test',str(report),'--preset',args.preset],
                          cwd=directory,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=190)
data=json.loads(report.read_text(encoding='utf-8'))
assert result.returncode==0 and data['ok'],data
assert data['frozen'] and Path(data['root'])==exe.parent,data
assert data['detectedPython'] and Path(data['detectedPython'])!=exe,data
print(json.dumps(data,ensure_ascii=False))
print('Report: '+str(report))
