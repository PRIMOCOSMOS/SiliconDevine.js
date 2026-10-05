"""Deploy verified staged files without deleting user content or overwriting concurrent edits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

stage=Path(__file__).resolve().parents[1]
target=Path(r'D:\SiliconDevine.js').resolve()
assert target == Path(r'D:\SiliconDevine.js').resolve() and stage != target
baseline=json.loads((stage/'.qa/deployment-baseline.json').read_text())
previous_report=stage/'.qa/deployment-v06.json'
previous=json.loads(previous_report.read_text()).get('files',{}) if previous_report.exists() else {}
baseline.update(previous)
parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
directories={'src','python','examples','docs','dist','demo','demo-dist','THIRD_PARTY_LICENSES','tests','assets'}
roots={key for key in baseline if '/' not in key}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
changes=[];conflicts=[]
for source in stage.rglob('*'):
    if not source.is_file():continue
    rel=source.relative_to(stage);key=rel.as_posix()
    if any(part in {'node_modules','__pycache__','.qa','.git','.build-exe'} for part in rel.parts):continue
    if rel.parts[0] not in directories and key not in roots:continue
    dest=(target/rel).resolve()
    if not dest.is_relative_to(target):raise RuntimeError('Path escaped deployment directory')
    new=digest(source);old=digest(dest) if dest.exists() else None
    if old==new:continue
    if old is not None and old!=baseline.get(key):conflicts.append(key)
    changes.append((source,dest,key,new))
if conflicts:raise RuntimeError('Target changed since staging: '+', '.join(conflicts))
print(f'{len(changes)} changed/new files; target {target}; no concurrent changes.')
if args.apply:
    backup=target/'.backups'/('v06-'+time.strftime('%Y%m%d-%H%M%S'))
    for source,dest,key,new in changes:
        if dest.exists():
            saved=backup/key;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,saved)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if digest(dest)!=new:raise RuntimeError('Copy verification failed: '+key)
    report={'version':'0.6.0','target':str(target),'backup':str(backup),'files':{**previous,**{key:new for _,_,key,new in changes}}}
    (stage/'.qa/deployment-v06.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (target/'DEPLOYMENT-v06.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Deployed, backed up and SHA-256 verified.')
