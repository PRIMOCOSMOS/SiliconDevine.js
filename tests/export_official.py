import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'python'),str(root/'examples')]
from silicondevine import export_model
from official_models import build_llama,build_sab,build_isab
for name,f in [('official_llama',build_llama),('official_sab',build_sab),('official_isab',build_isab)]:
    c=f();g=export_model(c['model'],c['args'],root/f'demo/public/models/{name}.sd.json',name=name,include_values=True,**c['options'])
    print(name,len(g['nodes']),'unrecognized',sum(n['op']=='opaque' for n in g['nodes']),'units',len(g.get('functionalUnits',[])),flush=True)
