import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'python'),str(root/'examples')]
from silicondevine import export_model
from vae_models import build_official_vae,build_conv_vae,build_conditional_vae
graphs=[]
for name,factory in [('official_vae',build_official_vae),('conv_vae',build_conv_vae),('conditional_vae',build_conditional_vae)]:
    spec=factory()
    graph=export_model(spec['model'],spec['args'],root/f'demo/public/models/{name}.sd.json',name=name,include_values=True,**spec['options'])
    assert all(n['op']!='opaque' for n in graph['nodes']),[(n['op'],n['source']) for n in graph['nodes']]
    units=[u for u in graph['functionalUnits'] if u['kind']=='gaussian_reparameterization']
    assert len(units)==1,graph['functionalUnits']
    tensors={t['id']:t for t in graph['tensors']};u=units[0]
    mu,sigma,eps,z=[tensors[u[k]]['data']['values'] for k in ('mean','std','noise','sample')]
    assert all(abs(m+s*e-y)<1e-5 for m,s,e,y in zip(mu,sigma,eps,z))
    graphs.append(graph)
    print(name,len(graph['nodes']),'operators; actual sample arithmetic verified',flush=True)
(root/'.qa/vae-cases.json').write_text(json.dumps(graphs),encoding='utf-8')
