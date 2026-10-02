import sys
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'python'),str(root/'examples')]
from silicondevine import export_model
from llm_models import build_tinygpt, build_llama
from set_attention import build_sab,build_isab

for name, factory in [('tinygpt',build_tinygpt),('llama',build_llama),('sab',build_sab),('isab',build_isab)]:
    spec=factory()
    graph=export_model(spec['model'],spec['args'],root/f'demo/public/models/{name}.sd.json',backend='export',include_values=True,name=name)
    print(name,len(graph['nodes']),Counter(n['op'] for n in graph['nodes']))
    print('Opaque:',Counter(n['source'] for n in graph['nodes'] if n['op']=='opaque'))
