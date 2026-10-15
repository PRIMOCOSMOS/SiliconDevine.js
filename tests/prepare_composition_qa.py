import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'python'),r'D:\WUDAN_RESEARCH_1\MMHVAE',str(root/'tests')]
from silicondevine.composition import annotate_composition
from silicondevine.exporter import _op
from network.mhvae import MHVAE2D
from test_composition import check
graph=json.loads((root/'.qa/mmhvae-model.json').read_text(encoding='utf-8'))
annotate_composition(graph,MHVAE2D.__new__(MHVAE2D))
for n in graph['nodes']:
    if n['op']=='opaque':n['op']=_op(n['source'])
check(graph)
(root/'.qa/mmhvae-composition.json').write_text(json.dumps(graph),encoding='utf-8')
index={u['id']:u for u in graph['composition']['units']}
print(json.dumps({'roots':[(index[k]['name'],len(index[k]['nodes']),len(index[k]['children'])) for k in graph['composition']['roots']], 'units':len(index),'hidden':graph['composition']['hiddenDiagnostics']},ensure_ascii=False))
