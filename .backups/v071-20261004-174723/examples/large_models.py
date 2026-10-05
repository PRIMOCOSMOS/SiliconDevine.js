"""Choose one factory in the launcher. No model weights are downloaded."""
from pathlib import Path
import json
from silicondevine import architecture_from_config, attach_mechanisms

ROOT = Path(__file__).resolve().parent / 'upstream' / 'large_models'

def build(family, mechanisms=False):
    config = ROOT / family / ('config_671B.json' if family == 'deepseek_v3' else 'config.json')
    sources = json.loads((ROOT/'SOURCE.json').read_text(encoding='utf-8'))
    graph = architecture_from_config(json.loads(config.read_text(encoding='utf-8')), family, provenance=sources[family])

    return attach_mechanisms(graph) if mechanisms else graph

def build_deepseek(): return build('deepseek_v3', True)
def build_glm(): return build('glm45', True)
def build_minimax(): return build('minimax_m1', True)

if __name__ == '__main__':
    folder = ROOT.parents[2] / 'demo' / 'public' / 'models'
    folder.mkdir(parents=True, exist_ok=True)
    for family in ('deepseek_v3','glm45','minimax_m1'):
        graph = build(family, True)
        (folder / (family+'.sd.json')).write_text(json.dumps(graph,ensure_ascii=False),encoding='utf-8')
        print(family, graph['architecture']['parameterCount'], 'logical backbone parameters')
