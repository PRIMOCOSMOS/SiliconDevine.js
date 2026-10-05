"""Choose one factory in the launcher. No model weights are downloaded."""
from pathlib import Path
import json
from silicondevine import architecture_from_config

ROOT = Path(__file__).resolve().parent / 'upstream' / 'large_models'

def build(family):
    config = ROOT / family / ('config_671B.json' if family == 'deepseek_v3' else 'config.json')
    sources = json.loads((ROOT/'SOURCE.json').read_text(encoding='utf-8'))
    return architecture_from_config(json.loads(config.read_text(encoding='utf-8')), family, provenance=sources[family])

def build_deepseek(): return build('deepseek_v3')
def build_glm(): return build('glm45')
def build_minimax(): return build('minimax_m1')

if __name__ == '__main__':
    folder = ROOT.parents[2] / 'demo' / 'public' / 'models'
    folder.mkdir(parents=True, exist_ok=True)
    for family in ('deepseek_v3','glm45','minimax_m1'):
        graph = build(family)
        (folder / (family+'.sd.json')).write_text(json.dumps(graph,ensure_ascii=False),encoding='utf-8')
        print(family, graph['architecture']['parameterCount'], 'logical backbone parameters')
