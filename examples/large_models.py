"""Choose one factory in the launcher. No model weights are downloaded."""
from pathlib import Path
import json
from silicondevine import architecture_from_config, attach_mechanisms

ROOT = Path(__file__).resolve().parent / 'upstream' / 'large_models'

def build(family, mechanisms=False):
    config = ROOT / family / ('config_671B.json' if family == 'deepseek_v3' else 'config.json')
    sources = json.loads((ROOT/'SOURCE.json').read_text(encoding='utf-8'))
    graph = architecture_from_config(json.loads(config.read_text(encoding='utf-8')), family, provenance=sources[family])

    if not mechanisms:return graph
    attach_mechanisms(graph)
    folder=ROOT.parents[2]/'demo/public/models'
    for suffix,keys in [('',('moe_block','attention','moe','expert','router','shared_expert')),('_lightning',('lightning_block','lightning'))] if family=='minimax_m1' else [('',('moe_block','attention','moe','expert','router','shared_expert')),('_dense',('dense_block','dense_ffn'))]:
        path=folder/(family+suffix+'.source.json')
        if not path.exists():raise FileNotFoundError('Run examples/source_models.py to capture upstream decoder code: '+str(path))
        ref='source'+suffix;execution=json.loads(path.read_text(encoding='utf-8'))
        graph['architecture']['mechanisms'][ref]=execution
        moe='decoder.block_sparse_moe' if family=='minimax_m1' else 'decoder.mlp'
        routes={'attention':'decoder.self_attn','lightning':'decoder.self_attn','moe':moe,'dense_ffn':moe,'router':moe+'.gate',
                'expert':moe+'.experts.0','shared_expert':moe+'.shared_experts'}
        modules={m['path'] for m in execution['modules']}
        for key in keys:
            if key not in graph['architecture']['scopes']:continue
            scope=graph['architecture']['scopes'][key];scope['executionRef']=ref
            scope['executionScope']=routes.get(key,'decoder') if routes.get(key,'decoder') in modules else moe
    return graph

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
