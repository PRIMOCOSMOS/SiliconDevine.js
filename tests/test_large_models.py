import ast
import copy
import dataclasses
import json
import math
from pathlib import Path
import sys
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
sys.path.insert(0,str(ROOT/'examples'))
from large_models import build
from silicondevine import architecture_from_config, export_architecture
from silicondevine.watch import capture_file

def count_scope(graph,key):
    s=graph['architecture']['scopes'][key]
    return sum(math.prod(t['shape']) for t in s['tensors'] if t['role']=='parameter')+sum(count_scope(graph,n['attrs']['scopeRef'])*n['attrs'].get('repeat',1) for n in s['nodes'] if 'scopeRef' in n['attrs'])

class LargeModels(unittest.TestCase):
    def test_compact_and_exact_schedules(self):
        for family,layers,experts in [('deepseek_v3',61,256),('glm45',92,160),('minimax_m1',80,32)]:
            g=build(family); a=g['architecture']
            self.assertEqual((a['layers'],a['experts']),(layers,experts))
            self.assertEqual(int(a['parameterCount']),count_scope(g,'root'))
            self.assertLess(len(json.dumps(g)),100000)
            self.assertLess(max(len(s['nodes']) for s in a['scopes'].values()),20)
            self.assertTrue(all('data' not in t for s in a['scopes'].values() for t in s['tensors']))
        a=build('minimax_m1')['architecture']
        self.assertEqual(a['layerSchedule'],([0]*7+[1])*10)
        self.assertEqual([n['attrs']['repeat'] for n in a['scopes']['hybrid_cycle']['nodes']],[7,1])
        deep=build('deepseek_v3')
        self.assertEqual(next(t for t in deep['tensors'] if t['id']=='head:out')['shape'],['B',129280])

    def test_glm_official_meta_modules(self):
        import torch
        from transformers import Glm4MoeConfig, Glm4MoeForCausalLM
        g=build('glm45'); c=copy.deepcopy(g['architecture']['config'])
        # Four real layers cover both dense and expert implementations, with original widths.
        c['num_hidden_layers']=4
        with torch.device('meta'):
            model=Glm4MoeForCausalLM(Glm4MoeConfig(**c))
        exported=export_architecture(model)
        self.assertTrue(exported['provenance']['countMatchesBackbone'])
        self.assertEqual(sum(p.numel() for p in model.model.layers[0].parameters()),count_scope(g,'dense_block'))
        self.assertEqual(sum(p.numel() for p in model.model.layers[3].parameters()),count_scope(g,'moe_block'))
        attention=model.model.layers[0].self_attn
        scope=g['architecture']['scopes']['attention'];ts={t['id']:t for t in scope['tensors']}
        self.assertEqual(list(attention.q_proj.weight.shape),ts['Q:W']['shape'])
        self.assertEqual(list(attention.k_proj.bias.shape),ts['K:bias']['shape'])
        self.assertEqual(list(attention.q_norm.weight.shape),ts['Q_norm:gamma']['shape'])
        self.assertTrue(all(p.is_meta for p in model.parameters()))

    def test_deepseek_official_constructors(self):
        import torch
        from torch import nn
        import torch.distributed as dist
        path=ROOT/'examples/upstream/large_models/deepseek_v3/model.py'
        selected={'ModelArgs','Linear','ColumnParallelLinear','RowParallelLinear','RMSNorm','MLA','MLP','Gate','Expert','MoE','Block'}
        # Compile only audited constructor classes, excluding module-level CUDA kernel imports.
        tree=ast.parse(path.read_text())
        classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in selected]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*classes],type_ignores=[])
        env={'torch':torch,'nn':nn,'dist':dist,'math':math,'dataclass':dataclasses.dataclass,'world_size':1,'rank':0,'attn_impl':'absorb','block_size':128}
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env)
        g=build('deepseek_v3');args=env['ModelArgs'](**g['architecture']['config'])
        with torch.device('meta'):
            dense=env['Block'](0,args);sparse=env['Block'](3,args)
        self.assertEqual(sum(p.numel() for p in dense.parameters()),count_scope(g,'dense_block'))
        self.assertEqual(sum(p.numel() for p in sparse.parameters()),count_scope(g,'moe_block'))
        self.assertEqual(list(sparse.attn.wkv_b.weight.shape),[32768,512])

    def test_minimax_official_constructors(self):
        import torch
        from torch import nn
        from transformers import PretrainedConfig
        folder=ROOT/'examples/upstream/large_models/minimax_m1'
        selected={'MiniMaxM1RMSNorm','MiniMaxM1RotaryEmbedding','MiniMaxM1LightningAttention','MiniMaxM1Attention','MiniMaxM1BlockSparseTop2MLP','MiniMaxM1SparseMoeBlock','MiniMaxM1Config'}
        classes=[]
        for file in ['configuration_minimax_m1.py','modeling_minimax_m1.py']:
            classes.extend(n for n in ast.parse((folder/file).read_text()).body if isinstance(n,ast.ClassDef) and n.name in selected)
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*classes],type_ignores=[])
        env={'torch':torch,'nn':nn,'math':math,'PretrainedConfig':PretrainedConfig,'get_activation_fn':lambda _:torch.nn.functional.silu,'ACT2FN':{'silu':torch.nn.functional.silu}}
        exec(compile(ast.fix_missing_locations(module),'audited-minimax-constructors','exec'),env)
        g=build('minimax_m1');c=env['MiniMaxM1Config'](**g['architecture']['config'])
        with torch.device('meta'):
            lightning=env['MiniMaxM1LightningAttention'](c,0)
            attention=env['MiniMaxM1Attention'](c,7)
            moe=env['MiniMaxM1SparseMoeBlock'](c)
        self.assertEqual(sum(p.numel() for p in lightning.parameters()),count_scope(g,'lightning'))
        self.assertEqual(sum(p.numel() for p in attention.parameters()),count_scope(g,'attention'))
        self.assertEqual(sum(p.numel() for p in moe.parameters()),count_scope(g,'moe'))

    def test_reject_unsupported_variants_and_invalid_schedule(self):
        c=build('minimax_m1')['architecture']['config'];c['attn_type_list']=[0]
        with self.assertRaises(ValueError):architecture_from_config(c,'minimax_m1')
        with self.assertRaises(ValueError):architecture_from_config({},'unverified_model')

    def test_watch_accepts_local_structure_factory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            capture_file(str(ROOT/'examples/large_models.py'),'build_minimax',d,'fx')
            result=json.loads(Path(d,'capture.json').read_text())
            self.assertEqual(result['manifest'],{})
            self.assertEqual(result['model']['architecture']['layers'],80)

if __name__=='__main__': unittest.main()
