"""Execute upstream decoder code with small, untrained configurations.

No hand-written graph or substituted attention/FFN equations. MiniMax classes
are loaded from the pinned local source without importing its CUDA dependencies.
"""
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import types
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).parent/'upstream/large_models'

def minimax_symbols():
    file=ROOT/'minimax_m1/modeling_minimax_m1.py'
    names={'MiniMaxM1RMSNorm','MiniMaxM1RotaryEmbedding','rotate_half','apply_rotary_pos_emb','repeat_kv',
           'MiniMaxM1Attention','MiniMaxM1LightningAttention','MiniMaxM1BlockSparseTop2MLP','MiniMaxM1SparseMoeBlock','MiniMaxM1DecoderLayer'}
    nodes=[n for n in ast.parse(file.read_text(encoding='utf-8')).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    def rearrange(x,pattern):
        if pattern!='b h n d -> b n (h d)':raise ValueError('Unsupported einops layout')
        return x.transpose(1,2).flatten(2)
    env={'__name__':__name__,'torch':torch,'nn':nn,'F':F,'math':math,'rearrange':rearrange,'do_eval':False,
         'ACT2FN':{'silu':F.silu},'get_activation_fn':lambda _:F.silu}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])),str(file),'exec'),env)
    # Original decoder chooses the FlashAttention wrapper. The CPU test calls
    # the upstream eager implementation, with identical weights and interface.
    class Eager(env['MiniMaxM1Attention']):
        def forward(self,hidden_states,attn_mask=None,**kwargs):
            kwargs.pop('slope_rate',None)
            return super().forward(hidden_states,attention_mask=attn_mask,**kwargs)
    env['MiniMaxM1FlashAttention2']=Eager
    return env,file

class DecoderInput(nn.Module):
    def __init__(self,decoder,family,lightning=False):
        super().__init__();self.decoder=decoder;self.family=family;self.lightning=lightning
    def forward(self,x,cos,sin,mask):
        if self.family=='minimax_m1':
            # Decode recurrence receives a genuine initial KV state as an input
            # buffer; the source still executes its own per-token update loop.
            return self.decoder(x,attention_mask=None if self.lightning else mask,
                position_ids=torch.arange(x.shape[1])[None],
                past_key_value=torch.zeros(1,2,4,4) if self.lightning else None,
                slope_rate=torch.full((2,1,1),.5) if self.lightning else None)[0]
        return self.decoder(x,attention_mask=mask,position_embeddings=(cos,sin),use_cache=False)

def build_source(family,lightning=False,dense=False):
    torch.manual_seed(73)
    if family=='minimax_m1':
        env,file=minimax_symbols()
        c=types.SimpleNamespace(hidden_size=8,intermediate_size=12,num_attention_heads=2,num_key_value_heads=1,
            head_dim=4,rotary_dim=2,max_position_embeddings=16,rope_theta=10000.,attention_dropout=0.,
            num_local_experts=4,num_experts_per_tok=2,router_jitter_noise=0.,hidden_act='silu',rms_norm_eps=1e-6,
            attention_type=0 if lightning else 1,postnorm=True)
        decoder=env['MiniMaxM1DecoderLayer'](c,0)
        provenance={'repository':json.loads((ROOT/'SOURCE.json').read_text())['minimax_m1'],'implementation':str(file),
            'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'adapter':'AST loads original class bodies; eager CPU attention / original decode recurrence'}
    else:
        import transformers
        if family=='deepseek_v3':
            from transformers import DeepseekV3Config as Config
            from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer as Decoder
            c=Config(hidden_size=8,intermediate_size=12,moe_intermediate_size=6,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
                q_lora_rank=4,kv_lora_rank=4,qk_nope_head_dim=2,qk_rope_head_dim=2,v_head_dim=2,n_routed_experts=4,n_shared_experts=1,
                num_experts_per_tok=2,n_group=2,topk_group=1,first_k_dense_replace=0,attention_dropout=0.)
        else:
            from transformers import Glm4MoeConfig as Config
            from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer as Decoder
            c=Config(hidden_size=8,intermediate_size=12,moe_intermediate_size=6,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
                head_dim=2,n_routed_experts=4,n_shared_experts=1,num_experts_per_tok=2,n_group=2,topk_group=1,first_k_dense_replace=0,
                use_qk_norm=True,attention_dropout=0.)
        if dense:c.first_k_dense_replace=1
        c._attn_implementation='eager';decoder=Decoder(c,0)
        file=Path(inspect.getsourcefile(Decoder))
        provenance={'repository':'https://github.com/huggingface/transformers','version':transformers.__version__,
            'implementation':Decoder.__module__+'.'+Decoder.__name__,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'adapter':'upstream constructor and forward unchanged; eager attention; reduced config'}
    model=DecoderInput(decoder,family,lightning).eval()
    provenance.update(weights='seed 73; random, untrained',capture='one executed decoder path; not a full-size checkpoint',family=family)
    model._silicondevine_source=provenance
    angle=torch.arange(3).float().reshape(1,3,1).expand(1,3,2)
    args=(torch.randn(1,3,8),angle.cos(),angle.sin(),torch.full((1,1,3,3),-math.inf).triu(1))
    return model,args

def export_sources():
    from silicondevine import export_model
    out=Path(__file__).parents[1]/'demo/public/models'
    for family in ('deepseek_v3','glm45','minimax_m1'):
        for lightning,dense in ([(False,False),(True,False)] if family=='minimax_m1' else [(False,False),(False,True)]):
            m,args=build_source(family,lightning,dense)
            name=family+('_lightning' if lightning else '_dense' if dense else '')
            g=export_model(m,args,backend='execution',include_values=True,name=name+' · 源码执行组合')
            g['notes'].insert(0,'原始 Decoder.forward 的一次连贯执行；张量跨模块共享同一 ID。小配置、未训练参数，路由只记录本次输入经过的路径。')
            (out/(name+'.source.json')).write_text(json.dumps(g,ensure_ascii=False),encoding='utf-8')
            print(name,len(g['nodes']),'boundary',[(n['source']) for n in g['nodes'] if n['op']=='opaque'],flush=True)
if __name__=='__main__':export_sources()
