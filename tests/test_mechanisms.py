import ast, math, types, unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from silicondevine.mechanisms import make_mechanism
from silicondevine.conditional import DynamicConv2d, ConditionalAttention, mix_kernels, batch_conv2d
from silicondevine import export_model
ROOT=Path(__file__).resolve().parents[1]/'examples/upstream/large_models'

def method(file,cls,name,extra=None):
    tree=ast.parse(file.read_text('utf-8'));node=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls);fn=next(f for f in node.body if isinstance(f,ast.FunctionDef) and f.name==name);fn.decorator_list=[]
    functions=[fn,*[f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name in (extra or [])]]
    env={'torch':torch,'F':F,'math':math,'attn_impl':'absorb','linear':F.linear}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*functions],type_ignores=[])),str(file),'exec'),env)
    return env

def values(g):return {t['id']:torch.tensor(t['data']['values']).reshape(t['shape']) for t in g['tensors'] if not t.get('specialValues')}
class Mechanisms(unittest.TestCase):
    def test_mla_against_official_forward(self):
        g=make_mechanism('attention','deepseek_v3');v=values(g)
        linear=lambda key:lambda x:F.linear(x,v[key+'.W'])
        norm=lambda x:F.rms_norm(x,(x.shape[-1],),eps=1e-6)
        obj=types.SimpleNamespace(q_lora_rank=3,n_local_heads=1,qk_head_dim=4,qk_nope_head_dim=2,qk_rope_head_dim=2,kv_lora_rank=4,v_head_dim=2,softmax_scale=.5,wq_a=linear('Query 低秩'),q_norm=norm,wq_b=linear('Query 展开'),wkv_a=linear('KV 联合压缩'),kv_norm=norm,wkv_b=types.SimpleNamespace(weight=torch.cat([v['W_K · 吸收权重'],v['W_V · 解压权重'].T],0),scale=None),kv_cache=torch.zeros(1,3,4),pe_cache=torch.zeros(1,3,2),wo=linear('输出投影'))
        env=method(ROOT/'deepseek_v3/model.py','MLA','forward',['apply_rotary_emb'])
        freq=torch.polar(torch.ones(3,1),torch.arange(3).float()[:,None]);mask=torch.full((3,3),-math.inf).triu(1)
        y=env['forward'](obj,v['Token X'][None],0,freq,mask)
        torch.testing.assert_close(y[0],v[g['outputs'][0]],atol=2e-6,rtol=1e-5)
    def test_lightning_against_official_inference(self):
        g=make_mechanism('lightning','minimax_m1');v=values(g);linear=lambda key:lambda x:F.linear(x,v[key+'.W'])
        obj=types.SimpleNamespace(act=F.silu,qkv_proj=linear('联合 QKV 投影'),num_heads=1,head_dim=2,offset=0,norm=lambda x:F.rms_norm(x,(2,),eps=1e-6),output_gate=linear('门投影'),out_proj=linear('输出投影'))
        env=method(ROOT/'minimax_m1/modeling_minimax_m1.py','MiniMaxM1LightningAttention','inference');env['rearrange']=lambda x,_:x.transpose(1,2).flatten(2)
        y,_,state=env['inference'](obj,v['Token X'][None],past_key_value=torch.zeros(1,1,2,2),slope_rate=torch.tensor([[[.5]]]))
        torch.testing.assert_close(y[0],v[g['outputs'][0]])
        torch.testing.assert_close(state[0,0],v['状态 S3'])
    def test_deepseek_gate_against_official(self):
        g=make_mechanism('router','deepseek_v3',{'route_scale':2.5});v=values(g)
        obj=types.SimpleNamespace(weight=v['路由 logits.W'],score_func='sigmoid',bias=v['选择校正'],n_groups=2,topk_groups=1,topk=2,route_scale=2.5)
        env=method(ROOT/'deepseek_v3/model.py','Gate','forward');weights,idx=env['forward'](obj,v['Token X'])
        torch.testing.assert_close(idx,v['分组 Top-2 索引'].long());torch.testing.assert_close(weights,v['选中权重 · 归一化'])
    def test_glm_gate_against_official(self):
        path=ROOT/'glm45/modeling_glm4_moe.py';tree=ast.parse(path.read_text());cls=next(c.name for c in tree.body if isinstance(c,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='get_topk_indices' for f in c.body))
        g=make_mechanism('router','glm45',{'routed_scaling_factor':2.5});v=values(g)
        obj=types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=4),weight=v['路由 logits.W'],n_routed_experts=4,n_group=2,topk_group=1,top_k=2,norm_topk_prob=True,routed_scaling_factor=2.5,e_score_correction_bias=v['选择校正'])
        obj.get_topk_indices=types.MethodType(method(path,cls,'get_topk_indices')['get_topk_indices'],obj)
        idx,weights=method(path,cls,'forward')['forward'](obj,v['Token X'])
        actual=torch.zeros(3,4).scatter_(1,idx,weights);expected=torch.zeros(3,4).scatter_(1,v['分组 Top-2 索引'].long(),v['选中权重 · 归一化']);torch.testing.assert_close(actual,expected)
    def test_glm_attention_against_official(self):
        from transformers import Glm4MoeConfig
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeAttention
        g=make_mechanism('attention','glm45',{'attention_bias':True,'use_qk_norm':True});v=values(g)
        config=Glm4MoeConfig(hidden_size=4,num_attention_heads=2,num_key_value_heads=1,head_dim=4,attention_bias=True,use_qk_norm=True,rms_norm_eps=1e-6)
        config._attn_implementation='eager';m=Glm4MoeAttention(config,0).eval()
        with torch.no_grad():
            for attr,key in [('q_proj','Q 投影'),('k_proj','K 投影'),('v_proj','V 投影'),('o_proj','输出投影')]:
                layer=getattr(m,attr);layer.weight.copy_(v[key+'.W'])
                if layer.bias is not None:layer.bias.copy_(v[key+'.b'])
            angle=torch.arange(3).float()[:,None].expand(3,2)[None]
            out,_=m(v['Token X'][None],(angle.cos(),angle.sin()),torch.full((1,1,3,3),-math.inf).triu(1))
        torch.testing.assert_close(out[0],v[g['outputs'][0]],atol=2e-6,rtol=1e-5)

    def test_dynamic_groups_and_stride(self):
        torch.manual_seed(9);x=torch.randn(2,4,7,7);bank=torch.randn(3,6,2,3,3);a=torch.randn(2,3).softmax(-1);w=mix_kernels(a,bank)
        y=batch_conv2d(x,w,2,1,1,2)
        expected=torch.cat([sum(a[b,e]*F.conv2d(x[b:b+1],bank[e],stride=2,padding=1,groups=2) for e in range(3)) for b in range(2)])
        torch.testing.assert_close(y,expected)
        graph=export_model(DynamicConv2d(4,6,groups=2,stride=2), (x,), include_values=True)
        self.assertTrue(any(n['op']=='dynamic_conv2d' for n in graph['nodes']))
    def test_conditional_equals_sdpa(self):
        torch.manual_seed(7);m=ConditionalAttention(4,6,2);x=torch.randn(1,3,4);c=torch.randn(1,5,6);mask=torch.zeros(3,5);mask[:,4]=-math.inf
        q=m.q(x).reshape(1,3,2,2).transpose(1,2);k=m.k(c).reshape(1,5,2,2).transpose(1,2);v=m.v(c).reshape(1,5,2,2).transpose(1,2)
        y=m.out(F.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2).reshape(1,3,4));torch.testing.assert_close(m(x,c,mask),y)
if __name__=='__main__':unittest.main()
