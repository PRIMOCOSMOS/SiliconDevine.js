"""Small, fully evaluated PyTorch mechanism graphs; never pretrained model values."""
import math
import torch
import torch.nn.functional as F

class Trace:
    def __init__(self, name, source):
        self.values={}; self.tensors=[]; self.nodes=[]; self.inputs=[]; self.source=source; self.name=name
        self.rng=torch.Generator().manual_seed(71)
    def random(self,*shape): return torch.randn(*shape,generator=self.rng)*.4
    def tensor(self,key,v,role='activation'):
        v=v.detach().cpu(); self.values[key]=v
        flat=v.flatten().tolist(); special={str(i):'-inf' for i,x in enumerate(flat) if isinstance(x,float) and x==-math.inf}
        t=dict(id=key,shape=list(v.shape),dtype=str(v.dtype).replace('torch.',''),role=role,data=dict(offset=0,values=[x if not isinstance(x,float) or math.isfinite(x) else None for x in flat]),semantic=key)
        if special:t['specialValues']=special
        self.tensors.append(t)
        if role=='input':self.inputs.append(key)
        return key
    def input(self,key,*shape):return self.tensor(key,self.random(*shape),'input')
    def op(self,key,op,inputs,v,parameters=None,**attrs):
        self.tensor(key,v);self.nodes.append(dict(id=key,name=key,op=op,inputs=inputs,outputs=[key],parameters=parameters or {},attrs=attrs,source=self.source));return key
    def linear(self,key,x,out,bias=False):
        w=self.tensor(key+'.W',self.random(out,self.values[x].shape[-1]),'parameter')
        b=self.tensor(key+'.b',self.random(out),'parameter') if bias else None
        return self.op(key,'linear',[x],F.linear(self.values[x],self.values[w],self.values[b] if b else None),{'weight':w,**({'bias':b} if b else {})})
    def unary(self,key,op,x,fn,**attrs):return self.op(key,op,[x],fn(self.values[x]),**attrs)
    def mm(self,key,a,b,**attrs):return self.op(key,'matmul',[a,b],self.values[a]@self.values[b],**attrs)
    def mul(self,key,a,b):return self.op(key,'multiply',[a,b],self.values[a]*self.values[b])
    def add(self,key,a,b):return self.op(key,'add',[a,b],self.values[a]+self.values[b])
    def norm(self,key,x):
        d=self.values[x].shape[-1];w=self.tensor(key+'.γ',torch.ones(d),'parameter')
        return self.op(key,'rmsnorm',[x],F.rms_norm(self.values[x],(d,),self.values[w],1e-6),{'weight':w},normalized_shape=[d],eps=1e-6)
    def transpose(self,key,x):return self.unary(key,'transpose',x,lambda v:v.transpose(-1,-2),dims=[-1,-2])
    def heads(self,key,x,h):
        v=self.values[x]; a=self.unary(key+'.split','reshape',x,lambda v:v.reshape(v.shape[0],h,-1))
        return self.unary(key,'permute',a,lambda v:v.permute(1,0,2),dims=[1,0,2])
    def rope(self,key,x,rotary_dim=None):
        v=self.values[x];d=rotary_dim or v.shape[-1]; pos=torch.arange(v.shape[-2]).float(); angles=pos[:,None]*10000**(-torch.arange(0,d,2).float()/d)
        cs=self.tensor(key+'.cos',angles.cos(),'constant');sn=self.tensor(key+'.sin',angles.sin(),'constant')
        a,b=v[...,:d:2],v[...,1:d:2];y=torch.stack([a*self.values[cs]-b*self.values[sn],a*self.values[sn]+b*self.values[cs]],-1).flatten(-2)
        if d<v.shape[-1]:y=torch.cat([y,v[...,d:]],-1)
        return self.op(key,'rope_pair',[x,cs,sn],y,rotaryDim=d)
    def probabilities(self,key,s,scale=1,causal=True):
        factor=self.tensor(key+'.scale',torch.tensor(scale),'constant');s=self.mul(key+'.scale_scores',s,factor)
        if causal:s=self.unary(key+'.mask','attention_mask',s,lambda v:v.masked_fill(torch.ones(v.shape[-2:],dtype=torch.bool).triu(1),-math.inf),causal=True)
        return self.unary(key,'softmax',s,lambda v:v.softmax(-1),dim=-1,attentionRole='probability')
    def finish(self,out):
        for node in self.nodes:
            if node['attrs'].get('attentionRole')=='context':
                node['attrs']['attention']={'probability':node['inputs'][0],'value':node['inputs'][1],'context':node['outputs'][0]}
        return dict(format='silicondevine',version=1,name=self.name,tensors=self.tensors,nodes=self.nodes,inputs=self.inputs,outputs=[out],producer={'backend':'pytorch-mechanism','version':torch.__version__},notes=['小规模 PyTorch 实算演示；固定种子未训练参数。遵循对应算式与数据依赖，不是模型权重切片或完整模型推理。'],provenance={'basis':'audited equations evaluated with PyTorch','source':self.source,'seed':71})

def swiglu(g,x,prefix='专家',middle=6):
    gate=g.linear(prefix+'.门控',x,middle);up=g.linear(prefix+'.内容',x,middle)
    gate=g.unary(prefix+'.SiLU','silu',gate,F.silu)
    return g.linear(prefix+'.输出',g.mul(prefix+'.门控乘积',gate,up),g.values[x].shape[-1])

def make_mechanism(kind,family,config=None):
    c=config or {}; source={'deepseek_v3':'DeepSeek-V3 / inference/model.py · MLA.forward / Gate.forward / Expert.forward','glm45':'Transformers / modeling_glm4_moe.py · Attention / MoE','minimax_m1':'MiniMax-M1 / modeling_minimax_m1.py · LightningAttention / SparseMoeBlock'}.get(family,'PyTorch functional')
    names={'attention':'MLA · 压缩空间实算' if family=='deepseek_v3' else 'GQA · Token 注意力实算','lightning':'Lightning · 状态写入与读取','moe':'MoE · 路由与独立专家实算','router':'路由 · 分组选择与归一化','expert':'SwiGLU · 门控实算','projection':'线性投影 · 权重逐项累加','norm':'RMSNorm · 均方根缩放','rope':'RoPE · 成对旋转','merge':'残差 · 原信号与更新汇合','gate':'门控 · 逐元素调制','embedding':'Token · 查表'}
    g=Trace(names.get(kind,kind),source); x=g.input('Token X',3,4); out=x
    if kind in ('expert','dense_ffn','shared_expert'):out=swiglu(g,x)
    elif kind=='projection':out=g.linear('投影输出',x,6)
    elif kind=='norm':out=g.norm('RMSNorm',x)
    elif kind=='rope':out=g.rope('成对旋转',x)
    elif kind=='embedding':
        ids=g.tensor('Token ID',torch.tensor([1,3,2]),'input');w=g.tensor('Embedding.W',g.random(6,4),'parameter');out=g.op('查表结果','embedding',[ids],F.embedding(g.values[ids],g.values[w]),{'weight':w})
    elif kind=='merge':out=g.add('残差输出',x,g.linear('子层更新',x,4))
    elif kind=='gate':out=g.mul('门控输出',x,g.unary('Sigmoid 门','sigmoid',g.linear('门投影',x,4),torch.sigmoid))
    elif kind=='attention' and family=='deepseek_v3':
        q=g.linear('Query 低秩',x,3);q=g.norm('Query RMSNorm',q);q=g.linear('Query 展开',q,4)
        qn=g.unary('Query 内容','slice',q,lambda v:v[:,:2],dim=1,start=0,end=2,step=1)
        qr=g.unary('Query 位置','slice',q,lambda v:v[:,2:],dim=1,start=2,end=4,step=1);qr=g.rope('Query RoPE',qr)
        kv=g.linear('KV 联合压缩',x,6)
        latent=g.unary('KV 潜变量','slice',kv,lambda v:v[:,:4],dim=1,start=0,end=4,step=1);latent=g.norm('KV RMSNorm',latent)
        kr=g.unary('Key 位置','slice',kv,lambda v:v[:,4:],dim=1,start=4,end=6,step=1);kr=g.rope('Key RoPE',kr)
        wk=g.tensor('W_K · 吸收权重',g.random(2,4),'parameter');wv=g.tensor('W_V · 解压权重',g.random(4,2),'parameter')
        qa=g.mm('吸收后的 Query',qn,wk)
        a=g.mm('内容分数',qa,g.transpose('潜变量转置',latent),attentionRole='score');b=g.mm('位置分数',qr,g.transpose('位置转置',kr))
        p=g.probabilities('注意力概率',g.add('联合分数',a,b),.5)
        o=g.mm('注意力聚合潜变量',p,latent,attentionRole='context');o=g.mm('Value 解压',o,wv);out=g.linear('输出投影',o,4)
    elif kind=='attention':
        q=g.heads('Query 头',g.linear('Q 投影',x,8,c.get('attention_bias',False)),2);k=g.heads('Key 组',g.linear('K 投影',x,4,c.get('attention_bias',False)),1);v=g.heads('Value 组',g.linear('V 投影',x,4,c.get('attention_bias',False)),1)
        if family=='glm45' and c.get('use_qk_norm'):q=g.norm('Q 逐头 RMSNorm',q);k=g.norm('K 逐头 RMSNorm',k)
        q=g.rope('Q RoPE',q,2);k=g.rope('K RoPE',k,2)
        s=g.mm('两头共享 Key · 分数',q,g.transpose('Key 转置',k),attentionRole='score');p=g.probabilities('注意力概率',s,4**-.5)
        o=g.mm('两头共享 Value · 加权输出',p,v,attentionRole='context');o=g.unary('拼接前换轴','permute',o,lambda v:v.permute(1,0,2),dims=[1,0,2]);o=g.unary('拼接多头','reshape',o,lambda v:v.reshape(3,8));out=g.linear('输出投影',o,4)
    elif kind=='lightning':
        qkv=g.unary('QKV SiLU','silu',g.linear('联合 QKV 投影',x,6),F.silu)
        q=g.unary('Query','slice',qkv,lambda v:v[:,:2],dim=1,start=0,end=2,step=1);k=g.unary('Key','slice',qkv,lambda v:v[:,2:4],dim=1,start=2,end=4,step=1);v=g.unary('Value','slice',qkv,lambda v:v[:,4:6],dim=1,start=4,end=6,step=1)
        state=g.tensor('初始状态 S₀',torch.zeros(2,2),'buffer');decay=g.tensor('每步衰减 λ',torch.tensor(math.exp(-.5)),'constant');reads=[]
        for t in range(3):
            kt=g.unary(f'k{t}','slice',k,lambda v,t=t:v[t:t+1],dim=0,start=t,end=t+1,step=1);vt=g.unary(f'v{t}','slice',v,lambda v,t=t:v[t:t+1],dim=0,start=t,end=t+1,step=1)
            write=g.mm(f'外积写入 {t}',g.transpose(f'k{t} 转置',kt),vt)
            # Atomic recurrent update keeps the state plane beside the write/read plane.
            state=g.op(f'状态 S{t+1}','state_update',[state,write,decay],g.values[state]*g.values[decay]+g.values[write])
            qt=g.unary(f'q{t}','slice',q,lambda v,t=t:v[t:t+1],dim=0,start=t,end=t+1,step=1);reads.append(g.mm(f'Query 读取 {t}',qt,state))
        joined=g.op('Token 输出','concat',reads,torch.cat([g.values[r] for r in reads],0),dim=0);norm=g.norm('输出 RMSNorm',joined)
        gate=g.unary('输出门 Sigmoid','sigmoid',g.linear('门投影',x,2),torch.sigmoid);out=g.linear('输出投影',g.mul('门控',norm,gate),4)
    elif kind in ('moe','router'):
        # 4 experts in 2 groups retains group-limited routing, correction and independent weights.
        logits=g.linear('路由 logits',x,4);mini=family=='minimax_m1';s=g.unary('原始路由分数','softmax' if mini else 'sigmoid',logits,lambda v:v.softmax(-1) if mini else v.sigmoid(),dim=-1)
        bias=g.tensor('选择校正',torch.tensor([.02,-.03,.08,0]),'buffer');scores=g.values[s];adjusted=scores if mini else scores+g.values[bias]
        groups=2 if not mini else 1;selected_groups=1 if groups==2 else 1
        allowed=torch.ones_like(adjusted,dtype=torch.bool)
        if groups==2:
            grouped=adjusted.reshape(3,2,2);gs=grouped.topk(2,dim=-1).values.sum(-1);best=gs.topk(1,-1).indices;allowed=torch.zeros_like(gs,dtype=torch.bool).scatter_(1,best,True).unsqueeze(-1).expand(-1,-1,2).reshape(3,4)
        ids=adjusted.masked_fill(~allowed,-math.inf).topk(2,-1).indices
        idx=g.op('分组 Top-2 索引','route_select',[s,bias] if not mini else [s],ids,groups=groups,selectedGroups=selected_groups,topK=2,scoreFunction='softmax' if mini else 'sigmoid')
        weights=torch.gather(scores,1,ids);weights=weights/weights.sum(-1,keepdim=True)*float(c.get('route_scale',c.get('routed_scaling_factor',1)))
        w=g.op('选中权重 · 归一化','route_weights',[s,idx],weights)
        if kind=='router':out=w
        else:
            ys=[swiglu(g,x,f'专家 {i}',3) for i in range(4)]
            combined=sum(g.values[ys[e]]*(weights*(ids==e)).sum(-1,keepdim=True) for e in range(4))
            out=g.op('稀疏加权汇聚','expert_combine',[idx,w,*ys],combined)
            if not mini and c.get('n_shared_experts',0):out=g.add('加上共享专家',out,swiglu(g,x,'共享专家',3))
    else:raise ValueError(kind)
    result=g.finish(out)
    ids=[n['id'] for n in g.nodes]
    cuts=[]
    if kind=='attention' and family=='deepseek_v3':cuts=[('Query 投影与位置',0,6),('KV 压缩与位置',6,11),('双分数与因果概率',11,20),('聚合与 Value 解压',20,len(ids))]
    elif kind=='attention':
        boundary=ids.index('两头共享 Key · 分数');end=ids.index('两头共享 Value · 加权输出')
        cuts=[('Q / K / V 投影与位置',0,boundary),('因果注意力分配',boundary,end),('Value 汇聚与输出',end,len(ids))]
    elif kind=='lightning':
        cuts=[('QKV 与激活',0,5)]
        for i in range(3):cuts.append((f'Token {i+1} · 写入与读取',5+i*7,12+i*7))
        cuts.append(('归一化、门控与输出',26,len(ids)))
    elif kind=='moe':cuts=[('Token 路由与权重',0,4),('独立 SwiGLU 专家',4,24),('加权汇聚与共享分支',24,len(ids))]
    if cuts:result['stages']=[{'name':name,'nodes':ids[start:end]} for name,start,end in cuts if start<end]

    if kind=='attention' and family=='deepseek_v3':result['notes'].append('演示单个注意力头的吸收式 MLA，保留内容/位置双分数、因果掩码和压缩空间聚合；省略多头复制与长上下文 mscale。')
    if kind=='lightning':result['notes'].append('一个头、三个连续 Token；λ=exp(-0.5)。展示 decode 递推等价式，不执行 CUDA 分块 kernel。')
    if kind in ('moe','router'):result['notes'].append('4 个独立专家，Top-2；分组数缩小为 2（MiniMax 不分组）。全尺寸配置与参数量在结构视图保留。')
    return result

def attach_mechanisms(graph):
    a=graph['architecture'];family=a['family'];c=a['config'];keys=['attention','expert','router','moe','projection','norm','rope','merge','gate','embedding']
    if family=='minimax_m1':keys+=['lightning']
    a['mechanisms']={k:make_mechanism(k,family,c) for k in keys}
    for demo in a['mechanisms'].values():demo['provenance']['upstream']=graph.get('provenance',{})
    for key,scope in a['scopes'].items():
        scope['mechanismRef']=key if key in keys else 'expert' if key in ('dense_ffn','shared_expert') else None
        for n in scope['nodes']:
            kind=n['attrs']['kind'];ref=n['attrs'].get('scopeRef')
            n['attrs']['mechanismRef']=ref if ref in keys else 'expert' if ref in ('dense_ffn','shared_expert') else {'ffn':'expert','experts':'moe','recurrent':'lightning'}.get(kind,kind) if kind in keys+['ffn','experts','recurrent'] else 'projection'
            if key in ('expert','dense_ffn','shared_expert') and n['id']=='product':n['attrs']['mechanismRef']='expert'
            if key=='attention' and n['id'] in ('absorbed','scores','mix'):n['attrs']['mechanismStage']=2 if family=='deepseek_v3' or n['id']=='mix' else 1
            if key=='lightning' and n['id']=='recurrence':n['attrs']['mechanismStage']=1
    return graph
