"""Bounded, metadata-only architecture graphs. No weight loading or remote execution."""
import copy
import math


def export_architecture(model, *, family=None):
    """Inspect a supported local PyTorch model's config without running forward.

    A model constructed under torch.device('meta') is supported. Config adapters
    describe the audited implementation; modified forward methods require a trace.
    """
    config = getattr(model, 'config', None)
    if config is None:
        raise ValueError('Model has no config; use architecture_from_config or export_model')
    c = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
    inferred = {'glm4_moe':'glm45', 'minimax_m1':'minimax_m1'}.get(c.get('model_type'))
    family = family or inferred
    if not family:
        raise ValueError('No audited adapter for this model_type; export_model captures a runnable submodule')
    result = architecture_from_config(c, family, provenance={'class':type(model).__module__+'.'+type(model).__qualname__,
                                                            'basis':'local config adapter; not a forward trace'})
    # Enumerating metadata never copies storage. Useful cross-check against the instantiated model.
    if hasattr(model, 'parameters'):
        count = sum(p.numel() for p in model.parameters())
        result['provenance']['instantiatedParameterCount'] = str(count)
        result['provenance']['countMatchesBackbone'] = str(count) == result['architecture']['parameterCount']
    return result


class Scope:
    def __init__(self, name, width, source=''):
        self.name, self.source = name, source
        self.tensors, self.nodes = [], []
        self.input = self.tensor('input', ['B', 'T', width], 'input')
        self.last = self.input

    def tensor(self, name, shape, role='activation'):
        labels={'input':'输入','compressed_KV_cache':'压缩 KV 缓存','position_cache':'位置分量缓存','KV_state':'递归 KV 状态','K_cache':'Key 缓存','V_cache':'Value 缓存','selection_bias':'路由选择校正'}
        self.tensors.append(dict(id=name, shape=shape, dtype='logical', role=role, representation='aggregate',semantic=labels.get(name,name)))
        return name

    def add(self, key, name, shape, inputs=None, weights=None, kind='projection', **attrs):
        out = self.tensor(key + ':out', shape)
        parameters = {}
        for label, dims in (weights or {}).items():
            parameters[label] = self.tensor(key + ':' + label, dims, 'parameter')
        self.nodes.append(dict(id=key, name=name, op='structure', inputs=inputs if inputs is not None else [self.last],
                               outputs=[out], parameters=parameters, attrs=dict(kind=kind, **attrs), source=self.source))
        self.last = out
        return out

    def graph(self):
        return dict(name=self.name, description='按源码组合的结构模板；同构实例拥有各自权重。', source=self.source,
                    tensors=self.tensors, nodes=self.nodes, inputs=[self.input], outputs=[self.last])


def architecture_from_config(config, family, *, provenance=None):
    """Audited inference-backbone adapters: deepseek_v3, glm45, minimax_m1.

    Config is supplied locally. Unsupported variants fail explicitly. These are
    logical graphs, not numerical traces, training graphs, or runtime memory estimates.
    """
    c = copy.deepcopy(config)
    if family not in ('deepseek_v3', 'glm45', 'minimax_m1'):
        raise ValueError('Supported families: deepseek_v3, glm45, minimax_m1')
    deep, mini = family == 'deepseek_v3', family == 'minimax_m1'
    D = c['dim'] if deep else c['hidden_size']
    H = c['n_heads'] if deep else c['num_attention_heads']
    L = c['n_layers'] if deep else c['num_hidden_layers']
    E = c['num_local_experts'] if mini else c['n_routed_experts']
    K = c['n_activated_experts'] if deep else c['num_experts_per_tok']
    M = c['moe_inter_dim'] if deep else c['intermediate_size'] if mini else c['moe_intermediate_size']
    F = c['inter_dim'] if deep else c['intermediate_size']
    dense = c['n_dense_layers'] if deep else 0 if mini else c['first_k_dense_replace']
    V = c['vocab_size']
    if not all(isinstance(n, int) and 0 < n <= 10000000 for n in (D,H,L,E,K,M,F,V)) or K > E or not 0 <= dense <= L:
        raise ValueError('Invalid architecture dimensions')
    if c.get('tie_word_embeddings', False):
        raise ValueError('This adapter expects an untied output head')
    if c.get('hidden_act','silu') != 'silu' or (deep and c.get('score_func','sigmoid') != 'sigmoid') or (family=='glm45' and not c.get('norm_topk_prob', True)):
        raise ValueError('Activation or routing variant has no audited adapter')
    if mini and (c.get('shared_intermediate_size', 0) or not c.get('postnorm', False)):
        raise ValueError('MiniMax adapter currently supports M1 postnorm without shared MLP')
    if deep and not c.get('q_lora_rank', 0):
        raise ValueError('This MLA adapter requires the V3 low-rank query path')
    source = {'deepseek_v3':'DeepSeek-V3 / inference/model.py', 'glm45':'Transformers / modeling_glm4_moe.py',
              'minimax_m1':'MiniMax-M1 / modeling_minimax_m1.py'}[family]
    scopes = {}
    shape = ['B','T',D]
    def scope(key, name):
        s = Scope(name, D, source); scopes[key] = s; return s
    def mlp(key, middle):
        s = scope(key, 'SwiGLU · 门控前馈')
        gate = s.add('gate','门控投影', ['B','T',middle], [s.input], {'W':[middle,D]})
        up = s.add('up','内容投影', ['B','T',middle], [s.input], {'W':[middle,D]})
        s.add('product','SiLU 门控 × 内容', ['B','T',middle], [gate,up], kind='gate', formula=r'h=\operatorname{SiLU}(xW_g^T)\odot xW_u^T')
        s.add('down','输出投影', shape, weights={'W':[D,middle]})
    mlp('expert',M)
    if dense: mlp('dense_ffn',F)
    shared = 0 if mini else c.get('n_shared_experts', 0)
    if shared: mlp('shared_expert', M*shared)
    s = scope('router','Token → 专家路由')
    logits = s.add('logits','路由投影', ['B','T',E], weights={'W':[E,D]})
    correction = []
    if not mini:
        correction = [s.tensor('selection_bias',[E], 'parameter' if deep and D==7168 else 'buffer')]
    s.add('select',f'{"Softmax" if mini else "Sigmoid"} · Top-{K}', ['B','T',K], [logits,*correction], kind='router',
          experts=E, topK=K, description=f'每个 Token 选择 {K}/{E} 个专家。' + ('先选专家组；校正项只用于选择，混合权重来自原始分数。' if not mini else 'Softmax 后取 Top-k，再归一化。'),
          formula=r'w=\operatorname{Normalize}(\operatorname{TopK}(s))',
          groups=c.get('n_expert_groups',c.get('n_group',1)), selectedGroups=c.get('n_limited_groups',c.get('topk_group',1)),
          scale=c.get('route_scale',c.get('routed_scaling_factor',1)))
    s = scope('moe','MoE · 稀疏专家混合')
    route = s.add('router',f'路由 · 每 Token {K}/{E}', ['B','T',K], [s.input], kind='router', scopeRef='router', topK=K, experts=E)
    experts = s.add('experts',f'{E} 个独立专家 · 激活 {K}', ['assignments',D],[s.input,route],kind='experts', scopeRef='expert', repeat=E, topK=K,
                    description=f'按路由索引分派 Token，各专家处理不同长度的序列。assignments = B × T × {K}；图示汇总这些稀疏分派结果。模板代表独立参数的专家组。')
    parts = [experts,route]
    if shared:
        parts.append(s.add('shared',f'共享专家 · 宽度 {M*shared}',shape,[s.input],kind='ffn',scopeRef='shared_expert'))
    s.add('combine','按路由权重汇聚 + 共享分支' if shared else '按路由权重汇聚',shape,parts,kind='merge',
          formula=r'y=\sum_{e\in\mathrm{TopK}}w_e f_e(x)' + (r'+f_s(x)' if shared else ''))
    if deep:
        s = scope('attention','MLA · 压缩潜变量注意力')
        R,Q,N,P,A = c['kv_lora_rank'],c['q_lora_rank'],c['qk_nope_head_dim'],c['qk_rope_head_dim'],c['v_head_dim']
        q = s.add('q_down','Query 低秩投影', ['B','T',Q],[s.input], {'W':[Q,D]})
        q = s.add('q_norm','Query RMSNorm',['B','T',Q],[q],{'gamma':[Q]},kind='norm')
        q = s.add('q_up','Query 多头展开',['B','T',H,N+P],[q],{'W':[H*(N+P),Q]})
        q = s.add('q_rope','Query · 位置子空间旋转',['B','T',H,N+P],[q],kind='rope',description=f'每头前 {N} 维保持原值；后 {P} 维应用 RoPE。')
        kv = s.add('kv_down','KV 压缩 + 位置分量',['B','T',R+P],[s.input],{'W':[R+P,D]})
        pe = s.add('k_rope','Key · 位置子空间旋转',['B','T',1,P],[kv],kind='rope',description=f'切分最后 {P} 维，对独立 Key 位置分量应用 RoPE。')
        kv = s.add('kv_norm','压缩 KV · RMSNorm',['B','T',R],[kv],{'gamma':[R]},kind='norm',description=f'切分前 {R} 维并归一化。')
        cache = s.tensor('compressed_KV_cache',['B','S',R],'buffer')
        pecache = s.tensor('position_cache',['B','S',P],'buffer')
        s.add('absorbed','吸收式 MLA · 因果注意力',['B','T',H,A],[q,kv,pe,cache,pecache],{'W_KV':[H*(N+A),R]},kind='attention',
              description='Query 的非位置分量乘 W_K；在压缩 KV 与 RoPE 缓存上计算分数。Softmax 聚合潜变量后乘 W_V。使用源码 absorb 路径，不物化完整多头 KV 缓存。',
              formula=r'p=\operatorname{softmax}(s[(q_nW_K)c^T+q_rk_r^T]+M),\quad o=(pc)W_V^T', scale=(N+P)**-.5,
              scaleNote='s 为源码 softmax_scale；长上下文时还应用 mscale 修正。')
        s.add('out','多头输出投影',shape,weights={'W':[D,H*A]})
    else:
        h, hd = c['num_key_value_heads'],c['head_dim']
        s = scope('attention','GQA · 分组查询注意力')
        parts=[]
        for key,heads in [('Q',H),('K',h),('V',h)]:
            weights={'W':[heads*hd,D]}
            if c.get('attention_bias',False): weights['bias']=[heads*hd]
            p=s.add(key,key+' 投影',['B','T',heads,hd],[s.input],weights)
            if not mini and key!='V' and c.get('use_qk_norm'):
                p=s.add(key+'_norm',key+' 逐头 RMSNorm',['B','T',heads,hd],[p],{'gamma':[hd]},kind='norm')
            parts.append(p)
        rotated=[]
        for key,p,heads in [('Q',parts[0],H),('K',parts[1],h)]:
            rotated.append(s.add('rope_'+key,key+' · 部分旋转位置编码',['B','T',heads,hd],[p],kind='rope',
                  rotaryDim=c.get('rotary_dim',int(hd*c.get('partial_rotary_factor',1))),description='旋转位置子空间；Value 保持不变。'))
        cache=s.tensor('K_cache',['B',h,'S',hd],'buffer')
        vcache=s.tensor('V_cache',['B',h,'S',hd],'buffer')
        s.add('scores','分组广播 · 因果 Softmax',['B',H,'T','S'],[*rotated,cache],kind='attention',
              formula=r'P=\operatorname{softmax}(QK^T/\sqrt{d_h}+M)',description=f'{H} 个 Query 头共享 {h} 组 Key/Value；图中按组概括，不复制缓存。')
        s.add('mix','注意力作用于 Value',['B','T',H,hd],[s.last,parts[2],vcache],kind='attention',formula=r'O=PV')
        s.add('out','多头输出投影',shape,weights={'W':[D,H*hd]})
        if mini:
            s=scope('lightning','Lightning Attention · 递归状态')
            qkv=s.add('qkv','融合 QKV 投影 + SiLU',['B','T',H,3*hd],weights={'W':[3*H*hd,D]},description='最后一维按 Q、K、V 各 head_dim 切分，形成逐头向量。')
            state=s.tensor('KV_state',['B',H,hd,hd],'buffer')
            s.add('recurrence','衰减状态更新 → Query 读取',['B','T',H,hd],[qkv,state],kind='recurrent',
                  formula=r'S_t=\lambda_hS_{t-1}+k_t^Tv_t,\quad o_t=q_tS_t',
                  description='展示 decode 递推。prefill 使用源码的分块因果等价计算；这里没有 Softmax。状态是缓存，不是训练参数。')
            norm=s.add('norm','多头输出 RMSNorm',['B','T',H*hd],weights={'gamma':[H*hd]},kind='norm')
            gate=s.add('gate','输出门投影 + Sigmoid',['B','T',H*hd],[s.input],{'W':[H*hd,D]},kind='gate')
            s.add('multiply','门控输出',['B','T',H*hd],[norm,gate],kind='gate')
            s.add('out','输出投影',shape,weights={'W':[D,H*hd]})
    def block(key, attention, ffn):
        s=scope(key, ('Lightning' if attention=='lightning' else 'MLA' if deep else 'GQA')+' + '+('MoE' if ffn=='moe' else 'SwiGLU'))
        n=s.add('norm1','注意力 RMSNorm',shape,weights={'gamma':[D]},kind='norm')
        a=s.add('attention',scopes[attention].name,shape,kind='attention',scopeRef=attention)
        residual=n if mini else s.input
        s.add('add1','缩放残差汇合' if mini else '注意力残差汇合',shape,[residual,a],kind='merge',residualInput=residual,
              formula=r'y=\alpha\,\mathrm{RMSNorm}(x)+\beta\,A(\mathrm{RMSNorm}(x))' if mini else r'y=x+A(\mathrm{RMSNorm}(x))',
              alpha=c.get('layernorm_linear_attention_alpha' if attention=='lightning' else 'layernorm_full_attention_alpha',1),beta=c.get('layernorm_linear_attention_beta' if attention=='lightning' else 'layernorm_full_attention_beta',1))
        residual=s.last
        n=s.add('norm2','前馈 RMSNorm',shape,weights={'gamma':[D]},kind='norm')
        f=s.add('ffn',scopes[ffn].name,shape,kind='experts' if ffn=='moe' else 'ffn',scopeRef=ffn,experts=E if ffn=='moe' else 0,topK=K)
        residual=n if mini else residual
        s.add('add2','前馈残差汇合',shape,[residual,f],kind='merge',residualInput=residual,
              alpha=c.get('layernorm_mlp_alpha',1),beta=c.get('layernorm_mlp_beta',1))
    if dense: block('dense_block','attention','dense_ffn')
    block('moe_block','attention','moe')
    if mini: block('lightning_block','lightning','moe')
    root=scope('root',{'deepseek_v3':'DeepSeek-V3','glm45':'GLM-4.5','minimax_m1':'MiniMax-M1-80k'}[family])
    root.tensors[0]['shape']=['B','T'];root.tensors[0]['dtype']='int64'
    root.add('embedding','Token 嵌入',shape,weights={'W':[V,D]},kind='embedding')
    schedule=[]
    if mini:
        schedule=c['attn_type_list']
        if len(schedule)!=L or any(t not in (0,1) for t in schedule): raise ValueError('Invalid attention schedule')
        period=next(p for p in range(1,L+1) if L%p==0 and schedule==schedule[:p]*(L//p))
        cycle=scope('hybrid_cycle',f'混合周期 · {period} 层')
        # Run-length encode the exact period; no reordering of attention types.
        i=0
        while i<period:
            end=i+1
            while end<period and schedule[end]==schedule[i]: end+=1
            ref='lightning_block' if schedule[i]==0 else 'moe_block'
            cycle.add('layers'+str(i),f'{"Lightning" if schedule[i]==0 else "全注意力"} × {end-i}',shape,kind='repeat',scopeRef=ref,repeat=end-i,layerRange=[i,end-1])
            i=end
        root.add('cycles',f'混合周期 × {L//period} · 共 {L} 层',shape,kind='repeat',scopeRef='hybrid_cycle',repeat=L//period)
    else:
        if dense: root.add('dense',f'稠密层 0–{dense-1} · ×{dense}',shape,kind='repeat',scopeRef='dense_block',repeat=dense)
        if L>dense: root.add('sparse',f'MoE 层 {dense}–{L-1} · ×{L-dense}',shape,kind='repeat',scopeRef='moe_block',repeat=L-dense)
    root.add('norm','最终 RMSNorm',shape,weights={'gamma':[D]},kind='norm')
    root.add('head','最后 Token → 词表' if deep else '词表输出投影',['B',V] if deep else ['B','T',V],weights={'W':[V,D]},kind='projection',
             description='官方推理入口先取归一化结果的最后一个 Token，再计算词表 logits。' if deep else '默认返回序列位置的词表 logits；运行时可选择保留的 Token。')
    graphs={key:s.graph() for key,s in scopes.items()}
    def count(key):
        g=graphs[key]
        local=sum(math.prod(t['shape']) for t in g['tensors'] if t['role']=='parameter')
        return local+sum(count(n['attrs']['scopeRef'])*n['attrs'].get('repeat',1) for n in g['nodes'] if 'scopeRef' in n['attrs'])
    notes=['结构模式：未加载权重、未运行推理；动画说明数据通路，不表示真实 Token 路由或数值。',
           '参数统计为推理主干逻辑权重；不含 MTP、量化尺度、优化器状态与缓存。模板复用不表示参数共享。']
    return dict(format='silicondevine',version=1,**graphs['root'], notes=notes,producer={'backend':'audited-config'},provenance=provenance or {},
                architecture=dict(mode='structure',entry='root',scopes=graphs,parameterCount=str(count('root')),layers=L,experts=E,topK=K,
                                  layerSchedule=schedule,family=family,config=c))
