import { topologicalNodes, type Model, type Operation } from './model.js';

/** Contract a verified subgraph. Boundaries always retain the captured tensor IDs. */
export function contract(model: Model, nodes: Operation[], children: Operation[], name: string, op: string, attrs: Record<string, unknown> = {}): Operation[] {
    if (!children.length) return nodes;
    const ids = new Set(children.map(n => n.id)), produced = new Set(children.flatMap(n => n.outputs));
    const external = new Set([...model.outputs, ...nodes.filter(n => !ids.has(n.id)).flatMap(n => [...n.inputs, ...Object.values(n.parameters ?? {})])]);
    const outputs = [...produced].filter(id => external.has(id));
    if (!outputs.length) outputs.push(...children.at(-1)!.outputs);
    const inputs = [...new Set(children.flatMap(n => [...n.inputs, ...Object.values(n.parameters ?? {})]).filter(id => !produced.has(id)))];
    const unit: Operation = { id: `${op}:${children[0].id}`, name, op, group: children[0].group, inputs, outputs,
        attrs: { children, description: name, ...attrs }, source: 'Captured operator composition' };
    const result = nodes.filter(n => !ids.has(n.id)); result.push(unit);
    // Non-convex groups would create cycles. Keep their original graph intact.
    try { return topologicalNodes({ ...model, nodes: result }); } catch { return nodes; }
}

export function semanticStages(model: Model, original: Operation[]): Operation[] {
    let nodes = original;
    const rope = new Map<string, Operation[]>();
    for (const n of nodes) {
        const trace = String(n.attrs?.codeTrace ?? '');
        if (/apply_rotary_pos_emb(?:_interleave)?\(/.test(trace) || n.attrs?.sourceFunction) {
            const key = n.group ?? ''; rope.set(key, [...rope.get(key) ?? [], n]);
        }
    }
    for (const children of rope.values()) if (children.length > 1)
        nodes = contract(model, nodes, children, 'RoPE · 注入相对位置信息', 'semantic', {
            motionKind: 'rope', description: '将 Q、K 的成对通道按位置对应的 cos / sin 旋转，使注意力点积携带相对位置。V 不参与旋转。原始乘法、配对重排和求和可继续展开。',
            formula: "\\begin{bmatrix}q'_1\\\\q'_2\\end{bmatrix}=\\begin{bmatrix}\\cos\\theta&-\\sin\\theta\\\\\\sin\\theta&\\cos\\theta\\end{bmatrix}\\begin{bmatrix}q_1\\\\q_2\\end{bmatrix}",
        });
    for (const softmax of [...nodes].filter(n => n.op === 'softmax' && n.attrs?.attentionRole === 'probability')) {
        const children = [softmax], producer = new Map(nodes.flatMap(n => n.outputs.map(id => [id, n] as const)));
        const consumers = (id: string) => nodes.filter(n => n.inputs.includes(id));
        const visit = (id: string) => {
            const p = producer.get(id);
            if (!p || !['multiply', 'divide', 'add', 'cast', 'identity', 'slice', 'reshape', 'routing_masked_fill'].includes(p.op) || p.outputs.some(t => consumers(t).some(n => !children.includes(n)))) return;
            children.unshift(p); p.inputs.forEach(visit);
        };
        softmax.inputs.forEach(visit);
        let tail = softmax;
        while (true) {
            const next = consumers(tail.outputs[0]);
            if (next.length !== 1 || !['cast', 'identity', 'dropout'].includes(next[0].op)) break;
            // Only remove dropout if its captured values prove it is an evaluation-time identity.
            if (next[0].op === 'dropout') {
                const a = model.tensors.find(t => t.id === tail.outputs[0])?.data?.values, b = model.tensors.find(t => t.id === next[0].outputs[0])?.data?.values;
                if (!a || !b || a.length !== b.length || a.some((v, i) => v !== b[i])) break;
            }
            children.push(next[0]); tail = next[0];
        }
        nodes = contract(model, nodes, children, '注意力分配 · 缩放、掩码与归一化', 'semantic', {
            ...softmax.attrs, motionKind: 'softmax', dim: -1, attentionRole: 'probability',
            description: '先将 QKᵀ 按头维度缩放，叠加实际掩码，再沿 Key 轴做 Softmax。每个 Query 对应一行概率；被屏蔽的位置概率为零。',
            formula: 'P=\\operatorname{softmax}_{K}(QK^{\\mathsf T}/\\sqrt{d_k}+M)',
        });
    }
    for (const gate of [...nodes].filter(n=>n.op==='multiply')) {
        const activation=nodes.find(n=>['silu','gelu','sigmoid'].includes(n.op)&&n.outputs.some(id=>gate.inputs.includes(id)));
        if(!activation||nodes.some(n=>n.id!==gate.id&&n.inputs.some(id=>activation.outputs.includes(id))))continue;
        nodes=contract(model,nodes,[activation,gate],activation.op==='silu'?'SwiGLU · 内容门控':'激活门控 · 调制内容通道','semantic',{
            motionKind:'gating',description:'门投影经过激活后，与内容投影按通道相乘；每个门值控制对应通道的通过幅度。两个投影保留独立的可学习权重。',
            formula:activation.op==='silu'?'h=\\operatorname{SiLU}(xW_g)\\odot(xW_u)':`h=\\operatorname{${activation.op}}(g)\\odot u`,
        });
    }
    return nodes.map(n => {
        if(n.op==='linear'){
            const key=(n.group??'').split('.').at(-1)??'';
            const names:Record<string,string>={q_proj:'Q · Query 投影',k_proj:'K · Key 投影',v_proj:'V · Value 投影',q_a_proj:'Q · 低秩压缩',q_b_proj:'Q · 多头投影',kv_a_proj_with_mqa:'KV · 内容与位置压缩',kv_b_proj:'KV · 多头展开',o_proj:'输出投影',out_proj:'输出投影',gate_proj:'门投影',up_proj:'内容升维',down_proj:'输出降维'};
            if(names[key])return {...n,attrs:{...n.attrs,displayLabel:names[key]}};
        }
        if (n.attrs?.residualInput) return { ...n, name: '残差汇合 · 保留输入信息', attrs: { ...n.attrs, description: '将子层输出与跳连接携带的输入相加，保留原有表征并叠加本层更新。' } };
        return n;
    });
}

/** Explain coordinate work by its producer and consumer, never by an axis verb alone. */
export function coordinatePurpose(model: Model, children: Operation[], inputs: string[], outputs: string[]) {
    const producer = new Map(model.nodes.flatMap(n => n.outputs.map(id => [id, n] as const)));
    const ancestors = new Set<string>();
    const walk = (id: string, depth = 0) => { const p = producer.get(id); if (!p || depth > 8 || ancestors.has(p.id)) return; ancestors.add(p.id); if (p.op !== 'linear') p.inputs.forEach(t => walk(t, depth + 1)); };
    inputs.forEach(id => walk(id));
    const upstream = model.nodes.filter(n => ancestors.has(n.id)), downstream = model.nodes.filter(n => n.inputs.some(id => outputs.includes(id)));
    const path = upstream.map(n => `${n.group ?? ''} ${n.name}`).join(' '), roles = upstream.flatMap(n => n.attrs?.projectionRoles as string[] ?? []);
    const role = /kv_.*proj/.test(path) ? 'KV' : /q_.*proj|q_proj/.test(path) ? 'Q' : /k_proj/.test(path) ? 'K' : /v_proj/.test(path) ? 'V' : roles.join('/');
    if (upstream.some(n => n.attrs?.attentionRole === 'context')) return { title: '合并多头 · 恢复 Token 通道', description: '将各头的 Value 汇聚结果从 [B,H,T,D] 重排为 [B,T,H·D]，交给输出投影。数值不变，恢复每个 Token 的完整通道。', motionKind: 'reorder' };
    if (children.some(n => n.op === 'expand')) return { title: `${role || 'KV'} · 对齐共享头`, description: '将可共享的维度扩展到目标头数；多个目标坐标引用同一源元素。广播不引入新权重，也不改变该元素的数值。', motionKind: 'broadcast' };
    if (downstream.some(n => n.attrs?.attentionRole === 'score') && children.some(n => n.op === 'transpose')) return { title: 'Key 对齐 · 让通道参与点积', description: '将 Key 的最后两轴从 [Token,通道] 交换为 [通道,Token]，让每个 Query 的通道与各 Key 的通道相乘求和，得到 Token × Token 相似度。数值保持不变。', motionKind: 'reorder' };
    return { title: role ? `${role} · 按注意力头组织` : '张量布局 · 对齐下游输入', description: role ? `把 ${role} 投影通道拆成 Head × 每头通道，并将 Head 放在 Token 轴之前，以便各头独立计算；只改变元素坐标。` : `将实际输入形状重排为下游需要的形状。拆分、合并或交换轴不等于数值计算；每个目标格仍可追溯到原始坐标。`, motionKind: 'reorder' };
}
