import { coordinates, flatIndex, valueAt, type Tensor } from './model.js';
import type { OperatorVisual, Dependency } from './operators.js';
export function installLLMOperators(register: (name: string, visual: OperatorVisual) => void) {
    register('repeat', { label: '张量复用', color: '#d3b788', formula: '按 repeats 沿指定各轴复用实际元素', detail: 'exact', dependencies: (n, out, index, t) => { const x = t.get(n.inputs[0])!, shape = x.shape as number[], c = coordinates(index, out.shape as number[]).slice(-shape.length).map((v, j) => v % shape[j]); return [{ tensor: x.id, index: flatIndex(c, shape) }]; } });
    const broadcast = (tensor: Tensor, out: Tensor, index: number) => {
        const shape = tensor.shape as number[], coords = coordinates(index, out.shape as number[]);
        return flatIndex(shape.map((v, i) => v === 1 ? 0 : coords[coords.length - shape.length + i]), shape);
    };
    for (const op of ['divide', 'power'])
        register(op, { label: op === 'divide' ? '逐元素除法' : '幂运算', color: '#bba4e5', formula: op === 'divide' ? 'y = a / b' : 'y = aᵇ', detail: 'exact', dependencies: (n, out, i, t) => n.inputs.map(id => ({ tensor: id, index: broadcast(t.get(id)!, out, i) })) });
    for (const [op, curve] of Object.entries({ rsqrt: (x: number) => 1 / Math.sqrt(x), negative: (x: number) => -x, sin: Math.sin, cos: Math.cos }))
        register(op, { label: op, color: '#bba4e5', formula: 'y = ' + op + '(x)', detail: 'exact', curve, dependencies: (n, out, i) => [{ tensor: n.inputs[0], index: i }] });
    register('cast', { label: '类型转换', color: '#a8c4e5', formula: '按目标 dtype 转换数值', detail: 'exact', dependencies: (n, out, i) => [{ tensor: n.inputs[0], index: i }] });
    register('expand', { label: '广播维度', color: '#a8c4e5', formula: '重复引用单元素轴，保留实际坐标', detail: 'exact', dependencies: (n, out, i, t) => [{ tensor: n.inputs[0], index: broadcast(t.get(n.inputs[0])!, out, i) }] });
    register('mean', { label: '均值归约', color: '#bba4e5', formula: 'y = Σx / N', detail: 'exact', dependencies: (n, out, i, t) => {
            const x = t.get(n.inputs[0])!, shape = x.shape as number[], dims = ((n.attrs?.dims ?? shape.map((_, j) => j)) as number[]).map(d => (d + shape.length) % shape.length), oc = coordinates(i, out.shape as number[]);
            let k = 0;
            const base = shape.map((_, j) => dims.includes(j) ? 0 : oc[n.attrs?.keepdim ? j : k++]);
            const reduced = dims.map(j => shape[j]), count = reduced.reduce((a, b) => a * b, 1);
            if (count > 65536)
                return [];
            return Array.from({ length: count }, (_, j) => { const rc = coordinates(j, reduced), c = [...base]; dims.forEach((d, k) => c[d] = rc[k]); return { tensor: x.id, index: flatIndex(c, shape), weight: 1 / count }; });
        } });
    register('subtract', { label: '配对差值', color: '#d3b788', formula: 'y = a − b', detail: 'exact', dependencies: (n, out, i, t) => n.inputs.map((id, k) => ({ tensor: id, index: broadcast(t.get(id)!, out, i), weight: k === 0 ? 1 : -Number(n.attrs?.alpha ?? 1) })) });
    register('rmsnorm', { label: 'RMSNorm', color: '#bba4e5', formula: 'y = x · γ / √(mean(x²) + ε)', detail: 'exact', dependencies: (n, out, index, t) => {
            const x = t.get(n.inputs[0])!, normalized = (n.attrs?.normalized_shape ?? [x.shape.at(-1)]) as number[], size = normalized.reduce((a, b) => a * b, 1), base = Math.floor(index / size) * size;
            const deps: Dependency[] = Array.from({ length: size }, (_, j) => ({ tensor: x.id, index: base + j }));
            if (n.parameters?.weight)
                deps.push({ tensor: n.parameters.weight, index: index % size });
            return deps;
        } });
    register('repeat_kv', { label: 'KV 共享 · GQA', color: '#89cfe0', formula: 'K,V[h] = K,V[⌊h / repeats⌋]', detail: 'exact', dependencies: (n, out, index, t) => {
            const x = t.get(n.inputs[0])!, shape = x.shape as number[], c = coordinates(index, out.shape as number[]);
            c[c.length - 3] = Math.floor(c[c.length - 3] / Number(n.attrs?.repeats ?? 1));
            return [{ tensor: x.id, index: flatIndex(c, shape) }];
        } });
    for (const op of ['chunk', 'split'])
        register(op, { label: '张量分区', color: '#89cfe0', formula: '沿指定轴保留连续坐标区间', detail: 'exact', dependencies: (n, out, index, t) => {
                const x = t.get(n.inputs[0])!, s = x.shape as number[], axis = (Number(n.attrs?.dim ?? 0) + s.length) % s.length, c = coordinates(index, out.shape as number[]), slot = n.outputs.indexOf(out.id);
                c[axis] += n.outputs.slice(0, slot).reduce((sum, id) => sum + Number(t.get(id)!.shape[axis]), 0);
                return [{ tensor: x.id, index: flatIndex(c, s) }];
            } });
    register('stack', { label: '分量堆叠', color: '#c3ace6', formula: '沿指定轴按输入顺序堆叠', detail: 'exact', dependencies: (n, out, index, t) => {
            const c = coordinates(index, out.shape as number[]), axis = (Number(n.attrs?.dim ?? 0) + c.length) % c.length, slot = c.splice(axis, 1)[0], x = t.get(n.inputs[slot])!;
            return [{ tensor: x.id, index: flatIndex(c, x.shape as number[]) }];
        } });
    register('arange', { label: '等差坐标', color: '#a9bfd8', formula: 'position[i] = start + i · step', detail: 'exact', dependencies: () => [] });
    register('attention_mask', { label: '因果 / 外部掩码', color: '#dfb076', formula: 'j > i → −∞；可见位置保留分数', detail: 'exact', dependencies: (n, out, index, t) => {
            const c = coordinates(index, out.shape as number[]), masked = n.attrs?.causal && c.at(-1)! > c.at(-2)!;
            const mask = n.inputs[1] ? t.get(n.inputs[1]) : undefined, mi = mask ? broadcast(mask, out, index) : 0;
            if (masked || mask && n.attrs?.mask_kind === 'bool' && !valueAt(mask, mi))
                return mask ? [{ tensor: mask.id, index: mi }] : [];
            return [{ tensor: n.inputs[0], index }, ...(mask ? [{ tensor: mask.id, index: mi }] : [])];
        } });
    register('attention', { label: '融合注意力', color: '#99c7e8', formula: 'softmax(QKᵀ / √d + mask) V', detail: 'boundary', dependencies: () => [] });
    register('function', { label: 'RoPE · 配对旋转', color: '#bda7ec', formula: 'Q/K 与实际 cos、sin 的配对旋转；点击进入捕获的函数体', detail: 'boundary', dependencies: () => [] });
    register('module', { label: '功能模块', color: '#8ebfe0', formula: '真实边界张量 · 点击展开内部计算', detail: 'boundary', dependencies: () => [] });
}
