import { installAPIOperators } from './apiOperators.js';
import { installSpatialOperators } from './spatialOperators.js';
import { installConditionalOperators } from './conditionalOperators.js';
import { coordinates, flatIndex, valueAt, type Tensor, type Operation } from './model.js';
import { installExtendedOperators } from './extendedOperators.js';
import { installLLMOperators } from './llmOperators.js';
/** A dependency names an actual scalar coordinate. Weight is the actual multiplier. */
export interface Dependency {
    tensor: string;
    index: number;
    weight?: number;
    parameter?: string;
    parameterIndex?: number;
}
export interface OperatorVisual {
    label: string;
    color: string;
    formula: string;
    detail: 'exact' | 'boundary';
    dependencies: (node: Operation, out: Tensor, index: number, tensors: Map<string, Tensor>) => Dependency[];
    curve?: (x: number, attrs: Record<string, unknown>) => number;
}
const registry = new Map<string, OperatorVisual>();
export function registerOperator(name: string, visual: OperatorVisual) { registry.set(name, visual); }
const boundary: OperatorVisual = { label: '自定义算子', color: '#96a3be', formula: '输入 → 已捕获的输出', detail: 'boundary', dependencies: () => [] };
export const operatorVisual = (op: string) => registry.get(op) ?? boundary;
const pair = (n: Operation, t: Map<string, Tensor>) => [t.get(n.inputs[0])!, t.get(n.parameters?.weight ?? '')!] as const;
registerOperator('linear', { label: '全连接', color: '#68e6d2', formula: 'y = Wx + b', detail: 'exact', dependencies: (n, out, index, t) => {
        const [x, w] = pair(n, t);
        if (!x || !w)
            return [];
        const input = x.shape.at(-1) as number, output = out.shape.at(-1) as number, row = Math.floor(index / output), o = index % output;
        if (!Number.isFinite(input) || input > 65536)
            return [];
        const deps = Array.from({ length: input }, (_, i) => ({ tensor: x.id, index: row * input + i, weight: valueAt(w, o * input + i), parameter: w.id, parameterIndex: o * input + i }));
        if (n.parameters?.bias)
            deps.push({ tensor: n.parameters.bias, index: o, weight: 1, parameter: n.parameters.bias, parameterIndex: o });
        return deps;
    } });
function erf(x: number) { const s = x < 0 ? -1 : 1, a = Math.abs(x), t = 1 / (1 + .3275911 * a); return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-a * a)); }
const functions: Record<string, (x: number, a: Record<string, unknown>) => number> = { relu: x => Math.max(0, x), sigmoid: x => 1 / (1 + Math.exp(-x)), tanh: x => Math.tanh(x), gelu: (x, a) => a.approximate === 'tanh' ? .5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + .044715 * x * x * x))) : .5 * x * (1 + erf(x / Math.SQRT2)) };
registerOperator('exp',{label:'指数变换',color:'#a7caff',formula:'y = exp(x)',detail:'exact',curve:Math.exp,dependencies:(n,_,i)=>[{tensor:n.inputs[0],index:i}]});
registerOperator('standard_normal',{label:'标准高斯噪声',color:'#c6b4ff',formula:'ε ∼ N(0,I) · 显示本次 PyTorch 实际抽样',detail:'exact',dependencies:()=>[]});
registerOperator('gaussian_sample',{label:'高斯重参数采样',color:'#c6b4ff',formula:'z = μ + exp(½ log σ²) ⊙ ε,  ε ∼ N(0,I)',detail:'exact',dependencies:(n,_,i)=>{const u=n.attrs?.gaussian as Record<string,string>;return [{tensor:u.mean,index:i},{tensor:u.logvar,index:i},{tensor:u.noise,index:i}];}});
for (const [name, curve] of Object.entries(functions))
    registerOperator(name, { label: name.toUpperCase(), color: '#b6a0e6', formula: name === 'relu' ? 'y = max(0, x)' : `y = ${name}(x)`, detail: 'exact', curve, dependencies: (n, _, index) => [{ tensor: n.inputs[0], index, weight: 1 }] });
for (const op of ['reshape', 'flatten', 'identity', 'dropout'])
    registerOperator(op, { label: op, color: '#a8c4e5', formula: op === 'dropout' ? 'eval: y = x' : '保持元素，变换形状', detail: 'exact', dependencies: (n, _, index) => [{ tensor: n.inputs[0], index }] });
for (const op of ['add', 'multiply'])
    registerOperator(op, { label: op === 'add' ? '逐元素相加' : '逐元素乘法', color: '#dfb076', formula: op === 'add' ? 'y = a + b' : 'y = a ⊙ b', detail: 'exact', dependencies: (n, out, index, t) => n.inputs.map(id => { const tensor = t.get(id)!, oc = coordinates(index, out.shape as number[]), s = tensor.shape as number[]; const c = oc.slice(oc.length - s.length).map((v, i) => s[i] === 1 ? 0 : v); return { tensor: id, index: flatIndex(c, s) }; }) });
for (const dim of [1, 2, 3])
    registerOperator(`conv${dim}d`, { label: `${dim}D 卷积`, color: '#84bce6', formula: 'Y[n,o,p] = b[o] + Σ W[o,c,k] X[n,c,p·s−pad+k·d]', detail: 'exact', dependencies: (n, out, index, t) => {
            const [x, w] = pair(n, t);
            if (!x || !w)
                return [];
            const xs = x.shape as number[], ws = w.shape as number[], os = out.shape as number[];
            if ([...xs, ...ws, ...os].some(v => typeof v !== 'number'))
                return [];
            const oc = coordinates(index, os), a = n.attrs ?? {}, arr = (v: unknown, def: number) => typeof v === 'number' ? Array(dim).fill(v) : Array.isArray(v) ? v as number[] : Array(dim).fill(def), stride = arr(a.stride, 1), pad = arr(a.padding, 0), dilation = arr(a.dilation, 1), groups = Number(a.groups ?? 1), group = Math.floor(oc[1] / (os[1] / groups)), kernel = ws.slice(2), count = kernel.reduce((a, b) => a * b, 1), deps: Dependency[] = [];
            if (typeof a.padding === 'string')
                return []; // requires resolved padding; never invent coordinates.
            for (let c = 0; c < ws[1]; c++)
                for (let k = 0; k < count; k++) {
                    const kc = coordinates(k, kernel), sp = kc.map((v, j) => oc[j + 2] * stride[j] - pad[j] + v * dilation[j]);
                    if (sp.some((v, j) => v < 0 || v >= xs[j + 2]))
                        continue;
                    const wi = flatIndex([oc[1], c, ...kc], ws);
                    deps.push({ tensor: x.id, index: flatIndex([oc[0], group * ws[1] + c, ...sp], xs), weight: valueAt(w, wi), parameter: w.id, parameterIndex: wi });
                }
            if (n.parameters?.bias)
                deps.push({ tensor: n.parameters.bias, index: oc[1], weight: 1 });
            return deps;
        } });
registerOperator('matmul', { label: '矩阵乘法', color: '#68e6d2', formula: 'C[...,i,j] = Σ A[...,i,k] B[...,k,j]', detail: 'exact', dependencies: (n, out, index, t) => {
        const a = t.get(n.inputs[0]), b = t.get(n.inputs[1]);
        if (!a || !b || a.shape.length < 2 || b.shape.length < 2)
            return [];
        const as = a.shape as number[], bs = b.shape as number[], os = out.shape as number[], oc = coordinates(index, os), batch = oc.slice(0, -2), i = oc.at(-2)!, j = oc.at(-1)!, deps: Dependency[] = [];
        const bc = (s: number[]) => batch.slice(batch.length - (s.length - 2)).map((v, d) => s[d] === 1 ? 0 : v);
        for (let k = 0; k < as.at(-1)!; k++) {
            const ai = flatIndex([...bc(as), i, k], as), bi = flatIndex([...bc(bs), k, j], bs);
            deps.push({ tensor: a.id, index: ai, weight: valueAt(b, bi), parameter: b.id, parameterIndex: bi }, { tensor: b.id, index: bi });
        }
        return deps;
    } });
registerOperator('softmax', { label: 'Softmax', color: '#b6a0e6', formula: 'pᵢ = exp(xᵢ − max x) / Σ exp(xⱼ − max x)', detail: 'exact', dependencies: (n, out, index) => { const s = out.shape as number[], axis = ((Number(n.attrs?.dim ?? -1) % s.length) + s.length) % s.length, c = coordinates(index, s); return Array.from({ length: s[axis] }, (_, i) => ({ tensor: n.inputs[0], index: flatIndex(c.map((v, j) => j === axis ? i : v), s) })); } });
for (const op of ['permute', 'transpose'])
    registerOperator(op, { label: '轴变换', color: '#84bce6', formula: '同一元素，重新排列坐标轴', detail: 'exact', dependencies: (n, out, index, t) => {
            const x = t.get(n.inputs[0])!;
            if (!x)
                return [];
            const rank = x.shape.length, c = coordinates(index, out.shape as number[]), dims = n.attrs?.dims as number[] | undefined;
            if (!dims)
                return [];
            let order = Array.from({ length: rank }, (_, i) => i);
            if (op === 'permute')
                order = dims.map(i => (i + rank) % rank);
            else {
                const [a, b] = dims.map(i => (i + rank) % rank);
                [order[a], order[b]] = [order[b], order[a]];
            }
            const original = Array(rank).fill(0);
            order.forEach((axis, i) => original[axis] = c[i]);
            return [{ tensor: x.id, index: flatIndex(original, x.shape as number[]) }];
        } });
registerOperator('layernorm', { label: '层归一化', color: '#b6a0e6', formula: 'y = (x − μ) / √(σ² + ε) · γ + β', detail: 'exact', dependencies: (n, out, index, t) => {
        const x = t.get(n.inputs[0]);
        if (!x)
            return [];
        const normalized = (n.attrs?.normalized_shape ?? (n.attrs?.arguments as unknown[])?.[0]) as number[] | undefined;
        if (!Array.isArray(normalized))
            return [];
        const size = normalized.reduce((a, b) => a * b, 1);
        if (!size || size > 65536)
            return [];
        const base = Math.floor(index / size) * size, terms: Dependency[] = Array.from({ length: size }, (_, i) => ({ tensor: x.id, index: base + i }));
        for (const key of ['weight', 'bias']) {
            const p = n.parameters?.[key];
            if (p)
                terms.push({ tensor: p, index: index % size });
        }
        return terms;
    } });
for (const op of ['batchnorm', 'maxpool2d', 'avgpool2d', 'concat'])
    registerOperator(op, { ...boundary, label: op, color: op.includes('norm') ? '#b6a0e6' : '#84bce6', formula: '真实输入 / 输出；内部标量依赖待插件扩展' });
installExtendedOperators(registerOperator);
installLLMOperators(registerOperator);
registerOperator('layout', { label: '坐标组织', color: '#84bce6', formula: '只改变坐标组织；元素数值不变', detail: 'exact', dependencies: (node, out, index, tensors) => {
        const children = node.attrs?.children as Operation[] ?? [], producer = new Map(children.flatMap(n => n.outputs.map(id => [id, n] as const)));
        const trace = (tensor: string, i: number, depth: number): Dependency[] => {
            if (depth > children.length)
                return [];
            const n = producer.get(tensor);
            if (!n)
                return [{ tensor, index: i }];
            const value = tensors.get(tensor);
            if (!value)
                return [];
            return operatorVisual(n.op).dependencies(n, value, i, tensors).flatMap(d => trace(d.tensor, d.index, depth + 1));
        };
        return trace(out.id, index, 0);
    } });

installConditionalOperators(registerOperator);

installAPIOperators(registerOperator);
installSpatialOperators(registerOperator);

registerOperator('semantic', { label: '功能组合', color: '#b6a0e6', formula: '保留原始计算子图与真实张量边界', detail: 'exact', dependencies: (node, out, index, tensors) => {
    const children = node.attrs?.children as Operation[] ?? [], producer = new Map(children.flatMap(n => n.outputs.map(id => [id,n] as const)));
    const memo = new Map<string, Dependency[]>();
    const trace = (id:string, i:number, depth=0):Dependency[] => {
        const key=id+':'+i; if(memo.has(key)) return memo.get(key)!;
        const n=producer.get(id), t=tensors.get(id);
        if(!n || !t || depth>children.length) return [{tensor:id,index:i}];
        const unique=new Map<string,Dependency>();
        for(const d of operatorVisual(n.op).dependencies(n,t,i,tensors)) for(const leaf of trace(d.tensor,d.index,depth+1)) unique.set(leaf.tensor+':'+leaf.index,{tensor:leaf.tensor,index:leaf.index});
        const result=[...unique.values()];memo.set(key,result);return result;
    };
    return trace(out.id,index);
}});

// Routing primitives emitted by the execution backend carry exact coordinate
// provenance for this captured path (including data-dependent expert selection).
for(const [kind,label,formula] of [
    ['topk','Top-k 选择','v, i = TopK(x)'],['index','按索引取 Token','y_j = x_{i_j}'],
    ['index_add','专家结果累加','y_i = x_i + Σ_{j: index_j=i} source_j'],
    ['scatter','索引写入','y[index] = value'],['not','布尔取反','y = ¬x'],
    ['masked_fill','掩码填充','y = mask ? c : x'],['one_hot','专家分配掩码','y_{i,e} = [index_i=e]'],
    ['nonzero','选中 Token 坐标','i = nonzero(mask)'],['unbind','分离坐标轴','y_j = x.select(dim,j)'],
    ['constant','常量初始化','y = constant'],
])registerOperator('routing_'+kind,{label,color:'#c3a2e0',formula,detail:'exact',dependencies:(node,out,index)=>{
    const maps=node.attrs?.coordinateDependencies as Dependency[][][]|undefined;
    return maps?.[node.outputs.indexOf(out.id)]?.[index]??[];
}});
