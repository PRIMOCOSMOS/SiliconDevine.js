import { coordinates, flatIndex, valueAt, type Tensor } from './model.js';
import type { Dependency, OperatorVisual } from './operators.js';
type Register = (name: string, visual: OperatorVisual) => void;
const shape = (t: Tensor) => t.shape as number[];
const list = (v: unknown, n: number, fallback: number) => typeof v === 'number' ? Array(n).fill(v) : Array.isArray(v) && v.length ? v as number[] : Array(n).fill(fallback);
const product = (s: number[]) => s.reduce((a, b) => a * b, 1);
const reflect = (v: number, size: number) => {
    if (size <= 1)
        return 0;
    const span = 2 * (size - 1), p = ((v % span) + span) % span;
    return p < size ? p : span - p;
};
export function installExtendedOperators(register: Register) {
    for (const op of ['select', 'slice'])
        register(op, { label: op === 'select' ? '轴索引' : '张量切片', color: '#86bed3', formula: '保留源坐标映射', detail: 'exact', dependencies: (n, out, index, t) => {
                const x = t.get(n.inputs[0]);
                if (!x)
                    return [];
                const s = shape(x), axis = (Number(n.attrs?.dim ?? 0) + s.length) % s.length, c = coordinates(index, shape(out));
                if (op === 'select') {
                    let selected = Number(n.attrs?.index ?? 0);
                    if (selected < 0)
                        selected += s[axis];
                    c.splice(axis, 0, selected);
                }
                else {
                    let start = Number(n.attrs?.start ?? 0);
                    start = start < 0 ? Math.max(0, s[axis] + start) : Math.min(start, s[axis]);
                    c[axis] = start + c[axis] * Number(n.attrs?.step ?? 1);
                }
                return [{ tensor: x.id, index: flatIndex(c, s) }];
            } });
    for (const [op, curve] of Object.entries({ silu: (x: number) => x / (1 + Math.exp(-x)), leaky_relu: (x: number, a: Record<string, unknown>) => x >= 0 ? x : Number(a.negative_slope ?? .01) * x, elu: (x: number, a: Record<string, unknown>) => x > 0 ? x : Number(a.alpha ?? 1) * Math.expm1(x), softplus: (x: number, a: Record<string, unknown>) => { const b = Number(a.beta ?? 1); return x * b > Number(a.threshold ?? 20) ? x : Math.log1p(Math.exp(x * b)) / b; } }))
        register(op, { label: op, color: '#b6a0e6', formula: `y = ${op}(x)`, detail: 'exact', curve, dependencies: (n, _, i) => [{ tensor: n.inputs[0], index: i }] });
    register('concat', { label: '通道 / 轴拼接', color: '#8cbedd', formula: 'Y = cat(X₁, …, Xₖ; dim)', detail: 'exact', dependencies: (n, out, i, t) => {
            const s = shape(out), axis = (Number(n.attrs?.dim ?? 0) + s.length) % s.length, c = coordinates(i, s);
            let cursor = c[axis];
            for (const id of n.inputs) {
                const x = t.get(id)!;
                if (cursor < Number(x.shape[axis])) {
                    c[axis] = cursor;
                    return [{ tensor: id, index: flatIndex(c, shape(x)) }];
                }
                cursor -= Number(x.shape[axis]);
            }
            return [];
        } });
    register('matmul', { label: '矩阵 / 向量乘法', color: '#68e6d2', formula: 'C[…,i,j] = Σ A[…,i,k] B[…,k,j]', detail: 'exact', dependencies: (n, out, index, t) => {
            const a = t.get(n.inputs[0]), b = t.get(n.inputs[1]);
            if (!a || !b)
                return [];
            const as = shape(a), bs = shape(b), av = as.length === 1, bv = bs.length === 1, oc = coordinates(index, shape(out)), bat = oc.slice(0, Math.max(0, oc.length - Number(!av) - Number(!bv))), row = av ? 0 : oc[oc.length - Number(!bv) - 1], col = bv ? 0 : oc.at(-1)!, count = as.at(-1)!;
            const batch = (s: number[]) => s.length <= 2 ? [] : bat.slice(bat.length - (s.length - 2)).map((v, d) => s[d] === 1 ? 0 : v), deps: Dependency[] = [];
            for (let k = 0; k < count; k++) {
                const ai = av ? k : flatIndex([...batch(as), row, k], as), bi = bv ? k : flatIndex([...batch(bs), k, col], bs);
                deps.push({ tensor: a.id, index: ai, weight: valueAt(b, bi), parameter: b.id, parameterIndex: bi }, { tensor: b.id, index: bi });
            }
            return deps;
        } });
    for (const dimensions of [1, 2, 3])
        for (const transpose of [false, true])
            register(transpose ? `conv_transpose${dimensions}d` : `conv${dimensions}d`, { label: `${dimensions}D ${transpose ? '转置卷积' : '卷积'}`, color: '#84bce6', formula: transpose ? 'Y[n,o,p] += X[n,c,q] W[c,o,k]' : 'Y[n,o,p] = b[o] + Σ W[o,c,k] X[n,c,p·s−pad+k·d]', detail: 'exact', dependencies: (n, out, index, t) => {
                    const x = t.get(n.inputs[0]), w = t.get(n.parameters?.weight ?? '');
                    if (!x || !w)
                        return [];
                    const xs = x.shape.length === dimensions + 1 ? [1, ...shape(x)] : shape(x), ws = shape(w), os = out.shape.length === dimensions + 1 ? [1, ...shape(out)] : shape(out), oc = coordinates(index, os), a = n.attrs ?? {}, stride = list(a.stride, dimensions, 1), dilation = list(a.dilation, dimensions, 1), kernel = ws.slice(2), pad = a.padding === 'same' ? kernel.map((k, j) => Math.floor(Math.max(0, (os[j + 2] - 1) * stride[j] + dilation[j] * (k - 1) + 1 - xs[j + 2]) / 2)) : list(a.padding, dimensions, 0), groups = Number(a.groups ?? 1), g = Math.floor(oc[1] / (os[1] / groups)), channels = xs[1] / groups, deps: Dependency[] = [];
                    for (let c = 0; c < channels; c++)
                        for (let k = 0; k < product(kernel); k++) {
                            const kc = coordinates(k, kernel), raw = kc.map((v, j) => transpose ? (oc[j + 2] + pad[j] - v * dilation[j]) / stride[j] : oc[j + 2] * stride[j] - pad[j] + v * dilation[j]);
                            if (raw.some(v => !Number.isInteger(v)))
                                continue;
                            const spatial = raw.map((v, j) => a.padding_mode === 'reflect' ? reflect(v, xs[j + 2]) : a.padding_mode === 'replicate' ? Math.max(0, Math.min(xs[j + 2] - 1, v)) : a.padding_mode === 'circular' ? ((v % xs[j + 2]) + xs[j + 2]) % xs[j + 2] : v);
                            if (spatial.some((v, j) => v < 0 || v >= xs[j + 2]))
                                continue;
                            const wi = flatIndex(transpose ? [g * channels + c, oc[1] % (os[1] / groups), ...kc] : [oc[1], c, ...kc], ws);
                            deps.push({ tensor: x.id, index: flatIndex([oc[0], g * channels + c, ...spatial], xs), weight: valueAt(w, wi), parameter: w.id, parameterIndex: wi });
                        }
                    if (n.parameters?.bias)
                        deps.push({ tensor: n.parameters.bias, index: oc[1], weight: 1 });
                    return deps;
                } });
    for (const dimensions of [1, 2, 3])
        for (const mode of ['max', 'avg'])
            for (const adaptive of [false, true]) {
                const op = adaptive ? `adaptive_${mode}_pool${dimensions}d` : `${mode}pool${dimensions}d`;
                register(op, { label: `${adaptive ? '自适应' : ''}${mode === 'max' ? '最大' : '平均'}池化`, color: '#cfb489', formula: mode === 'max' ? 'Y[p] = max X[感受野]' : 'Y[p] = Σ X[感受野] / divisor', detail: 'exact', dependencies: (n, out, index, t) => {
                        const x = t.get(n.inputs[0])!;
                        if (!x)
                            return [];
                        const xs = x.shape.length === dimensions + 1 ? [1, ...shape(x)] : shape(x), os = out.shape.length === dimensions + 1 ? [1, ...shape(out)] : shape(out), oc = coordinates(index, os), a = n.attrs ?? {}, kernel = list(a.kernel_size, dimensions, 2), stride = list(a.stride, dimensions, 0).map((v, j) => v || kernel[j]), pad = list(a.padding, dimensions, 0), dilation = list(a.dilation, dimensions, 1);
                        const start = oc.slice(2).map((p, j) => adaptive ? Math.floor(p * xs[j + 2] / os[j + 2]) : p * stride[j] - pad[j]);
                        const extent = adaptive ? start.map((s, j) => Math.ceil((oc[j + 2] + 1) * xs[j + 2] / os[j + 2]) - s) : kernel;
                        const refs: Dependency[] = [];
                        let paddedCount = 0;
                        for (let k = 0; k < product(extent); k++) {
                            const kc = coordinates(k, extent), p = kc.map((v, j) => start[j] + v * (adaptive ? 1 : dilation[j]));
                            if (p.every((v, j) => v >= -pad[j] && v < xs[j + 2] + pad[j]))
                                paddedCount++;
                            if (p.every((v, j) => v >= 0 && v < xs[j + 2]))
                                refs.push({ tensor: x.id, index: flatIndex([...oc.slice(0, 2), ...p], xs) });
                        }
                        if (mode === 'avg') {
                            const divisor = Number(a.divisor_override ?? (adaptive || a.count_include_pad === false ? refs.length : paddedCount));
                            return refs.map(d => ({ ...d, weight: 1 / divisor }));
                        }
                        const values = refs.map(d => valueAt(x, d.index)), maximum = Math.max(...values), winner = values.indexOf(maximum);
                        return refs.map((d, i) => ({ ...d, weight: Number.isFinite(maximum) ? Number(i === winner) : NaN }));
                    } });
            }
    for (const op of ['batchnorm', 'instancenorm', 'groupnorm'])
        register(op, { label: op === 'batchnorm' ? '批归一化' : op === 'groupnorm' ? '分组归一化' : '实例归一化', color: '#b6a0e6', formula: 'Y = (X − μ) / √(σ² + ε) · γ + β', detail: 'exact', dependencies: (n, out, index, t) => {
                const x = t.get(n.inputs[0])!;
                if (!x)
                    return [];
                const s = shape(x), c = coordinates(index, s), channel = c[1], spatial = product(s.slice(2)), a = n.attrs ?? {}, p = n.parameters ?? {}, gamma = p.weight ? valueAt(t.get(p.weight)!, channel) : 1, eps = Number(a.eps ?? 1e-5), refs: Dependency[] = [];
                if ((op === 'batchnorm' || op === 'instancenorm') && p.running_mean && p.running_var && a.training !== true) {
                    const inv = gamma / Math.sqrt(valueAt(t.get(p.running_var)!, channel) + eps);
                    refs.push({ tensor: x.id, index, weight: inv }, { tensor: p.running_mean, index: channel, weight: -inv }, { tensor: p.running_var, index: channel });
                }
                else {
                    const indices: number[] = [];
                    const groups = op === 'groupnorm' ? Number(a.num_groups ?? 1) : s[1], groupSize = s[1] / groups, first = op === 'batchnorm' ? channel : Math.floor(channel / groupSize) * groupSize, last = op === 'batchnorm' ? channel + 1 : first + groupSize;
                    for (let batch = op === 'batchnorm' ? 0 : c[0]; batch < (op === 'batchnorm' ? s[0] : c[0] + 1); batch++)
                        for (let ch = first; ch < last; ch++)
                            for (let i = 0; i < spatial; i++)
                                indices.push((batch * s[1] + ch) * spatial + i);
                    const values = indices.map(i => valueAt(x, i)), mean = values.reduce((a, b) => a + b, 0) / values.length, variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length, scale = gamma / Math.sqrt(variance + eps);
                    indices.forEach(i => refs.push({ tensor: x.id, index: i, weight: scale * (Number(i === index) - 1 / indices.length) }));
                }
                if (p.weight)
                    refs.push({ tensor: p.weight, index: channel });
                if (p.bias)
                    refs.push({ tensor: p.bias, index: channel, weight: 1 });
                return refs;
            } });
    register('embedding', { label: '嵌入查表', color: '#70d5c9', formula: 'Y[…,j] = W[index[…],j]', detail: 'exact', dependencies: (n, out, index, t) => {
            const x = t.get(n.inputs[0]), w = t.get(n.parameters?.weight ?? '');
            if (!x || !w)
                return [];
            const width = Number(w.shape[1]), i = Math.floor(index / width), row = valueAt(x, i);
            return Number.isInteger(row) ? [{ tensor: x.id, index: i }, { tensor: w.id, index: row * width + index % width, weight: 1 }] : [];
        } });
}
