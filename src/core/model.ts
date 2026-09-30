/** Versioned interchange contract. Shapes are logical row-major coordinates. */
export type Dimension = number | string;
export interface Tensor {
    id: string;
    shape: Dimension[];
    dtype: string;
    role: 'input' | 'activation' | 'parameter' | 'buffer' | 'constant';
    /** A contiguous, flattened window; missing values are unknown, never zero. */
    data?: {
        offset: number;
        values: (number | null)[];
    };
    stats?: {
        min: number;
        max: number;
        absmax: number;
    };
    stride?: number[];
    source?: string;
}
export interface Operation {
    id: string;
    name: string;
    op: string;
    inputs: string[];
    outputs: string[];
    parameters?: Record<string, string>;
    attrs?: Record<string, unknown>;
    group?: string;
    source?: string;
}
export interface Model {
    format: 'silicondevine';
    version: 1;
    name: string;
    tensors: Tensor[];
    nodes: Operation[];
    inputs: string[];
    outputs: string[];
    producer?: {
        backend: string;
        version?: string;
    };
    notes?: string[];
}
export function numel(shape: Dimension[]): number | undefined {
    if (shape.some(n => typeof n !== 'number'))
        return undefined;
    const n = (shape as number[]).reduce((a, b) => a * b, 1);
    return Number.isSafeInteger(n) ? n : undefined;
}
export function valueAt(t: Tensor, index: number): number {
    const n = t.data?.values[index - t.data.offset];
    return typeof n === 'number' && Number.isFinite(n) ? n : NaN;
}
export function coordinates(index: number, shape: number[]): number[] {
    const result = Array(shape.length).fill(0);
    for (let d = shape.length - 1; d >= 0; d--) {
        result[d] = index % shape[d];
        index = Math.floor(index / shape[d]);
    }
    return result;
}
export function flatIndex(coords: number[], shape: number[]): number { return coords.reduce((n, v, i) => n * shape[i] + v, 0); }
/** Validate before allocation. Reject malformed topology instead of silently rewriting it. */
export function validateModel(input: unknown): Model {
    const m = input as Model;
    if (!m || m.format !== 'silicondevine' || m.version !== 1)
        throw Error('需要 SiliconDevine v1 模型文件。');
    if (typeof m.name !== 'string' || !Array.isArray(m.tensors) || !Array.isArray(m.nodes) || !Array.isArray(m.inputs) || !Array.isArray(m.outputs))
        throw Error('模型缺少名称、张量、节点或输入输出。');
    if (m.nodes.length > 10000 || m.tensors.length > 40000)
        throw Error('超过图规模上限：10,000 个算子 / 40,000 个张量。请导出子模块。');
    const tensors = new Map<string, Tensor>(), producers = new Map<string, string>(), ids = new Set<string>();
    let stored = 0;
    for (const t of m.tensors) {
        if (!t || typeof t.id !== 'string' || !t.id || tensors.has(t.id) || !Array.isArray(t.shape) || typeof t.dtype !== 'string' || !['input', 'activation', 'parameter', 'buffer', 'constant'].includes(t.role))
            throw Error('张量定义无效或 ID 重复。');
        if (t.shape.length > 16 || t.shape.some(d => typeof d === 'number' ? !Number.isSafeInteger(d) || d < 0 : typeof d !== 'string' || !d))
            throw Error(`无效形状：${t.id}`);
        if (t.stats && (![t.stats.min,t.stats.max,t.stats.absmax].every(Number.isFinite) || t.stats.absmax < 0 || t.stats.min > t.stats.max))
            throw Error(`无效数值统计：${t.id}`);
        if (t.data) {
            if (!Array.isArray(t.data.values) || !Number.isSafeInteger(t.data.offset) || t.data.offset < 0 || t.data.values.some(v => v !== null && (typeof v !== 'number' || !Number.isFinite(v))))
                throw Error(`无效数值窗口：${t.id}`);
            stored += t.data.values.length;
            const size = numel(t.shape);
            if (size !== undefined && t.data.offset + t.data.values.length > size)
                throw Error(`数值窗口超出张量：${t.id}`);
        }
        if (stored > 2000000)
            throw Error('内嵌数值超过 2,000,000 个；请降低导出窗口。');
        tensors.set(t.id, t);
    }
    const exists = (id: string) => { if (!tensors.has(id))
        throw Error(`缺失张量：${id}`); };
    for (const n of m.nodes) {
        if (!n || typeof n.id !== 'string' || !n.id || ids.has(n.id) || typeof n.op !== 'string' || typeof n.name !== 'string' || !Array.isArray(n.inputs) || !Array.isArray(n.outputs) || !n.outputs.length)
            throw Error('算子无效或 ID 重复。');
        ids.add(n.id);
        [...n.inputs, ...n.outputs, ...Object.values(n.parameters ?? {})].forEach(exists);
        for (const o of n.outputs) {
            if (producers.has(o))
                throw Error(`张量有多个生产者：${o}`);
            producers.set(o, n.id);
        }
    }
    [...m.inputs, ...m.outputs].forEach(exists);
    topologicalNodes(m);
    return m;
}
export function topologicalNodes(m: Model): Operation[] {
    const producer = new Map(m.nodes.flatMap(n => n.outputs.map(o => [o, n.id] as const))), pending = new Map<string, number>(), children = new Map<string, Operation[]>(), nodes: Operation[] = [], queue: Operation[] = [];
    for (const n of m.nodes) {
        const parents = new Set([...n.inputs, ...Object.values(n.parameters ?? {})].map(i => producer.get(i)).filter((id): id is string => id !== undefined));
        pending.set(n.id, parents.size);
        if (!parents.size)
            queue.push(n);
        for (const parent of parents)
            children.set(parent, [...children.get(parent) ?? [], n]);
    }
    for (let cursor = 0; cursor < queue.length; cursor++) {
        const n = queue[cursor];
        nodes.push(n);
        for (const child of children.get(n.id) ?? []) {
            const left = pending.get(child.id)! - 1;
            pending.set(child.id, left);
            if (left === 0)
                queue.push(child);
        }
    }
    if (nodes.length !== m.nodes.length)
        throw Error('计算图含环；请导出一个已展开的执行路径。');
    return nodes;
}
