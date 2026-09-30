import { validateModel, numel, type Model, type Tensor, type Operation } from './model';
/** Fluent graph builder; never evaluates code strings. Use operation() for custom DAGs. */
export class ModelBuilder {
    private model: Model;
    private current = '';
    private state: number;
    constructor(name: string, seed = 17) { this.model = { format: 'silicondevine', version: 1, name, tensors: [], nodes: [], inputs: [], outputs: [], producer: { backend: 'declarative' } }; this.state = seed; }
    private random() { this.state = (1664525 * this.state + 1013904223) >>> 0; return (this.state / 4294967296 - .5) * 2; }
    tensor(t: Tensor) { this.model.tensors.push(t); return this; }
    input(id: string, shape: number[], values?: number[]) { const size = numel(shape); if (!shape.length || shape.some(n => !Number.isSafeInteger(n) || n < 1) || size === undefined || size > 65536)
        throw Error('声明式数值演示需要正整数形状，最多 65,536 个元素。'); if (values && (values.length !== size || values.some(n => !Number.isFinite(n))))
        throw Error('输入数值必须完整且为有限数值。'); this.tensor({ id, shape, dtype: 'float32', role: 'input', data: { offset: 0, values: values ?? Array.from({ length: size }, () => this.random()) } }); this.model.inputs.push(id); this.current = id; return this; }
    operation(node: Operation, outputs: Tensor[]) { outputs.forEach(t => this.tensor(t)); this.model.nodes.push(node); this.current = node.outputs[0]; return this; }
    linear(id: string, outFeatures: number) {
        const x = this.model.tensors.find(t => t.id === this.current);
        if (!x?.data)
            throw Error('linear 需要数值输入。');
        const shape = x.shape as number[], inFeatures = shape.at(-1)!;
        if (!Number.isInteger(outFeatures) || outFeatures < 1 || outFeatures > 128 || !inFeatures || inFeatures > 128)
            throw Error('声明式演示通道数需为 1–128。');
        const w = Array.from({ length: outFeatures * inFeatures }, () => this.random() / Math.sqrt(inFeatures)), b = Array.from({ length: outFeatures }, () => this.random() * .1), v = x.data.values as number[], y: number[] = [];
        for (let r = 0; r < v.length / inFeatures; r++)
            for (let o = 0; o < outFeatures; o++) {
                let sum = b[o];
                for (let i = 0; i < inFeatures; i++)
                    sum += v[r * inFeatures + i] * w[o * inFeatures + i];
                y.push(sum);
            }
        this.tensor({ id: id + '.weight', shape: [outFeatures, inFeatures], dtype: 'float32', role: 'parameter', data: { offset: 0, values: w } }).tensor({ id: id + '.bias', shape: [outFeatures], dtype: 'float32', role: 'parameter', data: { offset: 0, values: b } });
        return this.operation({ id, name: id, op: 'linear', inputs: [x.id], outputs: [id + '.out'], parameters: { weight: id + '.weight', bias: id + '.bias' }, group: id }, [{ id: id + '.out', shape: [...shape.slice(0, -1), outFeatures], dtype: 'float32', role: 'activation' as const, data: { offset: 0, values: y } }]);
    }
    activation(id: string, kind: 'relu' | 'gelu' | 'sigmoid' | 'tanh' = 'relu') {
        const x = this.model.tensors.find(t => t.id === this.current)!;
        const fn = (v: number) => kind === 'relu' ? Math.max(0, v) : kind === 'sigmoid' ? 1 / (1 + Math.exp(-v)) : kind === 'tanh' ? Math.tanh(v) : .5 * v * (1 + erf(v / Math.SQRT2));
        return this.operation({ id, name: kind, op: kind, inputs: [x.id], outputs: [id + '.out'], group: id }, [{ id: id + '.out', shape: x.shape, dtype: x.dtype, role: 'activation', data: { offset: 0, values: x.data!.values.map(v => v === null ? null : fn(v)) } }]);
    }
    build() { this.model.outputs = [this.current]; this.model.notes = ['声明式示例使用固定种子生成数值；不是训练后的参数。']; return validateModel(structuredClone(this.model)); }
}
function erf(x: number) { const sign = x < 0 ? -1 : 1, a = Math.abs(x), t = 1 / (1 + .3275911 * a); return sign * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-a * a)); }
export function defineModel(name: string, seed?: number) { return new ModelBuilder(name, seed); }
