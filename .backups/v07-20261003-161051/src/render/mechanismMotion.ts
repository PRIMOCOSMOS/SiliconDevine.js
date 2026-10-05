import * as T from 'three';
import { valueAt, type Operation, type Tensor } from '../core/model.js';
import type { TensorPlane } from '../core/layout.js';
import { operatorVisual } from '../core/operators.js';
import { numericColor } from './numericPalette.js';
export interface MechanismState {
    kind: string;
    row: number;
    values: number[];
    mean?: number;
    rms?: number;
    sum?: number;
    visibleItems: number;
}
/** Reusable instanced motion, anchored to actual tensor coordinates. */
export class MechanismMotion {
    private cubes: T.InstancedMesh;
    private bars: T.InstancedMesh;
    private dummy = new T.Object3D();
    private color = new T.Color();
    private cached?: {
        key: string;
        mean: number;
        rms: number;
        values: number[];
    };
    state?: MechanismState;
    constructor(parent: T.Group) {
        this.cubes = new T.InstancedMesh(new T.BoxGeometry(.22, .22, .22), new T.MeshBasicMaterial({ transparent: true, opacity: .82, depthWrite: false }), 64);
        this.bars = new T.InstancedMesh(new T.BoxGeometry(.11, 1, .11), new T.MeshBasicMaterial({ transparent: true, opacity: .76, depthWrite: false }), 64);
        this.cubes.frustumCulled = this.bars.frustumCulled = false;
        parent.add(this.cubes, this.bars);
        this.hide();
    }
    private hide() { this.cubes.count = this.bars.count = 0; this.state = undefined; }
    update(node: Operation | undefined, out: TensorPlane | undefined, index: number, phase: number, planes: Map<string, TensorPlane>, tensors: Map<string, Tensor>) {
        this.hide();
        if (!node || !out || index === undefined)
            return;
        const op = node.op, columns = Number(out.tensor.shape.at(-1) ?? 1), row = Math.floor(index / columns);
        const items = out.indices.map((i, j) => ({ i, p: out.positions[j] })).filter(v => Math.floor(v.i / columns) === row).slice(0, 24);
        const source = planes.get(node.inputs[0]), smooth = phase * phase * (3 - 2 * phase);
        const put = (mesh: T.InstancedMesh, position: number[], value: number, extent: number, height?: number) => {
            if (mesh.count >= 64)
                return;
            this.dummy.position.set(position[0], position[1], position[2]);
            this.dummy.scale.set(1, height ?? 1, 1);
            this.dummy.updateMatrix();
            mesh.setMatrixAt(mesh.count, this.dummy.matrix);
            mesh.setColorAt(mesh.count, numericColor(value, extent, this.color));
            mesh.count++;
        };
        if (op === 'softmax') {
            const rank = out.tensor.shape.length, dim = Number(node.attrs?.dim ?? -1);
            if ((dim + rank) % rank !== rank - 1)
                return;
            const values = items.map(v => valueAt(out.tensor, v.i));
            items.forEach((item, j) => {
                const p = values[j];
                if (Number.isFinite(p) && p > 0) {
                    const height = p * 2.8;
                    put(this.bars, [item.p[0], item.p[1] + .18 + height / 2, item.p[2]], p, 1, height);
                }
            });
            const tensor = tensors.get(node.outputs[0])!, base = row * columns, full = Array.from({ length: columns }, (_, j) => valueAt(tensor, base + j));
            this.state = { kind: 'probability', row, values, sum: full.every(Number.isFinite) ? full.reduce((a, b) => a + b, 0) : undefined, visibleItems: this.bars.count };
        }
        else if (['layernorm', 'rmsnorm'].includes(op) && source) {
            const x = tensors.get(node.inputs[0])!, size = ((node.attrs?.normalized_shape ?? [columns]) as number[]).reduce((a, b) => a * b, 1), base = Math.floor(index / size) * size, key = `${node.id}/${base}`;
            if (this.cached?.key !== key) {
                const values = Array.from({ length: size }, (_, j) => valueAt(x, base + j));
                this.cached = { key, values, mean: values.reduce((a, b) => a + b, 0) / size, rms: Math.sqrt(values.reduce((a, b) => a + b * b, 0) / size) };
            }
            const extent = Math.max(1e-8, x.stats?.absmax ?? 1, out.tensor.stats?.absmax ?? 1), mean = this.cached.mean, rms = this.cached.rms;
            for (const item of items) {
                const j = source.indices.indexOf(item.i), a = valueAt(x, item.i), b = valueAt(out.tensor, item.i);
                if (j < 0 || !Number.isFinite(a) || !Number.isFinite(b))
                    continue;
                const p = source.positions[j].map((v, k) => v + (item.p[k] - v) * smooth);
                put(this.cubes, p, a + (b - a) * smooth, extent);
            }
            this.state = { kind: op, row, values: items.map(v => valueAt(out.tensor, v.i)), mean, rms, visibleItems: this.cubes.count };
        }
        else if (['add', 'multiply', 'subtract'].includes(op) && node.inputs.length === 2) {
            const result = items.map(v => valueAt(out.tensor, v.i)), extent = Math.max(1e-8, ...node.inputs.map(id => tensors.get(id)?.stats?.absmax ?? 1), out.tensor.stats?.absmax ?? 1);
            for (const item of items) {
                const y = valueAt(out.tensor, item.i);
                if (!Number.isFinite(y))
                    continue;
                const deps = operatorVisual(op).dependencies(node, out.tensor, item.i, tensors);
                if (phase > .78) {
                    put(this.cubes, item.p, y, extent);
                    continue;
                }
                for (const d of deps) {
                    const p = planes.get(d.tensor), j = p?.indices.indexOf(d.index) ?? -1, value = valueAt(tensors.get(d.tensor)!, d.index);
                    if (!p || j < 0 || !Number.isFinite(value))
                        continue;
                    const travel = Math.min(1, phase / .78), t = travel * travel * (3 - 2 * travel);
                    let position = p.positions[j].map((v, k) => v + (item.p[k] - v) * t);
                    if (node.attrs?.residualInput === d.tensor) {
                        const left = Math.min(...[...planes.values()].filter(v => v.level >= p.level && v.level <= out.level).map(v => v.center[0] - v.width / 2)) - .7, z = Math.max(p.depth, out.depth) / 2 + .35;
                        const curve = new T.CatmullRomCurve3([new T.Vector3(...p.positions[j]), new T.Vector3(left, p.center[1] - .5, z), new T.Vector3(left, out.center[1] + .5, z), new T.Vector3(...item.p)]);
                        position = curve.getPoint(t).toArray();
                    }
                    put(this.cubes, position, value, extent);
                }
            }
            this.state = { kind: op, row, values: result, visibleItems: this.cubes.count };
        }
        for (const mesh of [this.cubes, this.bars]) {
            mesh.instanceMatrix.needsUpdate = true;
            if (mesh.instanceColor)
                mesh.instanceColor.needsUpdate = true;
        }
    }
}
