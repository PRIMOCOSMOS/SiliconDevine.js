import * as T from 'three';
import { coordinates, valueAt, type Operation, type Tensor } from '../core/model.js';
import type { TensorPlane } from '../core/layout.js';
import { operatorVisual } from '../core/operators.js';
import { numericColor } from './numericPalette.js';
/** Exact sampled receptive-field coordinates, separated by channel, not one box across empty lanes. */
export class ReceptiveFieldMotion {
    readonly cells: T.InstancedMesh;
    readonly shells: T.InstancedMesh;
    state?: {
        output: number[];
        visibleSamples: number;
        totalSamples: number;
        channels: number;
        kernel: number[];
        stride: unknown;
        dilation: unknown;
    };
    private dummy = new T.Object3D();
    constructor(parent: T.Group) { this.cells = new T.InstancedMesh(new T.BoxGeometry(.31, .31, .31), new T.MeshBasicMaterial({ color: '#f9d58f', transparent: true, opacity: .44, depthWrite: false }), 512); this.shells = new T.InstancedMesh(new T.BoxGeometry(1, 1, 1), new T.MeshBasicMaterial({ color: '#a2ebf3', transparent: true, opacity: .10, depthWrite: false }), 32); this.cells.frustumCulled = this.shells.frustumCulled = false; parent.add(this.cells, this.shells); this.cells.count = this.shells.count = 0; }
    update(n: Operation | undefined, out: TensorPlane | undefined, index: number, planes: Map<string, TensorPlane>, tensors: Map<string, Tensor>) {
        this.cells.count = this.shells.count = 0;
        this.state = undefined;
        if (!n || (!n.op.startsWith('conv') && !['dynamic_conv2d','interpolate','grid_sample2d','unfold2d'].includes(n.op)) || !out)
            return;
        const x = planes.get(n.inputs[0]);
        if (!x)
            return;
        const deps = operatorVisual(n.op).dependencies(n, out.tensor, index, tensors).filter(d => d.tensor === x.tensor.id), groups = new Map<string, T.Box3>(), rank = x.tensor.spatialRank ?? (n.op==='interpolate'?x.tensor.shape.length-2:Number(n.op.match(/[123]/)?.[0] ?? 2)), spatial = x.tensor.shape.length - rank;
        for (const d of deps) {
            const j = x.indices.indexOf(d.index);
            if (j < 0 || this.cells.count >= 512)
                continue;
            const p = x.positions[j], key = coordinates(d.index, x.tensor.shape as number[]).slice(0, spatial).join('/');
            const box = groups.get(key) ?? new T.Box3();
            box.expandByPoint(new T.Vector3(...p));
            groups.set(key, box);
            this.dummy.position.set(...p);
            this.dummy.scale.setScalar(1);
            this.dummy.updateMatrix();
            this.cells.setMatrixAt(this.cells.count++, this.dummy.matrix);
        }
        for (const box of groups.values()) {
            if (this.shells.count >= 32)
                break;
            box.expandByScalar(.19);
            box.getCenter(this.dummy.position);
            box.getSize(this.dummy.scale);
            this.dummy.updateMatrix();
            this.shells.setMatrixAt(this.shells.count++, this.dummy.matrix);
        }
        this.cells.instanceMatrix.needsUpdate = this.shells.instanceMatrix.needsUpdate = true;
        this.state = { output: coordinates(index, out.tensor.shape as number[]), visibleSamples: this.cells.count, totalSamples: deps.length, channels: groups.size, kernel: (tensors.get(n.parameters?.weight ?? '')?.shape.slice(n.op==='dynamic_conv2d'?3:2) ?? []) as number[], stride: n.attrs?.stride ?? 1, dilation: n.attrs?.dilation ?? 1 };
    }
}
/** Each flow item is p(query,key) times a real V(key,channel), arriving in the corresponding output token. */
export class AttentionTokenMotion {
    private cubes: T.InstancedMesh;
    private lines: T.InstancedMesh;
    private dummy = new T.Object3D();
    private color = new T.Color();
    state?: {
        query: number;
        headIndex: number;
        probabilities: number[];
        output: number[];
        contributors: number;
        source: string;
        destination: string;
    };
    constructor(parent: T.Group) { this.cubes = new T.InstancedMesh(new T.BoxGeometry(.25, .25, .25), new T.MeshBasicMaterial({ transparent: true, opacity: .9, depthWrite: false }), 256); this.lines = new T.InstancedMesh(new T.CylinderGeometry(1, 1, 1, 5), new T.MeshBasicMaterial({ color: '#9bcbdc', transparent: true, opacity: .3, depthWrite: false }), 256); this.cubes.frustumCulled = this.lines.frustumCulled = false; parent.add(this.cubes, this.lines); this.cubes.count = this.lines.count = 0; }
    update(n: Operation | undefined, out: TensorPlane | undefined, index: number, phase: number, planes: Map<string, TensorPlane>, tensors: Map<string, Tensor>) {
        this.cubes.count = this.lines.count = 0;
        this.state = undefined;
        const unit = n?.attrs?.attention as Record<string, string> | undefined;
        if (!unit || !out || n?.attrs?.attentionRole !== 'context')
            return;
        const p = tensors.get(unit.probability)!, v = tensors.get(unit.value)!, y = tensors.get(unit.context)!, vp = planes.get(v.id), yp = planes.get(y.id);
        if (!vp || !yp)
            return;
        const keys = Number(p.shape.at(-1)), queries = Number(p.shape.at(-2)), width = Number(y.shape.at(-1)), query = Math.floor(index / width) % queries, headIndex = Math.floor(index / (width * queries));
        const probabilities = Array.from({ length: keys }, (_, k) => valueAt(p, (headIndex * queries + query) * keys + k));
        let contributors = 0;
        const columns = yp.indices.filter(i => Math.floor(i / width) === headIndex * queries + query);
        for (const yi of columns) {
            const to = yp.positions[yp.indices.indexOf(yi)], c = yi % width;
            for (let k = 0; k < Math.min(keys, 32); k++) {
                const weight = probabilities[k], vi = ((headIndex % Math.max(1,Number(v.shape.slice(0,-2).reduce<number>((a,b)=>a*Number(b),1)))) * keys + k) * width + c, j = vp.indices.indexOf(vi);
                if (j < 0 || !Number.isFinite(weight) || weight <= 0 || this.cubes.count >= 256)
                    continue;
                const from = vp.positions[j], a = new T.Vector3(...from), b = new T.Vector3(...to), delta = b.clone().sub(a), length = delta.length();
                this.dummy.position.copy(a).add(b).multiplyScalar(.5);
                this.dummy.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), delta.normalize());
                this.dummy.scale.set(.008 + .025 * weight, length, .008 + .025 * weight);
                this.dummy.updateMatrix();
                this.lines.setMatrixAt(this.lines.count++, this.dummy.matrix);
                const t = (phase + k / Math.max(1, keys) * .4) % 1, value = valueAt(v, vi) * weight;
                if (!Number.isFinite(value))
                    continue;
                this.dummy.position.copy(a).lerp(b, t);
                this.dummy.quaternion.identity();
                this.dummy.scale.setScalar(.7 + .3 * Math.sqrt(weight));
                this.dummy.updateMatrix();
                this.cubes.setMatrixAt(this.cubes.count, this.dummy.matrix);
                this.cubes.setColorAt(this.cubes.count++, numericColor(value, Math.max(1e-8, y.stats?.absmax ?? 1), this.color));
                contributors++;
            }
        }
        this.cubes.instanceMatrix.needsUpdate = this.lines.instanceMatrix.needsUpdate = true;
        if (this.cubes.instanceColor)
            this.cubes.instanceColor.needsUpdate = true;
        this.state = { query, headIndex, probabilities, output: columns.map(i => valueAt(y, i)), contributors, source: v.id, destination: y.id };
    }
}
