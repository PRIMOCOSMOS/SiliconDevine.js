import { coordinates, flatIndex, numel, topologicalNodes, type Model, type Tensor } from './model.js';
export type Vec3 = [
    number,
    number,
    number
];
export interface TensorPlane {
    tensor: Tensor;
    indices: number[];
    positions: Vec3[];
    center: Vec3;
    width: number;
    depth: number;
    height: number;
    level: number;
    windowShape: number[];
    partial: boolean;
    origin: number[];
    owner?: string;
}
/** Exact coordinate windows, not arbitrary N cells repacked as a square. */
export function tensorWindow(t: Tensor, limit = 128, requestedOrigin: number[] = []): TensorPlane {
    const known = t.shape.every(d => typeof d === 'number'), shape = t.shape as number[];
    const visible = known ? shape.map(d => Math.min(d, 8)) : [];
    // Keep channels and batches explicit, while preserving a genuine depth axis.
    if (visible.length >= 4) {
        visible[0] = Math.min(visible[0], 1);
        visible[1] = Math.min(visible[1], 2);
    }
    if (visible.length === 2) {
        visible[0] = Math.min(shape[0], 32);
        visible[1] = Math.min(shape[1], 32);
    }
    if (visible.length === 1)
        visible[0] = Math.min(shape[0], 32);
    while (visible.reduce((a, b) => a * b, 1) > limit) {
        let axis = visible.indexOf(Math.max(...visible));
        visible[axis] = Math.ceil(visible[axis] / 2);
    }
    const origin = known ? shape.map((d, i) => Math.max(0, Math.min(d - visible[i], Math.floor(requestedOrigin[i] ?? 0)))) : [];
    const count = known ? visible.reduce((a, b) => a * b, 1) : 0, indices: number[] = [], positions: Vec3[] = [];
    const spacing = .38;
    for (let i = 0; i < count; i++) {
        const c = coordinates(i, visible);
        indices.push(flatIndex(c.map((v, j) => v + origin[j]), shape));
        let x = 0, y = 0, z = 0;
        if (c.length === 1)
            x = c[0] * spacing;
        else if (c.length === 2) {
            x = c[1] * spacing;
            z = c[0] * spacing;
        }
        else if (c.length === 3) {
            x = (c[2] + c[0] * (visible[2] + 2)) * spacing;
            z = c[1] * spacing;
        }
        else if (c.length >= 4) {
            const prefix = c.slice(0, -3), dims = visible.slice(0, -3), plane = flatIndex(prefix, dims);
            x = (c.at(-1)! + plane * (visible.at(-1)! + 2)) * spacing;
            z = c.at(-2)! * spacing;
            y = c.length >= 5 || t.spatialRank === 3 ? c.at(-3)! * spacing : 0;
            if (c.length === 4 && t.spatialRank !== 3)
                x += (c.at(-3)! * (visible.at(-1)! + 2)) * spacing;
        }
        positions.push([x, y, z]);
    }
    const max = (axis: number) => Math.max(0, ...positions.map(p => p[axis]));
    const w = max(0), h = max(1), d = max(2);
    positions.forEach(p => { p[0] -= w / 2; p[2] -= d / 2; });
    return { tensor: t, indices, positions, center: [0, 0, 0], width: w + .5, height: h + .28, depth: d + .5, level: 0, windowShape: visible, origin, partial: !known || count !== numel(t.shape) };
}
export function layoutModel(model: Model, cellLimit = 128, origins = new Map<string, number[]>()): Map<string, TensorPlane> {
    const planes = new Map(model.tensors.map(t => [t.id, tensorWindow(t, cellLimit, origins.get(t.id))])), levels = new Map<string, number>();
    model.inputs.forEach(id => levels.set(id, 0));
    for (const n of topologicalNodes(model)) {
        const level = 1 + Math.max(0, ...n.inputs.map(id => levels.get(id) ?? 0));
        for (const o of n.outputs) {
            levels.set(o, level);
            planes.get(o)!.owner = n.id;
        }
    }
    const rows = new Map<number, TensorPlane[]>();
    for (const p of planes.values())
        if (p.tensor.role === 'input' || p.tensor.role === 'activation') {
            p.level = levels.get(p.tensor.id) ?? 0;
            rows.set(p.level, [...rows.get(p.level) ?? [], p]);
        }
    let y = 0;
    const footprint = (p: TensorPlane) => {
        const owner = model.nodes.find(n => n.id === p.owner);
        return Math.max(p.width, 3.6, ...Object.values(owner?.parameters ?? {}).map(id => planes.get(id)?.width ?? 0));
    };
    for (const [level, row] of [...rows].sort((a, b) => a[0] - b[0])) {
        const width = row.reduce((s, p) => s + footprint(p) + 1.2, 0) - 1.2;
        let x = -width / 2;
        for (const p of row) {
            const w = footprint(p);
            p.center = [x + w / 2, y, 0];
            x += w + 1.2;
        }
        y -= Math.max(4.1, ...row.map(p => p.height + 3.4));
    }
    const placed = new Set<string>();
    for (const n of topologicalNodes(model)) {
        const output = planes.get(n.outputs[0])!, input = planes.get(n.inputs[0]);
        let slot = 0;
        for (const [key, id] of Object.entries(n.parameters ?? {})) {
            if (placed.has(id))
                continue;
            placed.add(id);
            const p = planes.get(id)!;
            // Bias stays beside its weight fabric, aligned with the trunk on X.
            const bias = key.includes('bias') || key.includes('beta');
            p.center = [output.center[0], input ? (input.center[1] + output.center[1]) / 2 : output.center[1] + 2, (bias ? 1 : -1) * (Math.max(output.depth, input?.depth ?? 0) / 2 + p.depth / 2 + .7 + slot * .45)];
            p.level = output.level;
            p.owner = n.id;
            slot++;
        }
    }
    // Constants and buffers used as explicit graph inputs retain an aligned lane.
    for (const p of planes.values())
        if (!levels.has(p.tensor.id) && !placed.has(p.tensor.id)) {
            const use = model.nodes.find(n => n.inputs.includes(p.tensor.id));
            if (use) {
                const target = planes.get(use.outputs[0])!;
                p.center = [target.center[0], target.center[1] + 2, -target.depth / 2 - p.depth / 2 - 1];
                p.owner = use.id;
            }
        }
    for (const p of planes.values())
        p.positions = p.positions.map(v => [v[0] + p.center[0], v[1] + p.center[1], v[2] + p.center[2]]);
    return planes;
}
