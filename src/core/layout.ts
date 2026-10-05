import { coordinates, flatIndex, numel, topologicalNodes, type Model, type Tensor } from './model.js';
export type Vec3 = [
    number,
    number,
    number
];
/** Each structural tile is a rectangular logical partition, never a sampled value. */
export function aggregatePartition(t:Tensor, tile:number, limit=18): (number[]|string)[] {
    const [rows,cols]=aggregateGrid(t,limit);
    if(!Number.isInteger(tile)||tile<0||tile>=rows*cols)throw Error('逻辑分区索引越界。');
    return t.shape.map((d,i)=>{const last=i===t.shape.length-1,penultimate=i===t.shape.length-2;
        const part=last?tile%cols:penultimate?Math.floor(tile/cols):0,total=last?cols:penultimate?rows:1;
        return typeof d==='number'?[Math.floor(d*part/total),Math.floor(d*(part+1)/total)]:`${d} × [${part}/${total}, ${part+1}/${total})`;
    });
}
function aggregateGrid(t:Tensor,limit:number):[number,number]{
    const budget=Math.max(1,Math.floor(limit)),cols=Math.min(6,budget,typeof t.shape.at(-1)==='number'?Math.max(1,Number(t.shape.at(-1))):6);
    const rows=t.shape.length<2?1:Math.min(3,Math.floor(budget/cols),typeof t.shape.at(-2)==='number'?Math.max(1,Number(t.shape.at(-2))):3);
    return [rows,cols];
}
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
    if(t.representation==='aggregate') {
        // A fixed grid of partitions, labelled as such. Symbolic axes remain symbolic.
        const [rows,cols]=aggregateGrid(t,limit);
        const positions:Vec3[]=Array.from({length:cols*rows},(_,i)=>[(i%cols-(cols-1)/2)*.76,0,(Math.floor(i/cols)-(rows-1)/2)*.76]);
        return {tensor:t,indices:positions.map((_,i)=>i),positions,center:[0,0,0],width:Math.max(1,cols*.76),depth:Math.max(.5,rows*.76),height:.3,level:0,windowShape:[rows,cols],partial:true,origin:[]};
    }
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
    const ordered=topologicalNodes(model), attention=ordered.some(n=>n.attrs?.attentionRole), lanes=new Map<string,number>();
    model.inputs.forEach(id => levels.set(id, 0));
    for (const n of ordered) {
        const level = 1 + Math.max(0, ...[...n.inputs,...Object.values(n.parameters??{})].map(id => levels.get(id) ?? 0));
        const path=n.group??n.name, roles=(n.attrs?.projectionRoles as string[]??[]).join('');
        let lane=n.inputs.length===1?lanes.get(n.inputs[0])??0:0;
        if(n.op==='linear'&&attention) lane=/q_.*proj|q_proj/.test(path)||roles==='Q'?-1:/kv_.*proj/.test(path)?1:/k_proj/.test(path)||roles==='K'?0:/v_proj/.test(path)||roles==='V'?1:0;
        if(n.attrs?.attentionRole || (n.op==='layout'&&String(n.name).includes('合并多头')))lane=0;
        for (const o of n.outputs) {
            levels.set(o, level);
            planes.get(o)!.owner = n.id;
            lanes.set(o,lane);
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
    const laneWidth=Math.max(4.8,...[...planes.values()].filter(p=>p.tensor.role!=='parameter').map(p=>p.width+1.2));
    for (const [level, row] of [...rows].sort((a, b) => a[0] - b[0])) {
        const width = row.reduce((s, p) => s + footprint(p) + 1.2, 0) - 1.2;
        let x = -width / 2;
        for (const p of row) {
            const w = footprint(p);
            p.center = [x + w / 2, y, 0];
            x += w + 1.2;
        }
        // Stable Q/K/V lanes converge on a single vertical attention/output trunk.
        if(attention) {
            const byLane=new Map<number,TensorPlane[]>();
            for(const p of row){const lane=lanes.get(p.tensor.id)??0;byLane.set(lane,[...byLane.get(lane)??[],p]);}
            for(const [lane,peers] of byLane){let z=-peers.reduce((s,p)=>s+p.depth+.7,0)/2;
                for(const p of peers){p.center=[lane*laneWidth,y,z+p.depth/2];z+=p.depth+.7;}}
        }
        y -= Math.max(model.architecture?6.2:attention?2.15:4.1, ...row.map(p => p.height + (attention?1.8:3.4)));
    }
    // A scalar/elementwise operation is one work surface. Its operands are arranged
    // around the output on that plane; convolution and learned projections retain depth.
    for(const n of ordered){
        const dot=n.op==='matmul'&&n.inputs.every(id=>planes.get(id)?.tensor.shape.length===1);
        if((!['add','multiply','divide','subtract'].includes(n.op)&&!dot)||n.attrs?.residualInput)continue;
        const out=planes.get(n.outputs[0])!;
        const operands=n.inputs.map(id=>planes.get(id)!).filter(p=>p&&ordered.filter(q=>q.inputs.includes(p.tensor.id)).length===1);
        let z=-out.depth/2-.8;
        for(const p of operands){p.center=[out.center[0],out.center[1],z-p.depth/2];p.level=out.level;z-=p.depth+.8;}
    }
    const placed = new Set<string>();
    for (const n of topologicalNodes(model)) {
        const output = planes.get(n.outputs[0])!, input = planes.get(n.inputs[0]);
        let slot = 0;
        for (const [key, id] of Object.entries(n.parameters ?? {})) {
            if (placed.has(id) || levels.has(id))
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
                const peers=[...planes.values()].filter(q=>!levels.has(q.tensor.id)&&!placed.has(q.tensor.id)&&use.inputs.includes(q.tensor.id));
                const offset=peers.slice(0,peers.indexOf(p)).reduce((sum,q)=>sum+q.width+.6,0);
                const total=peers.reduce((sum,q)=>sum+q.width+.6,0)-.6;
                p.center = [target.center[0]-total/2+offset+p.width/2, target.center[1] + 2, -target.depth / 2 - p.depth / 2 - 1];
                p.owner = use.id;
            }
        }
    for (const p of planes.values())
        p.positions = p.positions.map(v => [v[0] + p.center[0], v[1] + p.center[1], v[2] + p.center[2]]);
    return planes;
}
