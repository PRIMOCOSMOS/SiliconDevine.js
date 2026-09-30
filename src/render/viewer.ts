import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createCrystalTensor, createArrowStream, type Position3 } from './crystalPrimitives';
import { numericPalette, valueExtent } from './numericPalette';
import { connectionFabric, type Connection } from './operationMotion';
import { layoutModel, type TensorPlane } from '../core/layout';
import { validateModel, valueAt, numel, topologicalNodes, type Model, type Operation, type Tensor } from '../core/model';
import { operatorVisual } from '../core/operators';
export interface ViewerOptions {
    maxCells?: number;
    cellsPerTensor?: number;
    maxConnections?: number;
    maxNodes?: number;
    pixelRatio?: number;
    labels?: 'auto' | 'all' | 'none';
    onSelect?: (node: Operation | undefined) => void;
    onStats?: (stats: ViewerStats) => void;
}
export interface ViewerStats {
    drawCalls: number;
    triangles: number;
    cells: number;
    connections: number;
    active: string;
    visibleNodes: number;
    totalNodes: number;
    pageStart: number;
    partialTensors: number;
    unknownTensors: number;
    paused: boolean;
}
type Fabric = ReturnType<typeof connectionFabric>;
interface Card {
    group: T.Group;
    owner?: string;
    id: string;
    priority: number;
}
/** Framework-independent viewer. No React, application context or global listeners survive dispose(). */
export class SiliconDevineViewer {
    readonly renderer: T.WebGLRenderer;
    readonly scene = new T.Scene();
    readonly camera = new T.PerspectiveCamera(42, 1, .05, 5000);
    readonly controls: OrbitControls;
    private root = new T.Group();
    private observer: ResizeObserver;
    private visibility: IntersectionObserver;
    private visible = true;
    private disposed = false;
    private frame = 0;
    private last = 0;
    private elapsed = 0;
    private playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
    private model?: Model;
    private view?: Model;
    private planes = new Map<string, TensorPlane>();
    private tensors = new Map<string, Tensor>();
    private nodes: Operation[] = [];
    private cards: Card[] = [];
    private fabrics = new Map<string, Fabric>();
    private edges = new Map<string, Connection[]>();
    private pageStart = 0;
    private selected?: string;
    private hovered?: string;
    private active = '';
    private cachedActive = '';
    private connectionCount = 0;
    private cells = 0;
    private labelMode: 'auto' | 'all' | 'none';
    private highlight?: ReturnType<typeof createCrystalTensor>;
    private receptive?: T.Box3Helper;
    private regions: {
        node: string;
        box: T.Box3;
    }[] = [];
    private positions = new Map<string, Map<number, Position3>>();
    private clock = 0;
    private pointerStart?: [
        number,
        number
    ];
    private ray = new T.Raycaster();
    private reduced: MediaQueryList;
    private activationCurves = new Map<string, {
        marker: T.Mesh;
        curve: (x: number) => number;
        x: number;
        y: number;
        z: number;
    }>();
    private arrows: ReturnType<typeof createArrowStream>[] = [];
    constructor(readonly container: HTMLElement, private options: ViewerOptions = {}) {
        this.labelMode = options.labels ?? 'auto';
        this.renderer = new T.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
        this.renderer.setClearColor('#050c12');
        this.renderer.setPixelRatio(Math.min(options.pixelRatio ?? 1.5, devicePixelRatio));
        this.renderer.outputColorSpace = T.SRGBColorSpace;
        container.append(this.renderer.domElement);
        this.renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;touch-action:none;';
        this.renderer.domElement.setAttribute('aria-label', '神经网络三维视图；拖动旋转，滚轮缩放，点击聚焦。可用旁侧索引选择模块。');
        this.scene.add(this.root, new T.HemisphereLight('#bcdff3', '#142335', 2));
        const key = new T.DirectionalLight('#e1f8ff', 3);
        key.position.set(8, 20, 15);
        this.scene.add(key);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = .1;
        this.controls.minDistance = .3;
        this.controls.maxDistance = 2000;
        this.controls.zoomToCursor = true;
        this.observer = new ResizeObserver(() => { const w = Math.max(1, container.clientWidth), h = Math.max(1, container.clientHeight); this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); });
        this.observer.observe(container);
        this.visibility = new IntersectionObserver(entries => { this.visible = entries[0].isIntersecting; });
        this.visibility.observe(container);
        this.reduced = matchMedia('(prefers-reduced-motion: reduce)');
        this.reduced.addEventListener('change', this.onReduced);
        const c = this.renderer.domElement;
        c.addEventListener('pointermove', this.onMove);
        c.addEventListener('pointerleave', this.onLeave);
        c.addEventListener('pointerdown', this.onDown);
        c.addEventListener('pointerup', this.onUp);
        c.addEventListener('webglcontextlost', this.onLost);
        c.addEventListener('webglcontextrestored', this.onRestored);
        this.frame = requestAnimationFrame(this.animate);
    }
    private onReduced = () => { if (this.reduced.matches)
        this.playing = false; };
    private onLost = (e: Event) => { e.preventDefault(); this.visible = false; };
    private onRestored = () => { this.visible = true; if (this.model)
        this.setPage(this.pageStart); };
    load(model: unknown) {
        const candidate = validateModel(model), previous = this.model;
        this.model = candidate;
        try { this.setPage(0); }
        catch (error) { this.model = previous; throw error; }
        return this;
    }
    setPage(start: number) {
        if (!this.model)
            return;
        const all = topologicalNodes(this.model), limit = Math.min(128, Math.max(1, this.options.maxNodes ?? 32));
        this.selected = undefined;
        this.hovered = undefined;
        this.pageStart = Math.max(0, Math.min(Math.max(0, all.length - 1), Math.floor(start)));
        const nodes = all.slice(this.pageStart, this.pageStart + limit), ids = new Set(nodes.flatMap(n => [...n.inputs, ...n.outputs, ...Object.values(n.parameters ?? {})]));
        if (!nodes.length)
            this.model.inputs.forEach(id => ids.add(id));
        if (ids.size > 512)
            throw Error('当前图段超过 512 个张量；请降低 maxNodes 或导出子模块。');
        const produced = new Set(nodes.flatMap(n => n.outputs));
        this.view = { ...this.model, nodes, tensors: this.model.tensors.filter(t => ids.has(t.id)), inputs: [...ids].filter(id => !produced.has(id) && this.model!.tensors.find(t => t.id === id)?.role !== 'parameter'), outputs: nodes.at(-1)?.outputs ?? this.model.outputs };
        this.build();
        this.fit();
    }
    setLabels(mode: 'auto' | 'all' | 'none') { this.labelMode = mode; }
    setPlaying(playing: boolean) { this.playing = playing; }
    get isPlaying() { return this.playing; }
    /** Select changes camera only on this explicit call; hover never moves the camera. */
    focus(id: string) { if (!this.model)
        return; const node = this.model.nodes.find(n => n.id === id); if (!node)
        return; if (!this.nodes.some(n => n.id === id)) {
        const i = topologicalNodes(this.model).findIndex(n => n.id === id);
        this.setPage(Math.floor(i / (this.options.maxNodes ?? 32)) * (this.options.maxNodes ?? 32));
    } this.selected = id; this.options.onSelect?.(node); const ids = [...node.inputs, ...node.outputs, ...Object.values(node.parameters ?? {})]; this.fit(ids); }
    clearFocus() { this.selected = undefined; this.options.onSelect?.(undefined); this.fit(); }
    zoom(factor: number) { const delta = this.camera.position.clone().sub(this.controls.target).multiplyScalar(factor); delta.clampLength(this.controls.minDistance, this.controls.maxDistance); this.camera.position.copy(this.controls.target).add(delta); this.controls.update(); }
    async fullscreen() { if (document.fullscreenElement === this.container)
        await document.exitFullscreen();
    else
        await this.container.requestFullscreen(); }
    fit(ids?: string[]) {
        const box = new T.Box3();
        for (const [id, p] of this.planes)
            if (!ids || ids.includes(id)) {
                box.expandByPoint(new T.Vector3(p.center[0] - p.width / 2, p.center[1] - 1, p.center[2] - p.depth / 2));
                box.expandByPoint(new T.Vector3(p.center[0] + p.width / 2, p.center[1] + p.height, p.center[2] + p.depth / 2 + 1));
            }
        if (box.isEmpty())
            box.setFromCenterAndSize(new T.Vector3(), new T.Vector3(5, 5, 5));
        const size = box.getSize(new T.Vector3()), center = box.getCenter(new T.Vector3());
        const vertical = Math.max(size.y, size.z * .7), horizontal = size.x / Math.max(.3, this.camera.aspect), distance = Math.max(vertical, horizontal, 4) / (2 * Math.tan(T.MathUtils.degToRad(this.camera.fov / 2))) * 1.3;
        this.controls.target.copy(center);
        this.camera.position.copy(center).add(new T.Vector3(.22, .22, 1).normalize().multiplyScalar(distance));
        this.controls.update();
    }
    private cleanup() {
        this.fabrics.forEach(f => f.dispose());
        this.fabrics.clear();
        this.edges.clear();
        const geometries = new Set<T.BufferGeometry>(), materials = new Set<T.Material>(), textures = new Set<T.Texture>();
        this.root.traverse(o => { const mesh = o as T.Mesh; if (mesh.geometry)
            geometries.add(mesh.geometry); if (mesh.material)
            for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
                materials.add(m);
                for (const v of Object.values(m))
                    if (v instanceof T.Texture)
                        textures.add(v);
            } if (o instanceof T.InstancedMesh)
            o.dispose(); });
        geometries.forEach(g => g.dispose());
        materials.forEach(m => m.dispose());
        textures.forEach(t => t.dispose());
        this.root.clear();
        this.cards = [];
        this.arrows = [];
        this.activationCurves.clear();
        this.regions = [];
        this.positions.clear();
        this.cachedActive = '';
        this.connectionCount = 0;
        this.highlight = undefined;
        this.receptive = undefined;
    }
    private build() {
        this.cleanup();
        const model = this.view!;
        this.nodes = topologicalNodes(model);
        this.tensors = new Map(model.tensors.map(t => [t.id, t]));
        const budget = Math.max(512, Math.min(16384, this.options.maxCells ?? 8192)), per = Math.max(1, Math.min(this.options.cellsPerTensor ?? 128, Math.floor(budget / Math.max(1, model.tensors.length))));
        this.planes = layoutModel(model, per);
        const pos: Position3[] = [], values: number[] = [];
        for (const p of this.planes.values()) {
            this.positions.set(p.tensor.id, new Map(p.indices.map((v, i) => [v, p.positions[i]])));
            const extent = p.tensor.stats?.absmax ?? valueExtent(p.tensor.data?.values.map(v => v ?? NaN) ?? []);
            pos.push(...p.positions);
            values.push(...p.indices.map(i => valueAt(p.tensor, i) / Math.max(extent, 1e-12)));
            const plate = new T.Mesh(new T.BoxGeometry(Math.max(p.width, .8), .018, Math.max(p.depth, .5)), new T.MeshBasicMaterial({ color: '#3c7187', transparent: true, opacity: .1, depthWrite: false }));
            plate.position.set(p.center[0], p.center[1] - .16, p.center[2]);
            this.root.add(plate);
            const outline = new T.LineSegments(new T.EdgesGeometry(plate.geometry), new T.LineBasicMaterial({ color: p.tensor.role === 'parameter' ? '#bc986c' : '#65b4c7', transparent: true, opacity: .48 }));
            outline.position.copy(plate.position);
            this.root.add(outline);
            this.addCard(p);
        }
        this.cells = pos.length;
        if (pos.length) {
            const crystal = createCrystalTensor(this.root, pos.length, .26, { valueEdges: true, valueScale: 1, bodyOpacity: .2, edgeOpacity: .5 });
            crystal.update(values, pos);
        }
        this.highlight = createCrystalTensor(this.root, 256, .27, { valueEdges: true, valueScale: 1 });
        this.arrows = Array.from({ length: 6 }, () => createArrowStream(this.root, '#d2edf5', .05, 1));
        this.receptive = new T.Box3Helper(new T.Box3(), new T.Color('#e5d596'));
        this.receptive.visible = false;
        this.root.add(this.receptive);
        // Disjoint vertical ownership slabs meet halfway between functional outputs.
        const nodePlanes = this.nodes.map(n => ({ n, p: this.planes.get(n.outputs[0])! })), ys = [...new Set(nodePlanes.map(o => o.p.center[1]))].sort((a, b) => b - a);
        const fullX = Math.max(2, ...[...this.planes.values()].map(p => Math.abs(p.center[0]) + p.width / 2));
        for (const { n, p } of nodePlanes) {
            const yi = ys.indexOf(p.center[1]), siblings = nodePlanes.filter(o => o.p.center[1] === p.center[1]).sort((a, b) => a.p.center[0] - b.p.center[0]), si = siblings.findIndex(o => o.n.id === n.id);
            const top = yi === 0 ? Math.max(...[...this.planes.values()].map(p => p.center[1] + p.height)) + 1 : (ys[yi - 1] + ys[yi]) / 2;
            const bottom = yi === ys.length - 1 ? ys[yi] - 2 : (ys[yi] + ys[yi + 1]) / 2;
            const left = si === 0 ? -fullX - 1 : (siblings[si - 1].p.center[0] + p.center[0]) / 2, right = si === siblings.length - 1 ? fullX + 1 : (siblings[si + 1].p.center[0] + p.center[0]) / 2;
            this.regions.push({ node: n.id, box: new T.Box3(new T.Vector3(left, bottom, -2), new T.Vector3(right, top, 2)) });
            const visual = operatorVisual(n.op);
            if (visual.curve) {
                const input = this.planes.get(n.inputs[0]);
                if (input)
                    this.addActivationCurve(n, input, p);
            }
        }
        // Small MLPs retain every weight line. Complex graphs instantiate only the active operator.
        const simple = this.nodes.length <= 12 && this.nodes.every(n => ['linear', 'relu', 'gelu', 'sigmoid', 'tanh', 'identity'].includes(n.op));
        if (simple)
            for (const n of this.nodes)
                this.createFabric(n);
        else {
            // Topology remains legible even when scalar computation is lazy.
            // One static draw preserves residuals and branches outside the active region.
            const backbone: Connection[] = [];
            for (const node of this.nodes) {
                const output = this.planes.get(node.outputs[0])!;
                for (const id of node.inputs) {
                    const input = this.planes.get(id);
                    if (input) {
                        const skip = output.level - input.level > 1;
                        const side = Math.max(input.width,output.width)/2 + 1;
                        backbone.push({from:input.center,to:output.center,via:skip?[[side,input.center[1],-.8],[side,output.center[1],-.8]]:undefined});
                    }
                }
            }
            connectionFabric(this.root,backbone).update(0,-1,-1,0,.55);
        }
    }
    private addCard(p: TensorPlane) {
        const canvas = document.createElement('canvas');
        canvas.width = 896;
        canvas.height = 192;
        const c = canvas.getContext('2d')!;
        const owner = this.nodes.find(n => n.id === p.owner), visual = operatorVisual(owner?.op ?? '');
        c.fillStyle = '#091d28';
        c.fillRect(0, 0, 896, 192);
        c.strokeStyle = owner ? visual.color : '#7ac8d7';
        c.lineWidth = 2;
        c.strokeRect(2, 2, 892, 188);
        c.lineWidth = 5;
        for (const x of [2, 850]) {
            c.beginPath();
            c.moveTo(x, 3);
            c.lineTo(x + 44, 3);
            c.stroke();
        }
        c.fillStyle = '#d9f3fa';
        c.font = '52px sans-serif';
        const name = p.tensor.role === 'parameter' ? `${p.tensor.source ?? p.tensor.id}` : owner ? `${visual.label} · ${owner.name}` : `输入 · ${p.tensor.id}`;
        c.fillText(name.length > 28 ? name.slice(0, 27) + '…' : name, 20, 60, 853);
        c.fillStyle = '#a8ccd7';
        c.font = '42px sans-serif';
        c.fillText(`[${p.tensor.shape.join(' × ')}]  ${p.tensor.dtype}`, 20, 118, 850);
        c.font = '30px sans-serif';
        c.fillStyle = owner ? visual.color : '#a8ccd7';
        const line = p.partial ? `坐标窗口 [${p.windowShape.join(' × ')}] / ${numel(p.tensor.shape) ?? '动态'} 元素` : owner ? visual.formula : p.tensor.data ? '已捕获数值' : '仅结构 · 数值未导出';
        c.fillText(line, 20, 167, 852);
        const texture = new T.CanvasTexture(canvas);
        texture.colorSpace = T.SRGBColorSpace;
        const width = Math.min(7.2, Math.max(5.6, p.width));
        const mesh = new T.Mesh(new T.PlaneGeometry(width, width * 192 / 896), new T.MeshBasicMaterial({ map: texture, transparent: true, opacity: .94, side: T.DoubleSide, depthWrite: false }));
        const group = new T.Group();
        mesh.position.y = -width * 192 / 896 / 2;
        group.add(mesh);
        group.position.set(p.center[0], p.center[1] - .18, p.center[2] + p.depth / 2 + .1);
        group.rotation.x = -Math.PI / 5;
        this.root.add(group);
        this.cards.push({ group, id: p.tensor.id, owner: p.owner, priority: p.tensor.role === 'parameter' ? 0 : 1 });
    }
    private addActivationCurve(n: Operation, input: TensorPlane, out: TensorPlane) {
        const fn = operatorVisual(n.op).curve!, curve = (x: number) => fn(x, n.attrs ?? {}), x = out.center[0], y = (input.center[1] + out.center[1]) / 2, z = out.center[2] + out.depth / 2 + .15;
        const points = Array.from({ length: 81 }, (_, i) => { const t = -2 + i / 20; return new T.Vector3(x + t * .45, y + curve(t) * .45, z); });
        this.root.add(new T.Line(new T.BufferGeometry().setFromPoints(points), new T.LineBasicMaterial({ color: '#b6a0e6', transparent: true, opacity: .8 })));
        const axes = new T.LineSegments(new T.BufferGeometry().setFromPoints([new T.Vector3(x - 1, y, z), new T.Vector3(x + 1, y, z), new T.Vector3(x, y - .6, z), new T.Vector3(x, y + 1, z)]), new T.LineBasicMaterial({ color: '#627790', transparent: true, opacity: .5 }));
        this.root.add(axes);
        const marker = new T.Mesh(new T.BoxGeometry(.13, .13, .13), new T.MeshBasicMaterial({ color: '#effaff' }));
        this.root.add(marker);
        this.activationCurves.set(n.id, { marker, curve, x, y, z });
    }
    private links(n: Operation): Connection[] {
        const visual = operatorVisual(n.op), result: Connection[] = [], out = this.planes.get(n.outputs[0])!;
        const cap = Math.max(64, this.options.maxConnections ?? 12000);
        for (let oi = 0; oi < out.indices.length; oi++) {
            const index = out.indices[oi], to = out.positions[oi];
            const deps = visual.dependencies(n, out.tensor, index, this.tensors);
            for (let term = 0; term < deps.length; term++) {
                const d = deps[term], from = this.positions.get(d.tensor)?.get(d.index);
                if (!from)
                    continue;
                const wt = d.parameter ? this.tensors.get(d.parameter) : undefined;
                const scale = wt?.stats?.absmax ?? valueExtent(wt?.data?.values.map(v => v ?? NaN) ?? []);
                result.push({ from, to, weight: Number.isFinite(d.weight) ? d.weight : undefined, weightScale: scale, output: oi, term });
                if (result.length >= cap)
                    return result;
            }
        }
        if (!result.length) {
            for (const id of n.inputs) {
                const from = this.planes.get(id);
                if (from)
                    result.push({ from: from.center, to: out.center });
            }
        }
        return result;
    }
    private createFabric(n: Operation) { const links = this.links(n); this.edges.set(n.id, links); const fabric = connectionFabric(this.root, links); this.fabrics.set(n.id, fabric); return fabric; }
    private setActive(id: string) {
        if (id === this.cachedActive)
            return;
        this.cachedActive = id;
        const n = this.nodes.find(n => n.id === id);
        if (!n)
            return;
        if (!this.fabrics.has(id)) {
            // At most 12 small fabrics, or one complex active region; no monotonically growing cache.
            if (this.fabrics.size >= 12 || this.nodes.some(n => !['linear', 'relu', 'gelu', 'sigmoid', 'tanh', 'identity'].includes(n.op))) {
                this.fabrics.forEach(f => f.dispose());
                this.fabrics.clear();
                this.edges.clear();
            }
            this.createFabric(n);
        }
    }
    private onMove = (e: PointerEvent) => {
        if (e.buttons)
            return;
        const r = this.renderer.domElement.getBoundingClientRect();
        this.ray.setFromCamera(new T.Vector2((e.clientX - r.left) / r.width * 2 - 1, 1 - (e.clientY - r.top) / r.height * 2), this.camera);
        let distance = Infinity, hover: string | undefined;
        const point = new T.Vector3();
        for (const region of this.regions) {
            const hit = this.ray.ray.intersectBox(region.box, point);
            if (hit) {
                const d = hit.distanceTo(this.ray.ray.origin);
                if (d < distance) {
                    distance = d;
                    hover = region.node;
                }
            }
        }
        this.hovered = hover;
        this.renderer.domElement.style.cursor = hover ? 'pointer' : 'grab';
    };
    private onLeave = () => { this.hovered = undefined; };
    private onDown = (e: PointerEvent) => { this.pointerStart = [e.clientX, e.clientY]; };
    private onUp = (e: PointerEvent) => { if (this.pointerStart && Math.hypot(e.clientX - this.pointerStart[0], e.clientY - this.pointerStart[1]) < 4 && this.hovered)
        this.focus(this.hovered); this.pointerStart = undefined; };
    private animate = (time: number) => {
        if (this.disposed)
            return;
        this.frame = requestAnimationFrame(this.animate);
        const dt = Math.min(.05, (time - this.last) / 1000 || 0);
        this.last = time;
        if (!this.visible || document.hidden)
            return;
        this.controls.update();
        if (this.playing)
            this.elapsed += dt;
        this.clock += dt;
        const auto = this.nodes[Math.floor(this.elapsed / 3.6) % Math.max(1, this.nodes.length)];
        this.active = this.hovered ?? this.selected ?? auto?.id ?? '';
        this.setActive(this.active);
        const active = this.nodes.find(n => n.id === this.active), out = active ? this.planes.get(active.outputs[0]) : undefined;
        const phase = (this.elapsed * .4) % 1;
        for (const [id, f] of this.fabrics) {
            const n = this.nodes.find(n => n.id === id)!, p = this.planes.get(n.outputs[0])!;
            f.update(phase, phase * p.indices.length, phase * 8, id === this.active ? 1 : .12, n.op.startsWith('conv') ? .12 : id === this.active ? 1 : .65);
        }
        this.connectionCount = [...this.edges.values()].reduce((s, e) => s + e.length, 0);
        const flowing = this.edges.get(this.active) ?? [], outputIndex = out ? Math.floor(phase * out.indices.length) : 0, current = flowing.filter(e => e.output === outputIndex);
        this.arrows.forEach((arrow, i) => { const edge = current[i]; arrow.group.visible = !!edge; if (edge) {
            const middle: Position3 = edge.from.map((v, j) => (v + edge.to[j]) / 2) as Position3;
            arrow.update([edge.from, middle, edge.to], phase + i * .13, 1);
        } });
        if (active && out && this.highlight) {
            const oi = Math.min(out.indices.length - 1, Math.floor(phase * out.indices.length)), index = out.indices[oi];
            const ps: Position3[] = [], vs: number[] = [];
            const push = (id: string, i: number) => { const p = this.positions.get(id)?.get(i), t = this.tensors.get(id); if (p && t && ps.length < 256) {
                ps.push(p);
                vs.push(valueAt(t, i) / Math.max(t.stats?.absmax ?? valueExtent(t.data?.values.map(v => v ?? NaN) ?? []), 1e-12));
            } };
            if (index !== undefined) {
                push(out.tensor.id, index);
                const deps = operatorVisual(active.op).dependencies(active, out.tensor, index, this.tensors), rf = new T.Box3();
                for (const d of deps) {
                    push(d.tensor, d.index);
                    const p = this.positions.get(d.tensor)?.get(d.index);
                    if (p && active.op.startsWith('conv') && d.tensor === active.inputs[0])
                        rf.expandByPoint(new T.Vector3(...p));
                    if (d.parameter && d.parameterIndex !== undefined)
                        push(d.parameter, d.parameterIndex);
                }
                if (this.receptive) {
                    this.receptive.visible = !rf.isEmpty();
                    if (!rf.isEmpty())
                        this.receptive.box.copy(rf.expandByScalar(.2));
                }
            }
            this.highlight.update(vs, ps, { active: ps.map((_, i) => i), focus: 0 });
            this.highlight.body.visible = this.highlight.edge.visible = false;
        }
        for (const [id, v] of this.activationCurves) {
            const x = -2 + phase * 4;
            v.marker.position.set(v.x + x * .45, v.y + v.curve(x) * .45, v.z);
            v.marker.visible = id === this.active;
        }
        let count = 0;
        const shown: T.Box2[] = [];
        for (const card of [...this.cards].sort((a, b) => Number(b.owner === this.active) - Number(a.owner === this.active) || b.priority - a.priority)) {
            const wanted = this.labelMode === 'all' || this.labelMode === 'auto' && (card.owner === this.active || !this.hovered && card.priority > 0 && count < 5);
            card.group.visible = this.labelMode !== 'none' && wanted;
            if (card.group.visible) {
                const p = card.group.position.clone().project(this.camera), w = card.group.children[0] as T.Mesh;
                const b = new T.Box3().setFromObject(w), corners = [new T.Vector3(b.min.x, b.min.y, b.min.z), new T.Vector3(b.max.x, b.max.y, b.max.z)].map(v => v.project(this.camera));
                const rect = new T.Box2().setFromPoints(corners.map(v => new T.Vector2(v.x, v.y)));
                if (this.labelMode === 'auto' && (p.z > 1 || p.z < -1 || shown.some(r => r.intersectsBox(rect))))
                    card.group.visible = false;
                else {
                    shown.push(rect);
                    count++;
                }
            }
        }
        this.renderer.render(this.scene, this.camera);
        if (this.clock > .25) {
            this.clock = 0;
            this.options.onStats?.(this.getStats());
        }
    };
    getStats(): ViewerStats { return { drawCalls: this.renderer.info.render.calls, triangles: this.renderer.info.render.triangles, cells: this.cells, connections: this.connectionCount, active: this.active, visibleNodes: this.nodes.length, totalNodes: this.model?.nodes.length ?? 0, pageStart: this.pageStart, partialTensors: [...this.planes.values()].filter(p => p.partial).length, unknownTensors: [...this.planes.values()].filter(p => !p.tensor.data).length, paused: !this.playing }; }
    dispose() { if (this.disposed)
        return; this.disposed = true; cancelAnimationFrame(this.frame); this.observer.disconnect(); this.visibility.disconnect(); this.reduced.removeEventListener('change', this.onReduced); const c = this.renderer.domElement; c.removeEventListener('pointermove', this.onMove); c.removeEventListener('pointerleave', this.onLeave); c.removeEventListener('pointerdown', this.onDown); c.removeEventListener('pointerup', this.onUp); c.removeEventListener('webglcontextlost', this.onLost); c.removeEventListener('webglcontextrestored', this.onRestored); this.controls.dispose(); this.cleanup(); this.renderer.dispose(); c.remove(); }
}
