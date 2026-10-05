import { FunctionPlot } from './functionPlot.js';
import { structureSummary } from '../core/structureSummary.js';
import { StructureMotion, structureColors } from './structureMotion.js';
import { GaussianMotion } from './gaussianMotion.js';
import * as T from 'three';
import { moduleView } from '../core/hierarchy.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createCrystalTensor, createArrowStream, type Position3 } from './crystalPrimitives.js';
import { numericPalette, numericColor, valueExtent } from './numericPalette.js';
import { connectionFabric, type Connection } from './operationMotion.js';
import { layoutModel, type TensorPlane } from '../core/layout.js';
import { validateModel, valueAt, numel, coordinates, topologicalNodes, type Model, type Operation, type Tensor } from '../core/model.js';
import { operatorVisual } from '../core/operators.js';
import { MechanismMotion, type MechanismState } from './mechanismMotion.js';
import { ReceptiveFieldMotion, AttentionTokenMotion } from './contextualMotion.js';
import type { Dependency } from '../core/operators.js';
export interface ViewerOptions {
    transparentBackground?: boolean;
    maxCells?: number;
    cellsPerTensor?: number;
    maxConnections?: number;
    maxNodes?: number;
    pixelRatio?: number;
    labels?: 'auto' | 'all' | 'none';
    onSelect?: (node: Operation | undefined) => void;
    onStats?: (stats: ViewerStats) => void;
    dataProvider?: (id: string, indices: number[], signal: AbortSignal) => Promise<(number | null)[]>;
}
export interface ViewerStats {
    receptiveField?: ReceptiveFieldMotion["state"];
    attention?: AttentionTokenMotion["state"];
    gaussian?: GaussianMotion["state"];
    residuals?: number;
    atomic?: {node:string;atoms:number;cells:number;connections:number;basis:string};
    mechanism?: MechanismState;
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
    weight?: {
        tensor: string;
        index: number;
        coordinates: number[];
        value: number;
    };
    sample?: {
        input: number;
        output: number;
        index: number;
    };
}
type Fabric = ReturnType<typeof connectionFabric>;
interface Card {
    group: T.Group;
    anchor: T.Vector3;
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
    private planeOutlines=new Map<string,T.LineSegments>();
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
    private activationCurves = new Map<string, FunctionPlot>();
    private arrows: ReturnType<typeof createArrowStream>[] = [];
    private origins = new Map<string, number[]>();
    private windowRequest?: AbortController;
    private hoveredWeight?: {
        tensor: string;
        index: number;
    };
    private weightState?: ViewerStats['weight'];
    private sampleState?: ViewerStats['sample'];
    private generation = 0;
    private moduleScope = '';
    private structureScope = '';
    private mechanismKey = '';
    private mechanismStage=0;
    private tour=false;
    private tourClock=0;
    get isTouring(){return this.tour;}
    setTour(enabled:boolean){this.tour=enabled&&!this.reduced.matches;this.tourClock=0;this.setPlaying(true);}
    get computationStages(){return this.isMechanism?this.getDisplayedModel()?.stages??[]:[];}
    setComputationStage(index:number){if(!Number.isInteger(index)||index< -1||index>=this.computationStages.length)return;this.mechanismStage=index;this.tourClock=0;this.setPage(0);}

    private cameraFlight?: {from:T.Vector3; to:T.Vector3; targetFrom:T.Vector3; targetTo:T.Vector3; time:number};
    private stopFlight = () => { this.cameraFlight=undefined;this.tour=false; };
    get isMechanism(){return !!this.mechanismKey;}
    get isSourceExecution(){return this.isMechanism&&this.getDisplayedModel()?.producer?.backend==='pytorch-execution';}
    get mechanismAvailable(){return !!this.model?.architecture?.scopes[this.structureScope]?.mechanismRef;}
    showMechanism(key?:string, stage=-1){const a=this.model?.architecture;if(!a)return;const ref=key??a.scopes[this.structureScope]?.mechanismRef;if(!ref||!a.mechanisms?.[ref])return;this.mechanismKey=ref;this.mechanismStage=Number.isInteger(stage)&&stage>=-1&&stage<(a.mechanisms[ref].stages?.length??1)?stage:-1;this.setPage(0);this.options.onSelect?.(undefined);}
    showStructure(){if(this.isSourceExecution){this.structureScope=this.model!.architecture!.entry;this.structureHistory=[];}this.tour=false;this.mechanismKey='';this.moduleScope='';this.functionScope=undefined;this.setPage(0);this.options.onSelect?.(undefined);}
    private structureHistory: string[] = [];
    private structureMotion?: StructureMotion;
    getDisplayedModel(): Model | undefined {
        const m=this.model;if(!m?.architecture)return m;
        if(this.mechanismKey)return m.architecture.mechanisms?.[this.mechanismKey];
        return {...m,...m.architecture.scopes[this.structureScope || m.architecture.entry]};
    }
    private scopedCount = 0;
    private functionScope?: {
        name: string;
        nodes: Operation[];
    };
    private mechanism?: MechanismMotion;
    private fieldMotion?: ReceptiveFieldMotion;
    private attentionMotion?: AttentionTokenMotion;
    private gaussianMotion?: GaussianMotion;
    private residualRoutes: {
        node: string;
        curve: T.CatmullRomCurve3;
        mesh: T.Mesh;
        stream: ReturnType<typeof createArrowStream>;
    }[] = [];
    private representation: 'architecture' | 'operators' = 'operators';
    private highlightDependency?: {
        key: string;
        items: Dependency[];
    };
    private extents = new Map<string, number>();
    private dirty = true;
    private speed = 1;
    private onControlsChange = () => { this.dirty = true; };
    constructor(readonly container: HTMLElement, private options: ViewerOptions = {}) {
        this.labelMode = options.labels ?? 'auto';
        this.renderer = new T.WebGLRenderer({ antialias: true, alpha: options.transparentBackground ?? false, powerPreference: 'high-performance' });
        this.renderer.setClearColor('#06080d',options.transparentBackground?0:1);
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
        this.controls.addEventListener('change', this.onControlsChange);
        this.controls.addEventListener('start',this.stopFlight);
        this.observer = new ResizeObserver(() => {
            const w = container.clientWidth, h = container.clientHeight;
            // Hidden library routes must not collapse the drawing buffer or camera aspect.
            if (w < 1 || h < 1) return;
            this.renderer.setSize(w, h, false);
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
            this.dirty = true;
        });
        this.observer.observe(container);
        this.visibility = new IntersectionObserver(entries => { this.visible = entries[0].isIntersecting; if (this.visible) this.dirty = true; });
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
    private onReduced = () => {
        if (this.reduced.matches)
            {this.playing = false;this.tour=false;this.cameraFlight=undefined;}
    };
    private onLost = (e: Event) => { e.preventDefault(); this.visible = false; };
    private onRestored = () => {
        this.visible = true;
        if (this.model)
            this.setPage(this.pageStart);
    };
    load(model: unknown) {
        const candidate = validateModel(model), previous = this.model;
        this.generation++;
        this.windowRequest?.abort();
        this.origins.clear();
        this.moduleScope = '';
        this.functionScope = undefined;
        this.representation = candidate.architecture || candidate.nodes.some(n=>n.attrs?.attentionRole) ? 'architecture' : candidate.modules?.length && (candidate.nodes.length > 48 || candidate.modules.some(m => m.type === 'MAB') || candidate.functionalUnits?.some(u=>u.kind==='gaussian_reparameterization')) ? 'architecture' : 'operators';
        this.model = candidate;
        this.structureScope = candidate.architecture?.entry ?? ''; this.structureHistory=[];this.mechanismKey='';this.tour=false;this.cameraFlight=undefined;
        const initial = moduleView(candidate);
        if (this.representation === 'architecture' && initial.length === 1 && initial[0].op === 'module')
            this.moduleScope = String(initial[0].attrs?.modulePath ?? '');
        try {
            this.setPage(0);
        }
        catch (error) {
            this.model = previous;
            throw error;
        }
        return this;
    }
    setPage(start: number) {
        if (!this.model)
            return;
        const current=this.getDisplayedModel()!;
        const stage=this.mechanismStage>=0&&this.isMechanism?current.stages?.[this.mechanismStage]:undefined;
        const all = this.isSourceExecution ? this.functionScope?.nodes??moduleView(current,this.moduleScope,this.representation==='operators') : this.model.architecture ? (stage?current.nodes.filter(n=>stage.nodes.includes(n.id)):current.nodes) : this.functionScope?.nodes ?? moduleView(this.model, this.moduleScope, this.representation === 'operators'), limit = Math.min(128, Math.max(1, this.options.maxNodes ?? (this.isSourceExecution?96:48)));
        this.scopedCount = all.length;
        this.selected = undefined;
        this.hovered = undefined;
        this.pageStart = Math.max(0, Math.min(Math.max(0, all.length - 1), Math.floor(start)));
        const nodes = all.slice(this.pageStart, this.pageStart + limit), ids = new Set(nodes.flatMap(n => [...n.inputs, ...n.outputs, ...Object.values(n.parameters ?? {})]));
        if (!nodes.length)
            this.model.inputs.forEach(id => ids.add(id));
        if (ids.size > 512)
            throw Error('当前图段超过 512 个张量；请降低 maxNodes 或导出子模块。');
        const produced = new Set(nodes.flatMap(n => n.outputs));
        this.view = { ...current, nodes, tensors: current.tensors.filter(t => ids.has(t.id)), inputs: [...ids].filter(id => !produced.has(id) && current.tensors.find(t => t.id === id)?.role !== 'parameter'), outputs: nodes.at(-1)?.outputs ?? this.model.outputs };
        this.build();
        this.fit();
    }
    setLabels(mode: 'auto' | 'all' | 'none') { this.labelMode = mode; this.dirty = true; }
    getNavigation() { return { stage:this.mechanismStage,mechanism:this.mechanismKey, scopeName: this.isSourceExecution?this.moduleScope||this.getDisplayedModel()?.name:this.model?.architecture ? this.getDisplayedModel()?.name : undefined, scope: this.isSourceExecution?this.moduleScope:this.model?.architecture ? (this.structureScope===this.model.architecture.entry?'':this.structureScope) : this.moduleScope, function: this.functionScope?.name, representation: this.representation, nodes: [...this.nodes] }; }
    setRepresentation(mode: 'architecture' | 'operators') { this.representation = mode; this.setPage(0); }
    parentModule() { if(this.isSourceExecution&&this.functionScope){this.functionScope=undefined;this.setPage(0);return;} if(this.isSourceExecution&&this.moduleScope&&this.moduleScope!=='decoder'){this.showModule(this.moduleScope.split('.').slice(0,-1).join('.'));return;} if(this.mechanismKey){this.showStructure();return;} if(this.model?.architecture){ this.structureScope=this.structureHistory.pop() ?? this.model.architecture.entry;this.setPage(0);return; } if (this.functionScope) {
        this.functionScope = undefined;
        this.setPage(0);
        return;
    } this.showModule(this.moduleScope.split('.').slice(0, -1).join('.')); }
    update(model: unknown) {
        const structure=this.structureScope,history=[...this.structureHistory];
        const position = this.camera.position.clone(), target = this.controls.target.clone(), selected = this.selected, scope = this.moduleScope, representation = this.representation, page = this.pageStart;
        this.load(model);
        if(this.model?.architecture?.scopes[structure]){this.structureScope=structure;this.structureHistory=history.filter(k=>this.model!.architecture!.scopes[k]);}
        this.representation = representation;
        if (!scope || this.getModules().includes(scope))
            this.moduleScope = scope;
        this.setPage(page);
        this.cameraFlight=undefined;
        this.camera.position.copy(position);
        this.controls.target.copy(target);
        this.selected = this.nodes.some(n => n.id === selected) ? selected : undefined;
        this.controls.update();
    }
    setDataProvider(provider: ViewerOptions['dataProvider']) { this.windowRequest?.abort(); this.options.dataProvider = provider; }
    setSpeed(speed: number) {
        if (!Number.isFinite(speed) || speed < .1 || speed > 4)
            throw Error('速度范围为 0.1–4。');
        this.speed = speed;
    }
    setProgress(progress: number) {
        if (!Number.isFinite(progress) || progress < 0 || progress > 1)
            throw Error('进度范围为 0–1。');
        this.elapsed = Math.min(.999999, progress) / .4;
        this.selected = this.hovered ?? this.selected ?? (this.active || undefined);
        this.playing = false;
        this.dirty = true;
    }
    showModule(path = '') {
        if(this.isSourceExecution){if(path&&!this.getModules().includes(path))throw Error('不存在这个执行模块');this.moduleScope=path;this.functionScope=undefined;this.setPage(0);this.options.onSelect?.(undefined);return;}
        if(this.model?.architecture){const key=path||this.model.architecture.entry;if(!this.model.architecture.scopes[key])throw Error('模板不存在');this.mechanismKey='';if(key!==this.structureScope)this.structureHistory.push(this.structureScope);this.tour=false;this.structureScope=key;const scope=this.model.architecture.scopes[key];if(scope.executionRef){this.mechanismKey=scope.executionRef;this.moduleScope=scope.executionScope??'decoder';this.mechanismStage=-1;this.representation='architecture';}this.setPage(0);return;}
        if (path && !this.getModules().includes(path))
            throw Error('不存在这个模块路径。');
        this.functionScope = undefined;
        this.moduleScope = path;
        this.setPage(0);
    }
    getModules() { if(this.isSourceExecution){const g=this.getDisplayedModel()!;return g.modules?.filter(m=>g.nodes.some(n=>n.group===m.path||n.group?.startsWith(m.path+'.'))).map(m=>m.path)??[];} if(this.model?.architecture)return Object.keys(this.model.architecture.scopes); return [...new Set(this.model?.nodes.flatMap(n => { const parts = (n.group ?? n.id).split('.'); return parts.map((_, i) => parts.slice(0, i + 1).join('.')); }) ?? [])].sort(); }
    async setTensorWindow(id: string, origin: number[]) {
        if(this.model?.architecture&&!this.isSourceExecution)throw Error("结构分区覆盖逻辑形状；数值窗口需要导出实际执行模型。");
        const t = this.getDisplayedModel()?.tensors.find(t => t.id === id);
        if (!t || origin.length !== t.shape.length || origin.some(v => !Number.isInteger(v) || v < 0))
            throw Error('起点必须与张量维度一致，且为非负整数。');
        if (!this.planes.has(id))
            throw Error('请先进入包含该张量的图段。');
        this.windowRequest?.abort();
        const request = new AbortController();
        this.windowRequest = request;
        const generation = this.generation;
        this.origins.set(id, origin);
        const affected = new Set([id]);
        for (const node of this.nodes)
            if (node.op === 'linear' && node.parameters?.weight) {
                const input = this.tensors.get(node.inputs[0])!, weight = node.parameters.weight;
                if (id === input.id) {
                    const current = this.origins.get(weight) ?? [0, 0];
                    this.origins.set(weight, [current[0], origin.at(-1)!]);
                    affected.add(weight);
                }
                if (id === weight) {
                    const current = this.origins.get(input.id) ?? input.shape.map(() => 0);
                    this.origins.set(input.id, [...current.slice(0, -1), origin[1]]);
                    affected.add(input.id);
                }
            }
        this.build();
        if (this.options.dataProvider&&!this.isSourceExecution) {
            try {
                const loaded = await Promise.all([...affected].map(async (tensor) => {
                    const indices = this.planes.get(tensor)!.indices, values = await this.options.dataProvider!(tensor, indices, request.signal);
                    if (values.length !== indices.length || values.some(v => v !== null && (typeof v !== 'number' || !Number.isFinite(v))))
                        throw Error('服务返回的张量数值无效。');
                    return { tensor, indices, values };
                }));
                if (request.signal.aborted || generation !== this.generation)
                    return;
                for (const entry of loaded)
                    this.tensors.get(entry.tensor)!.samples = Object.fromEntries(entry.indices.map((i, j) => [i, entry.values[j]]));
                this.build();
            }
            catch (e) {
                if (!request.signal.aborted)
                    throw e;
            }
        }
    }
    setPlaying(playing: boolean) { this.playing = playing; this.dirty = true; }
    get isPlaying() { return this.playing; }
    /** Select changes camera only on this explicit call; hover never moves the camera. */
    focus(id: string) {
        if (!this.model)
            return;
        if(this.model.architecture&&!this.mechanismKey){const n=this.nodes.find(n=>n.id===id);if(!n)return;if(n.attrs?.scopeRef){this.showModule(String(n.attrs.scopeRef));this.options.onSelect?.(undefined);}else if(n.attrs?.mechanismRef){this.showMechanism(String(n.attrs.mechanismRef),Number(n.attrs.mechanismStage??-1));}else{this.selected=id;this.options.onSelect?.(n);this.fit([...n.inputs,...n.outputs,...Object.values(n.parameters??{})]);}return;}
        if(this.isMechanism&&!this.nodes.some(n=>n.id===id)){const i=this.computationStages.findIndex(s=>s.nodes.includes(id));if(i>=0)this.setComputationStage(i);}
        const node = this.getDisplayedModel()!.nodes.find(n => n.id === id);
        if (!node) {
            const group = this.nodes.find(n => n.id === id);
            if ((group?.op === 'function' || group?.op === 'gaussian_sample' || group?.op === 'semantic' || group?.op === 'layout')) {
                this.functionScope = { name: group.name, nodes: group.attrs?.children as Operation[] };
                this.setPage(0);
                this.options.onSelect?.(group);
                return;
            }
            if (group?.op === 'layout') {
                const child = (group.attrs?.children as Operation[])?.at(-1);
                this.setRepresentation('operators');
                if (child)
                    this.focus(child.id);
                return;
            }
            if (group?.attrs?.modulePath) {
                this.showModule(String(group.attrs.modulePath));
                this.options.onSelect?.(group);
            }
            return;
        }
        if (!this.nodes.some(n => n.id === id)) {
            this.representation = 'operators';
            this.moduleScope = '';
            const i = topologicalNodes(this.model).findIndex(n => n.id === id);
            this.setPage(Math.floor(i / (this.options.maxNodes ?? 48)) * (this.options.maxNodes ?? (this.isSourceExecution?96:48)));
        }
        this.selected = id;
        this.options.onSelect?.(node);
        const ids = [...node.inputs, ...node.outputs, ...Object.values(node.parameters ?? {})];
        this.fit(ids);
    }
    clearFocus() { this.selected = undefined; this.options.onSelect?.(undefined); this.fit(); }
    zoom(factor: number) { this.stopFlight(); const delta = this.camera.position.clone().sub(this.controls.target).multiplyScalar(factor); delta.clampLength(this.controls.minDistance, this.controls.maxDistance); this.camera.position.copy(this.controls.target).add(delta); this.controls.update(); }
    async fullscreen() {
        if (document.fullscreenElement === this.container)
            await document.exitFullscreen();
        else
            await this.container.requestFullscreen();
    }
    fit(ids?: string[]) {
        this.dirty = true;
        const box = new T.Box3();
        for (const [id, p] of this.planes)
            if (!ids || ids.includes(id)) {
                box.expandByPoint(new T.Vector3(p.center[0] - p.width / 2, p.center[1] - 1, p.center[2] - p.depth / 2));
                box.expandByPoint(new T.Vector3(p.center[0] + p.width / 2, p.center[1] + Math.max(p.height,this.model?.architecture&&!this.mechanismKey?3.3:0), p.center[2] + p.depth / 2 + 1));
            }
        this.root.updateMatrixWorld(true);
        // Fit the model geometry; captions adapt to this view instead of shrinking the model.
        if (!ids) box.expandByVector(new T.Vector3(2.2, .2, .2));
        if (box.isEmpty())
            box.setFromCenterAndSize(new T.Vector3(), new T.Vector3(5, 5, 5));
        const size = box.getSize(new T.Vector3()), center = box.getCenter(new T.Vector3());
        const vertical = Math.max(size.y, size.z * .7), horizontal = size.x / Math.max(.3, this.camera.aspect), distance = Math.max(vertical, horizontal, 4) / (2 * Math.tan(T.MathUtils.degToRad(this.camera.fov / 2))) * 1.3;
        const destination=center.clone().add(new T.Vector3(.22,.22,1).normalize().multiplyScalar(distance));
        if(!this.reduced.matches&&this.camera.position.length()>1){
            this.cameraFlight={from:this.camera.position.clone(),to:destination,targetFrom:this.controls.target.clone(),targetTo:center,time:0};
        }else{this.cameraFlight=undefined;this.controls.target.copy(center);this.camera.position.copy(destination);this.controls.update();}
    }
    private cleanup() {
        this.fabrics.forEach(f => f.dispose());
        this.fabrics.clear();
        this.structureMotion=undefined;
        this.edges.clear();
        const geometries = new Set<T.BufferGeometry>(), materials = new Set<T.Material>(), textures = new Set<T.Texture>();
        this.root.traverse(o => {
            const mesh = o as T.Mesh;
            if (mesh.geometry)
                geometries.add(mesh.geometry);
            if (mesh.material)
                for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
                    materials.add(m);
                    for (const v of Object.values(m))
                        if (v instanceof T.Texture)
                            textures.add(v);
                }
            if (o instanceof T.InstancedMesh)
                o.dispose();
        });
        geometries.forEach(g => g.dispose());
        materials.forEach(m => m.dispose());
        textures.forEach(t => t.dispose());
        this.root.clear();
        this.cards = [];
        this.planeOutlines.clear();
        this.arrows = [];
        this.activationCurves.clear();
        this.regions = [];
        this.positions.clear();
        this.extents.clear();
        this.hoveredWeight = undefined;
        this.cachedActive = '';
        this.connectionCount = 0;
        this.highlight = undefined;
        this.receptive = undefined;
        this.residualRoutes = [];
        this.mechanism=undefined;this.fieldMotion=undefined;this.attentionMotion=undefined;this.gaussianMotion=undefined;this.arrows=[];
    }
    private build() {
        this.dirty = true;
        this.highlightDependency = undefined;
        this.cleanup();
        const model = this.view!;
        this.nodes = topologicalNodes(model);
        this.tensors = new Map(this.getDisplayedModel()!.tensors.map(t => [t.id, t]));
        const budget = Math.max(512, Math.min(16384, this.options.maxCells ?? 8192)), per = Math.max(1, Math.min(this.options.cellsPerTensor ?? 128, Math.floor(budget / Math.max(1, model.tensors.length))));
        this.planes = layoutModel(model, per, this.origins);
        const pos: Position3[] = [], values: number[] = [], weightPos:Position3[]=[], weightValues:number[]=[], structuralColors:string[]=[], weightColors:string[]=[];
        for (const p of this.planes.values()) {
            this.positions.set(p.tensor.id, new Map(p.indices.map((v, i) => [v, p.positions[i]])));
            const extent = Math.max(p.tensor.stats?.absmax ?? 0, valueExtent([...(p.tensor.data?.values ?? []), ...Object.values(p.tensor.samples ?? {})].map(v => v ?? NaN)));
            this.extents.set(p.tensor.id, Math.max(extent, 1e-12));
            const weight=p.tensor.role==='parameter';
            (weight?weightPos:pos).push(...p.positions);
            (weight?weightValues:values).push(...p.indices.map(i => valueAt(p.tensor, i) / Math.max(extent, 1e-12)));
            const owner=this.nodes.find(n=>n.id===p.owner), color=weight?'#d0b790':p.tensor.role==='buffer'?'#a3b7a5':structureColors[String(owner?.attrs?.kind)]??'#91bbca';
            (weight?weightColors:structuralColors).push(...p.positions.map(()=>color));
            const plate = new T.Mesh(new T.BoxGeometry(Math.max(p.width, .8), .018, Math.max(p.depth, .5)), new T.MeshBasicMaterial({ color: '#3c7187', transparent: true, opacity: .1, depthWrite: false }));
            plate.position.set(p.center[0], p.center[1] - .16, p.center[2]);
            this.root.add(plate);
            const outline = new T.LineSegments(new T.EdgesGeometry(plate.geometry), new T.LineBasicMaterial({ color: p.tensor.role === 'parameter' ? '#bc986c' : '#65b4c7', transparent: true, opacity: .48 }));
            outline.position.copy(plate.position);
            this.root.add(outline);
            this.planeOutlines.set(p.tensor.id,outline);
            if (p.tensor.specialValues) {
                const crosses: T.Vector3[] = [];
                p.indices.forEach((index, i) => {
                    if (p.tensor.specialValues?.[index] === '-inf') {
                        const [x, y, z] = p.positions[i];
                        crosses.push(new T.Vector3(x - .11, y + .17, z - .11), new T.Vector3(x + .11, y + .17, z + .11), new T.Vector3(x - .11, y + .17, z + .11), new T.Vector3(x + .11, y + .17, z - .11));
                    }
                });
                if (crosses.length)
                    this.root.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(crosses), new T.LineBasicMaterial({ color: '#dcad7e', transparent: true, opacity: .65 })));
            }
            this.addCard(p);
        }
        this.cells = pos.length+weightPos.length;
        if (pos.length) {
            const crystal = createCrystalTensor(this.root, pos.length, model.architecture?.mode==='structure'?.42:.26, { valueEdges: true, valueScale: 1, bodyOpacity: .2, edgeOpacity: .5 });
            crystal.update(values, pos);
            if(model.architecture)for(let i=0;i<pos.length;i++){crystal.body.setColorAt(i,new T.Color(structuralColors[i]));crystal.edge.setColorAt(i,new T.Color(structuralColors[i]));}
        }
        if(weightPos.length){const weights=createCrystalTensor(this.root,weightPos.length,model.architecture?.mode==='structure'?.44:.28,{parameter:true,valueEdges:true,valueScale:1,bodyOpacity:.3,edgeOpacity:.7});weights.update(weightValues,weightPos);if(model.architecture)for(let i=0;i<weightPos.length;i++){weights.body.setColorAt(i,new T.Color(weightColors[i]));weights.edge.setColorAt(i,new T.Color(weightColors[i]));}}
        if(model.architecture)this.structureMotion=new StructureMotion(this.root,this.nodes,this.planes,model);
        if(!model.architecture) {
        this.highlight = createCrystalTensor(this.root, 256, .27, { valueEdges: true, valueScale: 1 });
        this.mechanism = new MechanismMotion(this.root);
        this.fieldMotion = new ReceptiveFieldMotion(this.root);
        this.attentionMotion = new AttentionTokenMotion(this.root);
        this.gaussianMotion = new GaussianMotion(this.root);
        this.arrows = Array.from({ length: 6 }, () => createArrowStream(this.root, '#d2edf5', .05, 1));
        this.receptive = new T.Box3Helper(new T.Box3(), new T.Color('#e5d596'));
        this.receptive.visible = false;
        this.root.add(this.receptive);
        }
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
                        const residual = node.attrs?.residualInput === id;
                        const skip = output.level - input.level > 1;
                        if (residual) {
                            const left = Math.min(...[...this.planes.values()].filter(p => p.level >= input.level && p.level <= output.level).map(p => p.center[0] - p.width / 2)) - .7 - this.residualRoutes.length * .20;
                            const z = Math.max(input.depth, output.depth) / 2 + .35;
                            const points = [new T.Vector3(...input.center), new T.Vector3(left, input.center[1] - .5, z), new T.Vector3(left, output.center[1] + .5, z), new T.Vector3(...output.center)];
                            const curve = new T.CatmullRomCurve3(points, false, 'centripetal');
                            const mesh = new T.Mesh(new T.TubeGeometry(curve, 40, .025, 5, false), new T.MeshBasicMaterial({ color: '#f4c991', transparent: true, opacity: .55, depthWrite: false }));
                            this.root.add(mesh);
                            this.residualRoutes.push({ node: node.id, curve, mesh, stream: createArrowStream(this.root, '#ffe1a7', .085, 3) });
                            continue;
                        }
                        // Coordinate relocation is animated with crystal motion, not calculation arrows.
                        if(!['layout','reshape','transpose','permute','expand','flatten','identity'].includes(node.op))
                            backbone.push({ from: input.center, to: output.center, via: skip ? [[input.center[0], input.center[1], -Math.max(input.depth,output.depth)/2-.8], [input.center[0], output.center[1], -Math.max(input.depth,output.depth)/2-.8]] : undefined });
                    }
                }
            }
            connectionFabric(this.root, backbone).update(0, -1, -1, 0, .55);
            if(model.architecture)this.connectionCount=this.nodes.reduce((sum,n)=>sum+n.inputs.length+Object.keys(n.parameters??{}).length,0);
        }
    }
    private addCard(p: TensorPlane) {
        const canvas = document.createElement('canvas');
        canvas.width = 896;
        const cardOwner=this.nodes.find(n=>n.id===p.owner);
        const inner=cardOwner?.op==='structure'?this.model?.architecture?.scopes[String(cardOwner.attrs?.scopeRef)]:undefined;
        
        const summary=inner?structureSummary(inner.nodes):[];
        const composition=Number(cardOwner?.attrs?.repeat??1)>1?summary.reduce<string[]>((rows,line,i)=>{if(i%2===0)rows.push(line);else rows[rows.length-1]+=' → '+line;return rows;},[]):summary;
        const textRows=composition.slice(0,2).flatMap(text=>{const lines:string[]=[];let line='';for(const char of text){if(line.length>=28){lines.push(line);line='';}line+=char;}if(line)lines.push(line);return lines;});
        canvas.height = (inner?185:140)+textRows.length*38;
        const c = canvas.getContext('2d')!;
        const owner = this.nodes.find(n => n.id === p.owner), visual = operatorVisual(owner?.op ?? '');
        const roles = (owner?.attrs?.projectionRoles ?? []) as string[], role = roles.join('/');
        const parameter = p.tensor.role === 'parameter' || p.tensor.role === 'buffer';
        const title = !parameter&&owner?.attrs?.displayLabel?String(owner.attrs.displayLabel):['semantic','layout'].includes(owner?.op??'')&&!parameter?owner!.name.split(' · ')[0]:modelTitle();
        function modelTitle(){return owner?.op==='structure'&&!parameter?owner.name:parameter ? ((owner?.parameters?.weight === p.tensor.id || owner?.parameters?.W === p.tensor.id || owner?.parameters?.W_KV === p.tensor.id) ? (owner.op.includes('norm') ? '缩放 γ' : role ? 'W' + role : '权重 W') : owner?.parameters?.bias === p.tensor.id ? '偏置 b' : p.tensor.id.endsWith(':gamma')?'归一化尺度 γ':p.tensor.semantic ?? p.tensor.source ?? p.tensor.id) : role ? role + ' · Token 投影' : owner?.attrs?.attentionRole === 'score' ? 'Query × Key' : owner?.attrs?.attentionRole === 'probability' ? '注意力分配' : owner?.attrs?.attentionRole === 'context' ? 'Token · 加权结果' : owner?.attrs?.residualInput ? '残差汇合' : p.tensor.semantic ?? (owner?.op === 'module' ? String(owner.attrs?.moduleType) : owner ? visual.label : '输入');}

        // Typography floats in world space; no frame or detached leader line.
        c.fillStyle = '#f4f8fa';
        c.font = '500 55px "Segoe UI", sans-serif';
        c.fillText(title.length > 28 ? title.slice(0, 27) + '…' : title, 14, 62, 855);
        c.fillStyle = owner ? visual.color : '#a8ccd7';
        c.font = '400 33px "Segoe UI", sans-serif';
        c.fillText(p.tensor.shape.join(' × ') + '  ' + (p.tensor.axes?.join(' · ') ?? p.tensor.dtype), 16, 115, 850);
        c.fillStyle = '#b7c7d3';
        c.font = '29px "Segoe UI", sans-serif';
        const line = p.tensor.representation==='aggregate' ? (owner?.attrs?.scopeRef?'点击展开 · ':'')+'逻辑分区 · 不代表单个标量' : p.partial ? `窗口 ${p.windowShape.join(' × ')} · ${owner?.name ?? p.tensor.id}` : owner?.op === 'module' ? '点击进入真实计算' : owner?.attrs?.attentionRole === 'context' ? '每个输出 Token = Σ 概率 × Value' : owner?.attrs?.attentionRole === 'probability' ? '每行对应一个 Query；沿 Key 归一化' : owner?.name ?? p.tensor.id;
        const repeats=Number(owner?.attrs?.repeat??1);
        if(inner)c.fillText(repeats>1?`× ${repeats} 独立单元 · 重复模板`:line,16,166,850);
        if(textRows.length){
            c.strokeStyle='#8bbdd96b';c.lineWidth=1;c.beginPath();c.moveTo(16,183);c.lineTo(700,183);c.stroke();
            c.font='32px \"Segoe UI\", sans-serif';
            textRows.forEach((text,i)=>{c.fillStyle='#c4dce7';c.fillText(text,16,223+i*38,850);});
        }
        const texture = new T.CanvasTexture(canvas);
        texture.colorSpace = T.SRGBColorSpace;
        const width = this.nodes.length>12?7.2:5.2;
        const mesh = new T.Mesh(new T.PlaneGeometry(width, width * canvas.height / 896), new T.MeshBasicMaterial({ map: texture, transparent: true, opacity: .94, side: T.DoubleSide, depthWrite: false }));
        const group = new T.Group();
        mesh.position.y = 0;
        group.add(mesh);
        // Place once, in the tensor's coordinate frame. Never move for screen-space collisions.
        const side=parameter||p.center[0]<-.1?-1:1;
        const peers=[...this.planes.values()].filter(v=>Math.abs(v.center[1]-p.center[1])<.01);
        const neighbors=peers.filter(v=>side<0?v.center[0]<p.center[0]:v.center[0]>p.center[0]);
        // Parallel lanes use their own depth so captions do not all get detached to a global gutter.
        const z=neighbors.length?p.center[2]+p.depth/2+.65:p.center[2];
        group.position.set(p.center[0]+side*(p.width/2+width/2+.35),p.center[1]+(inner?1.55:.35),z);
        group.userData.composition=composition;
        group.userData.tensorId=p.tensor.id;
        group.userData.fixedAnchor=true;
        const accent=new T.Mesh(new T.BoxGeometry(.2,.035,.025),new T.MeshBasicMaterial({color:owner?.op==='structure'?structureColors[String(owner.attrs?.kind)]??visual.color:visual.color,transparent:true,opacity:.8}));
        accent.position.set(-width/2+.06,.12,.025);group.add(accent);
        this.root.add(group);
        this.cards.push({ group, anchor: group.position.clone(), id: p.tensor.id, owner: p.owner, priority: parameter || !owner || owner.op==='layout' ? 0 : owner.op==='structure'||owner.attrs?.attentionRole||owner.attrs?.gaussianHead ? 2 : 1 });
    }
    private addActivationCurve(n: Operation, input: TensorPlane, out: TensorPlane) {
        const fn=operatorVisual(n.op).curve!, curve=(x:number)=>fn(x,n.attrs??{});
        const extent=Math.max(.5,input.tensor.stats?.absmax??0,valueExtent(input.tensor.data?.values.map(v=>v??NaN)??[]));
        const gap=Math.abs(input.center[1]-out.center[1]);
        const position:Position3=[out.center[0],(input.center[1]+out.center[1])/2,Math.max(input.center[2]+input.depth/2,out.center[2]+out.depth/2)+.45];
        this.activationCurves.set(n.id,new FunctionPlot(this.root,position,curve,extent,gap));
    }
    private links(n: Operation): Connection[] {
        if (['layout','reshape','transpose','permute','expand','flatten','identity'].includes(n.op)) return [];
        const visual = operatorVisual(n.op), result: Connection[] = [], out = this.planes.get(n.outputs[0])!;
        const cap = Math.max(64, this.options.maxConnections ?? 12000);
        let outputOffset = 0;
        for (const outputId of n.outputs) {
            const target = this.planes.get(outputId)!;
            for (let oi = 0; oi < target.indices.length; oi++) {
                const index = target.indices[oi], to = target.positions[oi];
                const deps = visual.dependencies(n, target.tensor, index, this.tensors);
                for (let term = 0; term < deps.length; term++) {
                    const d = deps[term], from = this.positions.get(d.tensor)?.get(d.index);
                    if (!from)
                        continue;
                    const wt = d.parameter ? this.tensors.get(d.parameter) : undefined;
                    const scale = wt?.stats?.absmax ?? valueExtent(wt?.data?.values.map(v => v ?? NaN) ?? []);
                    const route = n.attrs?.residualInput === d.tensor ? this.residualRoutes.find(r => r.node === n.id) : undefined;
                    result.push({ from, to, via: route ? route.curve.points.slice(1, -1).map(p => [p.x, p.y, p.z] as Position3) : undefined, weight: Number.isFinite(d.weight) ? d.weight : undefined, weightScale: scale, output: outputOffset + oi, term, parameterIndex: d.parameter === n.parameters?.weight ? d.parameterIndex : undefined });
                    if (result.length >= cap)
                        return result;
                }
            }
            outputOffset += target.indices.length;
        }
        if (!result.length && visual.detail === 'boundary') {
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
        this.dirty = true;
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
        this.hoveredWeight = undefined;
        const cellBox = new T.Box3(), cellSize = new T.Vector3(.29, .29, .29);
        let cellDistance = Infinity;
        for (const n of this.nodes) {
            const id = n.parameters?.weight, p = id ? this.planes.get(id) : undefined;
            if (!p)
                continue;
            const bounds = new T.Box3(new T.Vector3(p.center[0] - p.width / 2, p.center[1] - .2, p.center[2] - p.depth / 2), new T.Vector3(p.center[0] + p.width / 2, p.center[1] + p.height, p.center[2] + p.depth / 2));
            if (!this.ray.ray.intersectsBox(bounds))
                continue;
            for (let i = 0; i < p.indices.length; i++) {
                cellBox.setFromCenterAndSize(new T.Vector3(...p.positions[i]), cellSize);
                const hit = this.ray.ray.intersectBox(cellBox, point);
                if (hit && hit.distanceTo(this.ray.ray.origin) < cellDistance) {
                    cellDistance = hit.distanceTo(this.ray.ray.origin);
                    this.hoveredWeight = { tensor: id!, index: p.indices[i] };
                    this.hovered = n.id;
                }
            }
        }
        const captionHit=this.ray.intersectObjects(this.cards.filter(c=>c.group.visible&&c.owner).map(c=>c.group.children[0]),false)[0];
        if(captionHit){const card=this.cards.find(c=>c.group.children[0]===captionHit.object)!;this.hovered=card.owner;this.hoveredWeight=undefined;}
        this.renderer.domElement.style.cursor = this.hovered ? 'pointer' : 'grab';
    };
    private onLeave = () => { this.hovered = undefined; this.hoveredWeight = undefined; this.dirty = true; };
    private onDown = (e: PointerEvent) => { this.pointerStart = [e.clientX, e.clientY]; };
    private onUp = (e: PointerEvent) => {
        if (this.pointerStart && Math.hypot(e.clientX - this.pointerStart[0], e.clientY - this.pointerStart[1]) < 4 && this.hovered)
            this.focus(this.hovered);
        this.pointerStart = undefined;
    };
    private animate = (time: number) => {
        if (this.disposed)
            return;
        this.frame = requestAnimationFrame(this.animate);
        const dt = Math.min(.05, (time - this.last) / 1000 || 0);
        this.last = time;
        if (!this.visible || document.hidden)
            return;
        if(this.cameraFlight){const f=this.cameraFlight;f.time+=dt;const t=Math.min(1,f.time/.72),e=1-Math.pow(1-t,4);this.camera.position.lerpVectors(f.from,f.to,e);this.controls.target.lerpVectors(f.targetFrom,f.targetTo,e);this.dirty=true;if(t===1)this.cameraFlight=undefined;}
        this.controls.update();
        if(this.tour&&this.playing&&!this.hovered&&!this.cameraFlight){this.tourClock+=dt;if(this.tourClock>12&&this.computationStages.length){this.setComputationStage((this.mechanismStage+1)%this.computationStages.length);}}
        if (!this.playing && !this.dirty)
            return;
        this.dirty = false;
        if (this.playing)
            this.elapsed += dt * this.speed;
        this.clock += dt;
        const candidates = this.nodes.filter(n => n.inputs.length && !['cast', 'identity', 'arange'].includes(n.op)), auto = candidates[Math.floor(this.elapsed / 3.6) % Math.max(1, candidates.length)];
        this.active = this.hovered ?? this.selected ?? auto?.id ?? '';
        if(this.structureMotion){this.structureMotion.update(this.active,this.elapsed);this.weightState=undefined;this.sampleState=undefined;for(const r of this.residualRoutes)r.stream.update(r.curve,this.elapsed*.25,r.node===this.active?1:.2);}
        else {
        this.setActive(this.active);
        const active = this.nodes.find(n => n.id === this.active), slot = active ? Math.floor(this.elapsed / 1.8) % active.outputs.length : 0, out = active ? this.planes.get(active.outputs[slot]) : undefined;
        const outputOffset = active ? active.outputs.slice(0, slot).reduce((sum, id) => sum + (this.planes.get(id)?.indices.length ?? 0), 0) : 0;
        const phase = (this.elapsed * .4) % 1;
        let linked = -1, outputIndex = out ? Math.floor(phase * out.indices.length) : 0, termCursor = phase * 8;
        this.weightState = undefined;
        this.sampleState = undefined;
        if (active?.op === 'linear' && out && active.parameters?.weight) {
            const weight = this.tensors.get(active.parameters.weight)!, width = Number(weight.shape[1]), outputs = Number(weight.shape[0]);
            const visibleWeights = this.planes.get(weight.id)?.indices ?? [], rows = [...new Set(visibleWeights.map(i => Math.floor(i / width)))].filter(row => out.indices.some(i => i % outputs === row));
            const row = this.hoveredWeight?.tensor === weight.id ? Math.floor(this.hoveredWeight.index / width) : rows[Math.floor(phase * rows.length)] ?? (out.indices[outputIndex] ?? 0) % outputs;
            const columns = visibleWeights.filter(i => Math.floor(i / width) === row).map(i => i % width);
            const cursor = Math.min(columns.length - 1, (this.elapsed * 2) % (columns.length + .5));
            termCursor = this.hoveredWeight?.tensor === weight.id ? this.hoveredWeight.index % width : (columns[Math.max(0, Math.floor(cursor))] ?? 0) + (cursor % 1);
            linked = row * width + termCursor;
            outputIndex = Math.max(0, out.indices.findIndex(i => i % outputs === row));
            const index = Math.floor(linked);
            this.weightState = { tensor: weight.id, index, coordinates: coordinates(index, weight.shape as number[]), value: valueAt(weight, index) };
        }
        if (active?.attrs?.gaussian && out) outputIndex = Math.floor(this.elapsed / 2.5) % out.indices.length;
        this.gaussianMotion?.update(active, out, out?.indices[outputIndex] ?? 0, phase, this.tensors);
        this.fieldMotion?.update(active, out, out?.indices[outputIndex] ?? 0, this.planes, this.tensors);
        this.attentionMotion?.update(active, out, out?.indices[outputIndex] ?? 0, phase, this.planes, this.tensors);
        for (const r of this.residualRoutes) {
            const selected = r.node === this.active;
            (r.mesh.material as T.MeshBasicMaterial).opacity = selected ? .88 : .5;
            r.stream.group.visible = selected;
            if (selected)
                r.stream.update(r.curve, phase, 1);
        }
        this.mechanism?.update(active, out, out?.indices[outputIndex] ?? 0, phase, this.planes, this.tensors);
        for (const [id, f] of this.fabrics) {
            const n = this.nodes.find(n => n.id === id)!, p = this.planes.get(n.outputs[0])!;
            f.update(phase, id === this.active ? outputOffset + outputIndex : phase * p.indices.length, id === this.active ? termCursor : phase * 8, id === this.active ? (n.attrs?.attentionRole === 'context' ? .10 : 1) : .12, n.attrs?.attentionRole === 'context' ? .025 : n.op.startsWith('conv') ? .07 : id === this.active ? 1 : .65, id === this.active ? linked : -1);
        }
        this.connectionCount = [...this.edges.values()].reduce((s, e) => s + e.length, 0);
        const flowing = this.edges.get(this.active) ?? [], current = flowing.filter(e => linked >= 0 ? e.parameterIndex !== undefined && Math.abs(e.parameterIndex - linked) < 1 : e.output === outputOffset + outputIndex);
        this.arrows.forEach((arrow, i) => {
            const edge = current[i];
            arrow.group.visible = !!edge;
            if (edge) {
                const middle: Position3 = edge.from.map((v, j) => (v + edge.to[j]) / 2) as Position3;
                arrow.update([edge.from, ...edge.via ?? [middle], edge.to], phase + i * .13, 1);
            }
        });
        if (active && out && this.highlight) {
            const oi = Math.min(out.indices.length - 1, outputIndex), index = out.indices[oi];
            const ps: Position3[] = [], vs: number[] = [];
            const push = (id: string, i: number) => {
                const p = this.positions.get(id)?.get(i), t = this.tensors.get(id);
                if (p && t && ps.length < 256) {
                    ps.push(p);
                    vs.push(valueAt(t, i) / (this.extents.get(id) ?? 1));
                }
            };
            if (index !== undefined) {
                push(out.tensor.id, index);
                if (linked >= 0 && active.op === 'linear') {
                    const width = Number(this.tensors.get(active.parameters!.weight)!.shape[1]), outputs = Number(out.tensor.shape.at(-1)), row = Math.floor(linked / width), column = Math.floor(linked) % width;
                    for (const output of out.indices)
                        if (output % outputs === row) {
                            push(out.tensor.id, output);
                            push(active.inputs[0], Math.floor(output / outputs) * width + column);
                        }
                }
                const key = `${active.id}/${out.tensor.id}/${index}`;
                if (this.highlightDependency?.key !== key)
                    this.highlightDependency = { key, items: operatorVisual(active.op).dependencies(active, out.tensor, index, this.tensors) };
                const deps = this.highlightDependency.items, rf = new T.Box3();
                for (const d of deps) {
                    if (linked >= 0 && d.parameter === active.parameters?.weight && Math.abs((d.parameterIndex ?? -100) - linked) > 1.2)
                        continue;
                    push(d.tensor, d.index);
                    const p = this.positions.get(d.tensor)?.get(d.index);
                    if (p && active.op.startsWith('conv') && d.tensor === active.inputs[0])
                        rf.expandByPoint(new T.Vector3(...p));
                    if (d.parameter && d.parameterIndex !== undefined)
                        push(d.parameter, d.parameterIndex);
                }
                if (this.receptive) {
                    this.receptive.visible = false;
                    if (!rf.isEmpty())
                        this.receptive.box.copy(rf.expandByScalar(.2));
                }
            }
            this.highlight.update(vs, ps, { active: ps.map((_, i) => i), focus: 0 });
            this.highlight.body.visible = this.highlight.edge.visible = false;
        }
        for (const [id, plot] of this.activationCurves) {
            const n=this.nodes.find(n=>n.id===id)!, plane=this.planes.get(n.outputs[0])!;
            const index=id===this.active?out?.indices[outputIndex]??plane.indices[0]:plane.indices[0];
            const x=valueAt(this.tensors.get(n.inputs[0])!,index),y=valueAt(this.tensors.get(n.outputs[0])!,index);
            const active=id===this.active&&Number.isFinite(x)&&Number.isFinite(y);
            plot.update(x,y,active,this.nodes.length<=12);
            if(active)this.sampleState={input:x,output:y,index};
        }
        }
        const shown:T.Box2[]=[];
        this.root.updateMatrixWorld(true);
        const projectPlane=(p:TensorPlane)=>new T.Box2().setFromPoints([-1,1].flatMap(x=>[-1,1].flatMap(z=>[-.2,Math.max(p.height,this.structureMotion?3.3:0)].map(y=>{
            const q=new T.Vector3(p.center[0]+x*p.width/2,p.center[1]+y,p.center[2]+z*p.depth/2).project(this.camera);return new T.Vector2(q.x,q.y);
        }))));
        const occupied=[...this.planes.values()].map(projectPlane);
        const chosen=this.hovered??this.selected;
        const cardCandidates=[...this.cards].sort((a,b)=>Number(b.owner===chosen)-Number(a.owner===chosen)||b.priority-a.priority);
        const defaults=new Set(cardCandidates.filter(c=>c.priority>0).slice(0,this.structureMotion?6:4));
        for(const card of cardCandidates){
            const wanted=this.labelMode==='all'||this.labelMode==='auto'&&(chosen?card.owner===chosen:defaults.has(card));
            card.group.visible=false;
            if(!wanted)continue;
            const mesh=card.group.children[0] as T.Mesh;mesh.geometry.computeBoundingBox();const b=mesh.geometry.boundingBox!;
            const projected=[[b.min.x,b.min.y],[b.min.x,b.max.y],[b.max.x,b.min.y],[b.max.x,b.max.y]].map(([x,y])=>mesh.localToWorld(new T.Vector3(x,y,0)).project(this.camera));
            const rect=new T.Box2().setFromPoints(projected.map(p=>new T.Vector2(p.x,p.y))).expandByScalar(.006);
            const blocked=shown.some(r=>r.intersectsBox(rect))||occupied.some(r=>r.intersectsBox(rect));
            // Hide an obscured caption at its own anchor; do not move it to another structure.
            const normal=new T.Vector3(0,0,1).transformDirection(card.group.matrixWorld);
            const facing=normal.dot(this.camera.position.clone().sub(card.group.getWorldPosition(new T.Vector3())))>0;
            card.group.visible=facing&&projected.every(p=>p.z>=-1&&p.z<=1)&&!blocked;
            if(card.group.visible)shown.push(rect);
            (mesh.material as T.MeshBasicMaterial).opacity=card.owner===chosen?1:.9;
        }
        for(const [id,outline] of this.planeOutlines){
            const p=this.planes.get(id)!,highlight=!!chosen&&p.owner===chosen;
            const m=outline.material as T.LineBasicMaterial;m.opacity=highlight?.95:.48;
            m.color.set(highlight?'#effaff':p.tensor.role==='parameter'?'#bc986c':'#65b4c7');
        }
        this.renderer.render(this.scene, this.camera);
        if (this.clock > .25 || !this.playing) {
            this.clock = 0;
            this.options.onStats?.(this.getStats());
        }
    };
    getStats(): ViewerStats { return {  gaussian: this.gaussianMotion?.state, receptiveField: this.fieldMotion?.state, attention: this.attentionMotion?.state, residuals: this.residualRoutes.length, mechanism: this.mechanism?.state, drawCalls: this.renderer.info.render.calls, triangles: this.renderer.info.render.triangles, cells: this.cells, connections: this.connectionCount, active: this.active, visibleNodes: this.nodes.length, totalNodes: this.scopedCount, pageStart: this.pageStart, partialTensors: [...this.planes.values()].filter(p => p.partial).length, unknownTensors: [...this.planes.values()].filter(p => !p.tensor.data).length, paused: !this.playing, weight: this.weightState, sample: this.sampleState }; }
    dispose() {
        if (this.disposed)
            return;
        this.disposed = true;
        this.windowRequest?.abort();
        cancelAnimationFrame(this.frame);
        this.observer.disconnect();
        this.visibility.disconnect();
        this.reduced.removeEventListener('change', this.onReduced);
        const c = this.renderer.domElement;
        c.removeEventListener('pointermove', this.onMove);
        c.removeEventListener('pointerleave', this.onLeave);
        c.removeEventListener('pointerdown', this.onDown);
        c.removeEventListener('pointerup', this.onUp);
        c.removeEventListener('webglcontextlost', this.onLost);
        c.removeEventListener('webglcontextrestored', this.onRestored);
        this.controls.removeEventListener('change', this.onControlsChange);
        this.controls.removeEventListener('start',this.stopFlight);
        this.controls.dispose();
        this.cleanup();
        this.renderer.dispose();
        c.remove();
    }
}
