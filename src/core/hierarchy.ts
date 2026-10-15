import { topologicalNodes, type Model, type Operation } from './model.js';
import { groupCoordinateChanges } from './layoutGroups.js';
import { contract,semanticStages } from './semanticGroups.js';
import {mathematicalView} from './designView.js';
/** Collapse native module boundaries; visible tensors keep their original IDs and data. */
export function moduleView(model: Model, scope = '', detail:boolean|'mathematical' = false): Operation[] {
    if (model.composition && (!scope || scope.startsWith('@source.'))) {
        const view = compositionView(model, scope, detail);
        if (view) return view;
    }
    const within = (path: string, parent: string) => !parent || path === parent || path.startsWith(parent + '.');
    const nodes = topologicalNodes(model).filter(n => within(n.group ?? n.id, scope));
    if (detail) return detail==='mathematical'?mathematicalView(model,nodes):nodes;
    if (!model.modules) return mathematicalView(model,nodes);
    const roots: string[] = [];
    for (const m of [...model.modules].sort((a, b) => a.path.split('.').length - b.path.split('.').length)) {
        if (m.path === scope || !within(m.path, scope) || roots.some(p => within(m.path, p)))
            continue;
        if (nodes.filter(n => within(n.group ?? n.id, m.path)).length > 1)
            roots.push(m.path);
    }
    const groups = new Map<string, Operation[]>();
    for (const n of nodes) {
        const path = roots.find(p => within(n.group ?? n.id, p));
        if (path)
            groups.set(path, [...groups.get(path) ?? [], n]);
    }
    const replacements = new Map<string, Operation>();
    for (const [path, children] of groups) {
        const childIds = new Set(children.map(n => n.id)), produced = new Set(children.flatMap(n => n.outputs));
        const external = new Set(model.nodes.filter(n => !childIds.has(n.id)).flatMap(n => [...n.inputs, ...Object.values(n.parameters ?? {})]));
        model.outputs.forEach(id => external.add(id));
        const outputs = [...produced].filter(id => external.has(id));
        if (!outputs.length)
            outputs.push(...children.at(-1)!.outputs);
        const inputs = [...new Set(children.flatMap(n => n.inputs).filter(id => !produced.has(id) && !['parameter', 'buffer'].includes(model.tensors.find(t => t.id === id)!.role)))];
        const type = model.modules.find(m => m.path === path)!.type;
        replacements.set(path, { id: 'module:' + path, name: path, group: path, op: 'module', inputs, outputs, parameters: {}, attrs: { modulePath: path, moduleType: type, operatorCount: children.length, operations: [...new Set(children.map(n => n.op))] }, source: type });
    }
    const result: Operation[] = [], seen = new Set<string>();
    for (const n of nodes) {
        const path = roots.find(p => within(n.group ?? n.id, p));
        if (!path)
            result.push(n);
        else if (!seen.has(path)) {
            result.push(replacements.get(path)!);
            seen.add(path);
        }
    }
    try {
        return mathematicalView(model,groupGaussianSampling(model,topologicalNodes({ ...model, nodes: result })));
    }
    catch {
        return nodes;
    }
}

/** Each visible unit is a partition of recorded operations, with exact tensor ports. */
function compositionView(model: Model, scope: string, detail: boolean|'mathematical'): Operation[] | undefined {
    const composition = model.composition!, units = new Map(composition.units.map(u => [u.id,u]));
    const parent = units.get(scope);
    if (scope && !parent) return;
    const members = parent ? new Set(parent.nodes) : undefined;
    const raw = model.nodes.filter(n => !members || members.has(n.id));
    // A probability distribution is a mathematical object even when its
    // implementation only broadcasts parameters. Preserve the source proof.
    if(detail!==true&&parent?.visual?.kind==='gaussian_parameters'){
        const d=parent.visual.distribution as {mean:string;std:string};
        return mathematicalView(model,contract(model,raw,raw,'高斯分布 · 均值与方差','semantic',{
            designKind:'gaussian_distribution',displayLabel:'高斯分布 · 均值与方差',
            description:parent.visual.description,formula:parent.visual.formula,
            compositionVisual:parent.visual,motionKind:'gating',
            portLabels:{[d.mean]:'均值 μ',[d.std]:'标准差 σ · 方差 = σ²'},
        }));
    }
    if (detail) return detail==='mathematical'?mathematicalView(model,topologicalNodes({...model,nodes:raw})):topologicalNodes({...model,nodes:raw});
    const keys = parent ? parent.children : composition.roots;
    if (!keys.length) return mathematicalView(model,topologicalNodes({...model,nodes:raw}));
    const tensors = new Map(model.tensors.map(t => [t.id,t]));
    const nodeMap = new Map(model.nodes.map(n => [n.id,n]));
    const consumers = new Map<string,Set<string>>();
    const liveIds=new Set(composition.roots.flatMap(id=>units.get(id)?.nodes??[]));
    for (const n of model.nodes.filter(n=>liveIds.has(n.id))) for (const id of [...n.inputs,...Object.values(n.parameters??{})]) {
        if (!consumers.has(id)) consumers.set(id,new Set());
        consumers.get(id)!.add(n.id);
    }
    const outputs = new Set(model.outputs);
    const result = keys.flatMap(key => {
        const u=units.get(key); if(!u)return [];
        const children=u.nodes.map(id=>nodeMap.get(id)).filter((n):n is Operation=>!!n);
        if(children.length===1) return children;
        const ids=new Set(u.nodes),produced=new Set(children.flatMap(n=>n.outputs));
        const inputs=[...new Set(children.flatMap(n=>[...n.inputs,...Object.values(n.parameters??{})]).filter(id=>!produced.has(id)&&!['parameter','buffer','constant'].includes(tensors.get(id)?.role??'')))];
        const out=[...produced].filter(id=>outputs.has(id)||[...consumers.get(id)??[]].some(id=>!ids.has(id)));
        const stages=(u.visual?.stages??[]) as {tensor:string}[];
        // Expose actual block activations as overview tower floors, never glyphs
        // standing in for invented data. These IDs belong to this exact subgraph.
        for(const s of stages)if(produced.has(s.tensor)&&!out.includes(s.tensor))out.push(s.tensor);
        if(!out.length)return [];
        return [{id:u.id,name:u.name,op:'module',group:u.id,inputs,outputs:out,parameters:{},
            attrs:{modulePath:u.id,moduleType:u.name,operatorCount:children.length,displayLabel:u.name,
                composition:true,sourceScope:u.source,compositionVisual:u.visual,
                portLabels:u.visual?.distribution?{
                    [(u.visual.distribution as {mean:string}).mean]:'均值 μ',
                    [(u.visual.distribution as {std:string}).std]:'标准差 σ · 方差 = σ²'
                }:undefined,
                description:u.visual?.description,formula:u.visual?.formula,
                operations:[...new Set(children.map(n=>n.op))]},source:composition.evidence} satisfies Operation];
    });
    return mathematicalView(model,topologicalNodes({...model,nodes:result}));
}
/** Sampling is one functional unit in architecture mode; raw arithmetic stays reachable. */
function groupGaussianSampling(model: Model,nodes: Operation[]): Operation[] {
    const replacement=new Map<string,Operation>();
    for(const unit of model.functionalUnits??[]) {
        if(unit.kind!=='gaussian_reparameterization')continue;
        const ids=unit.children as string[], children=nodes.filter(n=>ids.includes(n.id));
        if(children.length!==ids.length)continue;
        const produced=new Set(children.flatMap(n=>n.outputs));
        const external=new Set([...model.outputs,...model.nodes.filter(n=>!ids.includes(n.id)).flatMap(n=>n.inputs)]);
        const outputs=[...produced].filter(t=>external.has(t));
        if(outputs.length!==1||outputs[0]!==unit.sample)continue;
        const node:Operation={id:'gaussian:'+unit.id,name:'高斯重参数采样',op:'gaussian_sample',group:children.at(-1)!.group,
            inputs:[unit.mean as string,unit.logvar as string],outputs,parameters:{},attrs:{children,gaussian:unit},source:'recognized actual PyTorch arithmetic'};
        children.forEach(n=>replacement.set(n.id,node));
    }
    const seen=new Set<string>();return nodes.flatMap(n=>{const node=replacement.get(n.id)??n;if(seen.has(node.id))return [];seen.add(node.id);return [node];});
}
