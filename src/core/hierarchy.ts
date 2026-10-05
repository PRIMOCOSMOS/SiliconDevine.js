import { topologicalNodes, type Model, type Operation } from './model.js';
import { groupCoordinateChanges } from './layoutGroups.js';
import { semanticStages } from './semanticGroups.js';
/** Collapse native module boundaries; visible tensors keep their original IDs and data. */
export function moduleView(model: Model, scope = '', detail = false): Operation[] {
    const within = (path: string, parent: string) => !parent || path === parent || path.startsWith(parent + '.');
    const nodes = topologicalNodes(model).filter(n => within(n.group ?? n.id, scope));
    if (detail) return nodes;
    if (!model.modules) return groupCoordinateChanges(model, semanticStages(model, nodes));
    const roots: string[] = [];
    for (const m of [...model.modules].sort((a, b) => a.path.split('.').length - b.path.split('.').length)) {
        if (m.path === scope || !within(m.path, scope) || roots.some(p => within(m.path, p)))
            continue;
        if (['ModuleList', 'ModuleDict', 'LlamaModel'].includes(m.type))
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
        return groupCoordinateChanges(model, groupGaussianSampling(model, semanticStages(model, topologicalNodes({ ...model, nodes: result }))));
    }
    catch {
        return nodes;
    }
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
