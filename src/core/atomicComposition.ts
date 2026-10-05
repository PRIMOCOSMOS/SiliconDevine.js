import { topologicalNodes, type Model, type Operation } from './model.js';
import { operatorVisual } from './operators.js';
import { moduleView } from './hierarchy.js';

/** One vocabulary for captured graphs, plugins and bounded LLM teaching graphs.
 * No class-name guessing, fabricated tensors, or replacement of captured values. */
export function atomicComposition(model: Model) {
    const nodes = topologicalNodes(model);
    const counts = new Map<string, number>();
    for (const n of nodes) counts.set(n.op, (counts.get(n.op) ?? 0) + 1);
    return {
        model,
        basis: model.producer?.backend === 'pytorch-mechanism' ? 'demonstration' as const : 'captured' as const,
        atoms: nodes.map(n => ({ id: n.id, op: n.op, label: operatorVisual(n.op).label,
            exact: operatorVisual(n.op).detail === 'exact', inputs: n.inputs, outputs: n.outputs,
            parameters: n.parameters ?? {}, source: n.source })),
        vocabulary: [...counts].map(([op, count]) => ({ op, count, label: operatorVisual(op).label })),
        boundaries: nodes.filter(n => operatorVisual(n.op).detail !== 'exact').map(n => n.id),
    };
}

/** Explicit references only: an unknown composite must never become a Linear. */
export function resolveAtomicComposition(model: Model, node?: Operation) {
    if (!model.architecture) return atomicComposition(model);
    const a = model.architecture;
    const scope = node?.attrs?.scopeRef ? a.scopes[String(node.attrs.scopeRef)] : undefined;
    if(scope?.executionRef){
        const full=a.mechanisms?.[scope.executionRef];
        if(!full)return;
        const nodes=moduleView(full,scope.executionScope??'',true),produced=new Set(nodes.flatMap(n=>n.outputs));
        const needed=new Set(nodes.flatMap(n=>[...n.inputs,...n.outputs,...Object.values(n.parameters??{})]));
        const external=new Set(full.nodes.filter(n=>!nodes.includes(n)).flatMap(n=>n.inputs).concat(full.outputs));
        return atomicComposition({...full,nodes,tensors:full.tensors.filter(t=>needed.has(t.id)),
            inputs:[...needed].filter(id=>!produced.has(id)&&full.tensors.find(t=>t.id===id)?.role!=='parameter'),
            outputs:[...produced].filter(id=>external.has(id))});
    }
    const ref = node?.attrs?.mechanismRef ?? scope?.mechanismRef ?? (!node ? model.mechanismRef : undefined);
    const graph = ref ? a.mechanisms?.[String(ref)] : undefined;
    return graph ? atomicComposition(graph) : undefined;
}

/** A compact DAG of reusable units. Repetition is an instance count, never an
 * instruction to allocate millions of neurons or to share independent weights. */
export function compositionCatalog(model:Model) {
    if(!model.architecture)return {entry:'model',units:[{id:'model',name:model.name,atoms:atomicComposition(model),children:[]}]};
    const a=model.architecture;
    return {entry:a.entry,units:Object.entries(a.scopes).map(([id,scope])=>({
        id,name:scope.name,
        atoms:scope.mechanismRef&&a.mechanisms?.[scope.mechanismRef]?atomicComposition(a.mechanisms[scope.mechanismRef]):undefined,
        children:scope.nodes.map(node=>({id:node.id,name:node.name,scope:node.attrs?.scopeRef,
            repeat:Number(node.attrs?.repeat??1),logicalInputs:node.inputs.map(key=>scope.tensors.find(t=>t.id===key)!),
            logicalOutputs:node.outputs.map(key=>scope.tensors.find(t=>t.id===key)!),
            atoms:resolveAtomicComposition({...model,...scope},node)})),
    }))};
}
