import { type Model, type Operation } from './model.js';
import { contract, coordinatePurpose } from './semanticGroups.js';
/** Directed, single-consumer chains cannot swallow a parallel Q/K/V branch. */
export function groupCoordinateChanges(model: Model, original: Operation[]): Operation[] {
    const coordinate = new Set(['reshape','flatten','transpose','permute','identity','chunk','split','concat','expand']);
    let nodes = original;
    const visited = new Set<string>();
    for (const first of original) {
        if (!coordinate.has(first.op) || visited.has(first.id)) continue;
        const children = [first]; visited.add(first.id);
        let tail = first;
        while (true) {
            const next = original.filter(n => n.inputs.some(id => tail.outputs.includes(id)));
            if (next.length !== 1 || !coordinate.has(next[0].op) || visited.has(next[0].id) || next[0].group !== first.group) break;
            children.push(next[0]); visited.add(next[0].id); tail = next[0];
        }
        const made = new Set(children.flatMap(n=>n.outputs)), inputs = children.flatMap(n=>n.inputs).filter(t=>!made.has(t));
        const purpose = coordinatePurpose(model, children, inputs, tail.outputs);
        nodes = contract(model, nodes, children, purpose.title, 'layout', {description:purpose.description,motionKind:purpose.motionKind});
    }
    // Fold safe coordinate forks (e.g. split -> per-head views) into one work surface.
    // A contraction is accepted only if it preserves a DAG, so RoPE joins cannot
    // accidentally absorb their own arithmetic dependency.
    let merged=true;
    while(merged){merged=false;
        for(const a of nodes.filter(n=>n.op==='layout')){
            const b=nodes.find(n=>n.op==='layout'&&n.id!==a.id&&n.group===a.group&&n.name.split(' · ')[0]===a.name.split(' · ')[0]&&n.inputs.some(id=>a.outputs.includes(id)));
            if(!b)continue;
            const result=contract(model,nodes,[a,b],a.name,'layout',{description:a.attrs?.description,motionKind:a.attrs?.motionKind==='broadcast'||b.attrs?.motionKind==='broadcast'?'broadcast':'reorder'});
            if(result!==nodes){nodes=result;merged=true;break;}
        }
    }
    return nodes;
}
