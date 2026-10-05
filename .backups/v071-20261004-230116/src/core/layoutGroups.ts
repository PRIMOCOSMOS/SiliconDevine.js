import { topologicalNodes, type Model, type Operation } from './model.js';
/** Collapse connected coordinate-only operations, never arithmetic or learned transforms. */
export function groupCoordinateChanges(model: Model, nodes: Operation[]): Operation[] {
    const eligible = new Map(nodes.filter(n => ['reshape', 'flatten', 'transpose', 'permute', 'identity', 'chunk', 'split', 'concat', 'expand'].includes(n.op)).map(n => [n.id, n]));
    const producer = new Map(nodes.flatMap(n => n.outputs.map(id => [id, n.id] as const))), neighbors = new Map<string, Set<string>>();
    for (const n of eligible.values())
        for (const id of n.inputs) {
            const p = producer.get(id);
            if (p && eligible.has(p)) {
                if (!neighbors.has(p))
                    neighbors.set(p, new Set());
                if (!neighbors.has(n.id))
                    neighbors.set(n.id, new Set());
                neighbors.get(p)!.add(n.id);
                neighbors.get(n.id)!.add(p);
            }
        }
    const seen = new Set<string>(), replacements = new Map<string, Operation>();
    for (const first of eligible.values()) {
        if (seen.has(first.id))
            continue;
        const component: string[] = [], queue = [first.id];
        while (queue.length) {
            const id = queue.pop()!;
            if (seen.has(id))
                continue;
            seen.add(id);
            component.push(id);
            queue.push(...neighbors.get(id) ?? []);
        }
        if (component.length < 2)
            continue;
        const ids = new Set(component), children = nodes.filter(n => ids.has(n.id)), produced = new Set(children.flatMap(n => n.outputs)), consumed = new Set([...nodes, ...model.nodes].filter(n => !ids.has(n.id)).flatMap(n => [...n.inputs, ...Object.values(n.parameters ?? {})]));
        model.outputs.forEach(id => consumed.add(id));
        const outputs = [...produced].filter(id => consumed.has(id));
        if (!outputs.length)
            outputs.push(...children.at(-1)!.outputs);
        const node: Operation = { id: 'layout:' + first.id, name: children.some(n => ['chunk', 'split', 'concat', 'expand'].includes(n.op)) ? '分区与坐标组织' : '坐标重排', op: 'layout', group: first.group, inputs: [...new Set(children.flatMap(n => n.inputs).filter(id => !produced.has(id)))], outputs, attrs: { children }, source: 'Captured coordinate-only operations' };
        component.forEach(id => replacements.set(id, node));
    }
    const emitted = new Set<string>(), result: Operation[] = [];
    for (const n of nodes) {
        const r = replacements.get(n.id) ?? n;
        if (!emitted.has(r.id)) {
            result.push(r);
            emitted.add(r.id);
        }
    }
    try {
        return topologicalNodes({ ...model, nodes: result });
    }
    catch {
        return nodes;
    }
}
