import type {Model,Operation} from './model.js';
import {semanticStages} from './semanticGroups.js';
import {operatorVisual} from './operators.js';

/** Visual membership only: the captured graph, IDs and arithmetic stay intact. */
export interface Assembly {
    id:string;
    title:string;
    nodes:string[];
    owned:string[];
    inputs:string[];
    related:string[];
    color:string;
}
export function modelAssemblies(model:Model,nodes:Operation[]=model.nodes):Assembly[]{
    const visible=new Map(nodes.map(n=>[n.id,n]));
    const tensors=new Map(model.tensors.map(t=>[t.id,t]));
    const leafIds=(n:Operation):string[]=>visible.has(n.id)?[n.id]:((n.attrs?.children as Operation[]|undefined)??[]).flatMap(leafIds);
    const grouped=model.architecture?nodes:semanticStages(model,nodes);
    return grouped.map(unit=>{
        const ids=[...new Set(leafIds(unit))],children=ids.map(id=>visible.get(id)!);
        const owned=new Set(children.flatMap(n=>n.outputs));
        for(const n of children){
            Object.values(n.parameters??{}).forEach(id=>owned.add(id));
            n.inputs.filter(id=>['parameter','buffer','constant'].includes(tensors.get(id)?.role??'')).forEach(id=>owned.add(id));
        }
        const inputs=[...new Set(children.flatMap(n=>n.inputs).filter(id=>!owned.has(id)))];
        const visual=operatorVisual(unit.op);
        return {id:unit.id,title:String(unit.attrs?.displayLabel??(unit.op==='semantic'?unit.name:visual.label)),nodes:ids,owned:[...owned],inputs,related:[...new Set([...owned,...inputs])],color:visual.color};
    }).filter(a=>a.nodes.length);
}
