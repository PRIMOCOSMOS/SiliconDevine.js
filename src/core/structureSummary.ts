import type {Operation} from './model.js';
/** Parallel branches occupy one explanatory row; dependency depths stay ordered. */
export function structureSummary(nodes:Operation[]):string[]{
    const producers=new Map(nodes.flatMap(n=>n.outputs.map(t=>[t,n.id] as const))),levels=new Map<string,number>();
    for(const n of nodes)levels.set(n.id,1+Math.max(0,...n.inputs.map(t=>levels.get(producers.get(t)??'')??0)));
    const name=(n:Operation)=>{
        const kind=String(n.attrs?.kind??'');
        if(kind==='norm')return n.name.includes('RMS')?'RMSNorm':'归一化';
        if(kind==='merge')return '残差汇合';
        if(kind==='experts')return 'MoE 专家';
        if(kind==='ffn')return 'SwiGLU';
        return n.name.replace(/ · .*/,'').replace('分组查询注意力','GQA').replace('多头潜在注意力','MLA');
    };
    return [...new Set(levels.values())].sort((a,b)=>a-b).map(l=>[...new Set(nodes.filter(n=>levels.get(n.id)===l).map(name))].join(' / '));
}
