import {topologicalNodes,type Model,type Operation} from './model.js';
import {contract} from './semanticGroups.js';
import {operatorVisual} from './operators.js';

/** Engineering work remains in dependency provenance, never a design layer. */
export function omitEngineering(model:Model, original:Operation[]):Operation[]{
    let nodes=original;
    const tensors=new Map(model.tensors.map(t=>[t.id,t]));
    const inner=(n:Operation):Operation[]=>(n.attrs?.children as Operation[]|undefined)??[n];
    const isPreparation=(n:Operation):boolean=>{
        if(n.attrs?.engineeringAbsorbed||['projection','gaussian_distribution'].includes(String(n.attrs?.designKind)))return false;
        if(['valid_selection','engineering_preparation'].includes(String(n.attrs?.designKind)))return true;
        // Index/top-k are model routing. Boolean predicate preparation is not.
        if(n.outputs.every(t=>tensors.get(t)?.dtype==='bool')&&!model.outputs.some(t=>n.outputs.includes(t)))return true;
        if(['cast','identity','routing_not','routing_masked_fill','broadcast_tensors'].includes(n.op))return true;
        if(n.op==='symbolic'&&['cast','identity','routing_not','routing_masked_fill','reshape','permute','transpose','expand'].includes(String(n.attrs?.knownOperator)))return true;
        if(n.op==='layout')return inner(n).every(isPreparation);
        if(n.op==='semantic')return inner(n).length>1&&inner(n).every(isPreparation);
        if(['reshape','transpose','permute','expand'].includes(n.op))return true;
        return false;
    };
    // Source-time guards are not forward outputs. Scoped external ports remain.
    const rawIds=new Set(original.flatMap(function leaves(n):string[]{return n.attrs?.children?(n.attrs.children as Operation[]).flatMap(leaves):[n.id];}));
    const needed=new Set([...model.outputs,...model.nodes.filter(n=>!rawIds.has(n.id)).flatMap(n=>n.inputs)]);
    const live=new Set<string>();
    for(const n of [...topologicalNodes({...model,nodes})].reverse())if(n.outputs.some(t=>needed.has(t))){live.add(n.id);n.inputs.forEach(t=>needed.add(t));}
    nodes=nodes.filter(n=>live.has(n.id));
    // Contract into the mathematical consumer first, retaining its visual
    // operator, parameters, and exact coordinate dependency maps.
    let changed=true;
    while(changed){changed=false;
        for(const prep of nodes.filter(isPreparation)){
            const consumers=nodes.filter(n=>n.id!==prep.id&&n.inputs.some(t=>prep.outputs.includes(t)));
            const targets=consumers.filter(n=>!isPreparation(n));
            for(const target of targets){
                const label=String(target.attrs?.displayLabel??(target.op==='semantic'?target.name:operatorVisual(target.op).label));
                const before=nodes;
                nodes=contract(model,nodes,[prep,target],label,'semantic',{
                    ...target.attrs,children:[prep,target],presentationChildren:[prep,target],
                    engineeringAbsorbed:true,displayLabel:label,
                    inputConditions:[...target.attrs?.inputConditions as string[]??[],...prep.attrs?.inputConditions as string[]??[],
                        ...(prep.op==='routing_masked_fill'||prep.attrs?.designKind==='valid_selection'?['仅有效数据参与计算；按源码指定值处理缺失位置。']:[])],
                    description:target.attrs?.description,
                });
                if(nodes===before)continue;
                const unit=nodes.find(n=>n.id===`semantic:${prep.id}`)!;
                unit.op=target.op;unit.parameters=target.parameters;
                // Preserve operand order of the mathematical operation, then
                // append its conditions. K preparation must not become Q.
                const ordered=[...new Set(target.inputs.flatMap(id=>prep.outputs.includes(id)?prep.inputs:[id]))];
                unit.inputs=[...ordered.filter(id=>unit.inputs.includes(id)),...unit.inputs.filter(id=>!ordered.includes(id))]
                    .filter(t=>!Object.values(unit.parameters??{}).includes(t));
                changed=true;break;
            }
            if(changed)break;
            // Preparation-only chains collapse before reaching their consumer.
            if(consumers.length===1&&isPreparation(consumers[0])){
                const result=contract(model,nodes,[prep,consumers[0]],'输入准备','semantic',{
                    designKind:'engineering_preparation',children:[prep,consumers[0]],
                    inputConditions:[...prep.attrs?.inputConditions as string[]??[],...consumers[0].attrs?.inputConditions as string[]??[],
                        ...([prep,consumers[0]].some(n=>n.op==='routing_masked_fill'||n.attrs?.designKind==='valid_selection')?['仅有效数据参与计算；按源码指定值处理缺失位置。']:[])],
                });
                if(result!==nodes){nodes=result;changed=true;break;}
            }
        }
    }
    // A scope may end at a prepared port consumed elsewhere. Present the exact
    // effective input there, with its source/constraint, without a fake operator.
    const result=nodes.flatMap(n=>{
        if(!isPreparation(n)){
            if(n.op==='layout')return [{...n,op:'flatten',name:'特征向量化',attrs:{...n.attrs,
                displayLabel:'特征向量化',presentationChildren:n.attrs?.children,
                description:'将空间或结构化特征组织为特征向量，交给后续数学模块；保留真实坐标对应关系。'}}];
            return [n];
        }
        if(n.outputs.some(t=>model.outputs.includes(t))){
            // A terminal mask/transpose really defines the returned function.
            const label=n.op==='routing_masked_fill'||n.attrs?.designKind==='valid_selection'?'条件约束后的输出':'输出表示';
            return [{...n,name:label,attrs:{...n.attrs,displayLabel:label,engineeringBoundary:true}}];
        }
        return [];
    });
    // A shared prepared predicate is a condition, not another output feature.
    // Keep it only when the visible math actually exposes that control port.
    return result.map(n=>({...n,outputs:n.outputs.filter(id=>tensors.get(id)?.dtype!=='bool'||model.outputs.includes(id)||
        result.some(other=>other.id!==n.id&&other.inputs.includes(id)))})).filter(n=>n.outputs.length);
}

/** Name effective ports from the original inputs, without substituting values. */
export function effectiveInputLabels(model:Model,nodes:Operation[]):Map<string,string>{
    const made=new Map(model.nodes.flatMap(n=>n.outputs.map(t=>[t,n] as const))),
        tensors=new Map(model.tensors.map(t=>[t.id,t])), produced=new Set(nodes.flatMap(n=>n.outputs));
    const labels=new Map<string,string>();
    function roots(id:string,seen=new Set<string>()):string[]{
        if(seen.has(id))return [];seen.add(id);
        const t=tensors.get(id);if(t?.role==='input')return [t.semantic??id];
        const n=made.get(id);if(!n)return [];
        return n.inputs.flatMap(t=>roots(t,seen));
    }
    for(const id of nodes.flatMap(n=>n.inputs))if(!produced.has(id)&&!model.inputs.includes(id)){
        const n=made.get(id),t=tensors.get(id);
        if(!n||['parameter','buffer','constant'].includes(t?.role??''))continue;
        const source=[...new Set(roots(id))];
        if(source.length&&['routing_masked_fill','cast','identity','reshape','transpose','permute','expand'].includes(n.op))
            labels.set(id,source.length===1?`${source[0]} · 有效输入`:source.join(' / ')+' · 有效输入');
    }
    return labels;
}
