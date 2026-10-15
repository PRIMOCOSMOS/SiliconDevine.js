import {type Model,type Operation,valueAt} from './model.js';
import {contract} from './semanticGroups.js';

/** Recover the RMS mathematical definition, independently of Python class names. */
export function normalizationView(model:Model,original:Operation[]):Operation[]{
    let nodes=original;
    const tensors=new Map(model.tensors.map(t=>[t.id,t]));
    for(const inverse of original.filter(n=>n.op==='rsqrt')){
        const made=new Map(nodes.flatMap(n=>n.outputs.map(t=>[t,n] as const)));
        const offset=made.get(inverse.inputs[0]);if(offset?.op!=='add')continue;
        const mean=offset.inputs.map(id=>made.get(id)).find(n=>n?.op==='mean');if(!mean)continue;
        const square=made.get(mean.inputs[0]);if(!square)continue;
        if(!['multiply','power'].includes(square.op))continue;
        const x=square.inputs[0],shape=tensors.get(x)?.shape,dims=mean.attrs?.dims as number[]|undefined;
        if(!shape?.length||dims?.length!==1||(dims[0]+shape.length)%shape.length!==shape.length-1||!mean.attrs?.keepdim)continue;
        const power=tensors.get(square.inputs[1]),exponent=Number((square.attrs?.scalarOperands as Record<string,number>|undefined)?.['1']??(power?valueAt(power,0):NaN));
        if(!(square.op==='multiply'&&square.inputs[0]===square.inputs[1])&&!(square.op==='power'&&exponent===2))continue;
        const normalized=nodes.find(n=>n.op==='multiply'&&n.inputs.includes(x)&&n.inputs.includes(inverse.outputs[0]));
        if(!normalized)continue;
        let tail=normalized;
        const weighted=nodes.find(n=>n.op==='multiply'&&n.inputs.includes(normalized.outputs[0])&&n.inputs.some(id=>tensors.get(id)?.role==='parameter'));
        const gamma=weighted?.inputs.find(id=>tensors.get(id)?.role==='parameter');
        if(weighted&&gamma&&JSON.stringify(tensors.get(gamma)?.shape)===JSON.stringify(shape.slice(-1)))tail=weighted;
        const children=[square,mean,offset,inverse,normalized,...tail===normalized?[]:[tail]],ids=new Set(children.map(n=>n.id));
        if(children.slice(0,-1).some(n=>n.outputs.some(id=>model.outputs.includes(id)||nodes.some(other=>!ids.has(other.id)&&other.inputs.includes(id)))))continue;
        const epsilonId=offset.inputs.find(id=>!mean.outputs.includes(id)),epsilonTensor=epsilonId?tensors.get(epsilonId):undefined;
        if(epsilonTensor?.shape.some(d=>d!==1)&&epsilonTensor.shape.length)continue;
        const eps=epsilonTensor?.role==='constant'?Number((offset.attrs?.scalarOperands as Record<string,number>|undefined)?.[String(offset.inputs.indexOf(epsilonId!))]??valueAt(epsilonTensor,0)):NaN;
        if(!Number.isFinite(eps)||eps<0)continue;
        const before=nodes;
        nodes=contract(model,nodes,children,'RMS 归一化','semantic',{
            displayLabel:'RMS 归一化',designKind:'rms_normalization',normalized_shape:shape.slice(-1),eps,
            presentationChildren:children,children,motionKind:'rmsnorm',
            description:'按向量的均方根归一化，保留通道方向。'+(tail===weighted?'γ 调节各通道。':'')+'平方、均值和数值稳定项归入此数学定义。',
            formula:(tail===weighted?'y=\\gamma\\odot x':'y=x')+'/\\sqrt{\\operatorname{mean}(x^2)+\\epsilon}',
        });
        if(nodes===before)continue;
        const unit=nodes.find(n=>n.id===`semantic:${square.id}`)!;unit.op='rmsnorm';
        if(tail===weighted&&gamma){unit.parameters={weight:gamma};unit.inputs=unit.inputs.filter(id=>id!==gamma);}
    }
    return nodes;
}
