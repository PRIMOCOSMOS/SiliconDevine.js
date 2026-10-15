import {topologicalNodes,type Model,type Operation} from './model.js';
import {contract,semanticStages} from './semanticGroups.js';
import {groupCoordinateChanges} from './layoutGroups.js';
import {omitEngineering} from './engineeringView.js';
import {normalizationView} from './normalizationView.js';

/** Mathematical presentation for architecture designers; raw graph stays inspectable. */
export function mathematicalView(model:Model,original:Operation[]):Operation[]{
    let nodes=normalizationView(model,original);
    const tensors=new Map(model.tensors.map(t=>[t.id,t]));
    for(const sum of [...nodes].filter(n=>n.op==='sum')){
        const product=nodes.find(n=>n.op==='multiply'&&n.outputs.includes(sum.inputs[0]));
        if(!product||product.inputs.length!==2)continue;
        const a=tensors.get(product.inputs[0]),b=tensors.get(product.inputs[1]),p=tensors.get(product.outputs[0]);
        if(!a||!b||!p||!p.shape.length||JSON.stringify(a.shape)!==JSON.stringify(b.shape)||JSON.stringify(a.shape)!==JSON.stringify(p.shape))continue;
        const dims=sum.attrs?.dims as number[]|undefined;
        if(!dims||dims.length!==1||((dims[0]+p.shape.length)%p.shape.length)!==p.shape.length-1)continue;
        // Products reused elsewhere retain a separate port; never hide their other purpose.
        if(nodes.some(n=>n.id!==sum.id&&n.inputs.some(t=>product.outputs.includes(t)))||product.outputs.some(t=>model.outputs.includes(t)))continue;
        const self=product.inputs[0]===product.inputs[1];
        nodes=contract(model,nodes,[product,sum],self?'平方范数':'向量点积','semantic',{
            designKind:self?'squared_norm':'dot_product',motionKind:'sum',
            displayLabel:self?'平方范数':'向量点积',
            formula:self?'s=\\lVert x\\rVert_2^2':'s=\\langle x,y\\rangle=\\sum_i x_i y_i',
            description:'沿最后一个向量轴相乘并求和，保留其余批次轴。根据乘法与归约的实际依赖识别。',
        });
    }
    // Helper functions form meaningful code boundaries, including imported files.
    for(const unit of model.designUnits??[]){
        const children=nodes.filter(n=>unit.nodes.includes(n.id));
        if(children.length!==unit.nodes.length)continue;
        nodes=contract(model,nodes,children,unit.name,'semantic',{
            designKind:unit.kind,displayLabel:unit.name,description:unit.description,
            sourceExcerpt:{...unit.source,code:unit.source.code??unit.description},motionKind:'gating',sourceEvidence:unit.evidence,
        });
    }
    nodes=semanticStages(model,nodes);
    // Collect mask preparation by its real shared boolean input, independently
    // of variable/class names. Float features are never absorbed upstream.
    const producer=()=>new Map(nodes.flatMap(n=>n.outputs.map(t=>[t,n] as const)));
    const maskFamilies=new Map<string,Set<Operation>>();
    for(const fill of nodes.filter(n=>n.op==='routing_masked_fill')){
        const made=producer(),family=new Set<Operation>([fill]);
        let input=fill.inputs[1];
        while(input){
            const n=made.get(input);
            if(!n||!['routing_not','reshape','identity','cast','expand','slice','select'].includes(n.op)||n.outputs.some(t=>tensors.get(t)?.dtype!=='bool'))break;
            // A shared validity predicate does not put otherwise independent
            // data preparations in the same mathematical function.
            if(nodes.some(other=>!family.has(other)&&other.id!==n.id&&other.inputs.some(t=>n.outputs.includes(t))))break;
            family.add(n);input=n.inputs[0];
        }
        const key=fill.id,existing=maskFamilies.get(key)??new Set<Operation>();
        family.forEach(n=>existing.add(n));maskFamilies.set(key,existing);
    }
    for(const children of maskFamilies.values())nodes=contract(model,nodes,[...children],'有效元素选择','semantic',{
        designKind:'valid_selection',displayLabel:'有效元素选择',motionKind:'gating',
        description:'根据有效性掩码选取参与计算的数据；屏蔽位置使用源码指定的值。掩码取反和维度对齐归入此功能。',
    });
    nodes=groupCoordinateChanges(model,nodes);
    // Coordinate preparation belongs to a projected vector, not another design layer.
    for(const layout of [...nodes].filter(n=>n.op==='layout')){
        const linear=nodes.find(n=>n.op==='linear'&&n.outputs.some(t=>layout.inputs.includes(t)));
        if(!linear||linear.outputs.some(t=>model.outputs.includes(t))||nodes.some(n=>n.id!==layout.id&&n.inputs.some(t=>linear.outputs.includes(t))))continue;
        const label=String(linear.attrs?.displayLabel??'全连接投影'),before=nodes;
        nodes=contract(model,nodes,[linear,layout],label,'semantic',{
            designKind:'projection',displayLabel:label,motionKind:'gating',
            description:'学习到的线性投影生成目标向量，并按下游矩阵运算所需形式组织通道。权重和偏置属于此投影。',
        });
        if(nodes===before)continue;
        const unit=nodes.find(n=>n.id===`semantic:${linear.id}`)!;
        unit.parameters=linear.parameters;
        unit.inputs=unit.inputs.filter(t=>!Object.values(unit.parameters??{}).includes(t));
    }
    for(const layout of [...nodes].filter(n=>n.op==='layout')){
        const parent=nodes.find(n=>n.op==='semantic'&&n.attrs?.designKind&&n.outputs.some(t=>layout.inputs.includes(t)));
        if(!parent)continue;
        const before=nodes;
        nodes=contract(model,nodes,[parent,layout],parent.name,'semantic',{
            ...parent.attrs,children:[parent,layout],displayLabel:parent.name,
        });
        if(nodes!==before&&parent.parameters){const unit=nodes.find(n=>n.id===`semantic:${parent.id}`)!;unit.parameters=parent.parameters;unit.inputs=unit.inputs.filter(t=>!Object.values(parent.parameters!).includes(t));}
    }
    nodes=nodes.map(n=>{
        if(n.op==='symbolic'){
            const names:Record<string,string>={add:'特征相加',multiply:'通道乘积',matmul:'矩阵乘法',linear:'全连接投影',getitem:'输入分量'};
            const known=String(n.attrs?.knownOperator);
            if(names[known])return {...n,attrs:{...n.attrs,displayLabel:names[known]}};
        }
        if(n.op==='add'&&!n.attrs?.residualInput&&!n.attrs?.displayLabel)return {...n,attrs:{...n.attrs,
            displayLabel:'特征相加',description:'多个特征来源相加形成输出；没有已确认的跳连接证据。',
        }};
        if(['concat','chunk','split'].includes(n.op))return {...n,attrs:{...n.attrs,
            displayLabel:n.op==='concat'?'特征拼接':'特征分组',
            description:n.op==='concat'?'沿源码指定轴拼接多个特征来源，保留各部分独立的通道与顺序。':'沿源码指定轴将特征通道分为多个分支，分别交给下游功能模块。',
        }};
        if(n.op==='multiply'&&!n.attrs?.displayLabel){
            const scalar=Object.keys(n.attrs?.scalarOperands as object??{}).length>0||n.inputs.some(t=>tensors.get(t)?.shape.length===0);
            const label=scalar?'张量缩放':'逐通道乘积';
            return {...n,attrs:{...n.attrs,displayLabel:label,description:scalar?'按源码系数缩放向量或矩阵。':'同位置通道相乘，保持通道结构。此处没有求和归约，保留通道乘积含义。'}};
        }
        if(n.op==='linear'&&!n.attrs?.displayLabel){
            const roles=[...new Set(n.attrs?.projectionRoles as string[]??[])];
            if(roles.length)return {...n,attrs:{...n.attrs,displayLabel:roles.join('/')+' 投影'}};
        }
        return n;
    });
    return topologicalNodes({...model,nodes:omitEngineering(model,nodes)});
}
