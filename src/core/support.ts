import type {Model} from './model.js';
import {operatorVisual} from './operators.js';
/** Uses the live registry, including user plugins; a recognized name alone is insufficient. */
export function inspectSupport(model:Model){
    const boundary=model.nodes.filter(n=>operatorVisual(n.op).detail==='boundary').map(n=>({id:n.id,op:n.op,source:n.source??n.name,reason:String(n.attrs?.boundary_reason??'保留真实输入输出；尚无内部标量依赖插件。')}));
    return {total:model.nodes.length,withDependencies:model.nodes.length-boundary.length,boundary,
        mode:model.architecture?'structure':'numeric',producer:model.producer,
        notes:['数值窗口外的元素保持未知；坐标依赖支持不等同于整张张量已加载。','捕获 eval 前向路径；训练、反向传播和未执行分支不在本次图内。']};
}
