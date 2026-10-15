import type {Model,InputStructure} from '../src/core/model';
import {moduleView} from '../src/core/hierarchy';

/** Read-only source contract in the docked inspector; no overlay on the model. */
export function installInputContract(){
    const inspector=document.querySelector<HTMLElement>('.inspector')!;
    const section=document.createElement('section');section.id='model-inputs';section.tabIndex=-1;
    section.setAttribute('aria-label','模型输入');
    inspector.querySelector('.inspector-explanation')!.after(section);
    const button=document.createElement('button');button.id='inputs-toggle';button.textContent='模型输入';
    button.setAttribute('aria-controls','model-inputs');document.querySelector('.navigation')!.append(button);
    button.onclick=()=>{
        document.dispatchEvent(new CustomEvent('silicondevine:inspect'));
        section.scrollIntoView({block:'nearest'});section.focus({preventScroll:true});
    };
    const put=(parent:HTMLElement,tag:string,text:string,cls?:string)=>{
        const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;parent.append(e);return e;
    };
    const shape=(dims:(number|string)[]|undefined)=>dims?.length?'['+dims.join(' × ')+']':dims?'标量':'尺寸待定';
    return (model:Model)=>{
        section.replaceChildren();put(section,'h2','模型输入');
        const contract=model.inputContract;
        if(contract){
            const architecture=moduleView(model);
            put(section,'code','forward'+contract.signature,'input-signature');
            put(section,'p',model.producer?.backend==='symbolic'?'尺寸待定，按源码输入接口展示。':'下列尺寸是本次捕获的示例；符号维度保留源码名称。','input-note');
            for(const p of contract.parameters){
                const row=put(section,'div','','input-parameter'),head=put(row,'div','','input-heading');
                put(head,'strong',p.name);put(head,'span',p.required?'必需':`可选 · 默认 ${JSON.stringify(p.default)??'未指定'}`);
                if(p.kind==='keyword_only')put(row,'p','通过关键字传入');
                if(p.annotation)put(row,'p',p.annotation,'input-annotation');
                const leaves=(s:InputStructure):InputStructure[]=>s.children?.flatMap(leaves)??[s];
                for(const leaf of leaves(p.structure)){
                    const line=put(row,'div','','input-leaf');
                    if(leaf.path!==p.name)put(line,'code',leaf.path);
                    put(line,'p',leaf.kind==='tensor'?`${shape(leaf.shape)} · ${leaf.dtype}`:
                        leaf.kind==='unknown'?'形状与类型待定':leaf.kind==='none'?`${p.status==='default'?'使用默认值':'本次传入'} None`:
                        `${leaf.type??leaf.kind} · ${JSON.stringify(leaf.value)}`);
                }
                if(p.symbolicShape)put(row,'p',`源码维度 ${shape(p.symbolicShape)}`,'input-dimensions');
                if(p.shapeSource)put(row,'p',`${p.shapeSource.file}:${p.shapeSource.line}`,'input-source');
                if(p.defaultExpression){put(row,'p','未提供时由源码生成');put(row,'code',p.defaultExpression,'input-expression');}
                const refs=new Set([...(p.tensors??[]).map(t=>t.id),...p.generatedTensors??[]]);
                const uses=[...new Set(architecture.filter(n=>n.inputs.some(id=>refs.has(id))).map(n=>String(n.attrs?.displayLabel??n.name)))];
                if(uses.length)put(row,'p',`进入 ${uses.slice(0,4).join(' / ')}${uses.length>4?' 等模块':''}`,'input-usage');
                if(p.sourceUses?.length){const details=document.createElement('details');row.append(details);put(details,'summary','查看源码用途');p.sourceUses.forEach(s=>put(details,'code',s,'input-expression'));}
            }
        }else{
            put(section,'p','输入张量来自模型文件；未附带 Python 函数签名。','input-note');
            for(const id of model.inputs){const t=model.tensors.find(t=>t.id===id);if(!t)continue;
                const row=put(section,'div','','input-parameter');put(row,'strong',t.semantic??t.id);put(row,'p',`${shape(t.shape)} · ${t.dtype}`);
            }
        }
    };
}
