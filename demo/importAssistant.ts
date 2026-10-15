import type {LiveConnection,LiveStatus,ImportField,ImportSchema,ImportIssue} from '../src/core/liveClient';
import './importAssistant.css';

const node=<K extends keyof HTMLElementTagNameMap>(tag:K,text?:string)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
const choice=(values:{label:string;value:string}[])=>{const s=node('select');for(const v of values)s.add(new Option(v.label,v.value));return s;};
type Reader=()=>unknown;

/** Literal controls only: model code executes in the user's selected Python process. */
function editor(host:HTMLElement,f:ImportField,depth=0):Reader {
    if(depth>7){host.append(node('p','嵌套过深，请使用模型已有的配置类默认值。'));return()=>{throw Error('参数嵌套过深。');};}
    let kind=f.kind;
    if(kind==='any'||kind==='none'){
        const select=choice([{label:'选择数据类型',value:''},{label:'张量',value:'tensor'},{label:'整数',value:'integer'},{label:'小数',value:'number'},{label:'文字',value:'text'},{label:'是 / 否',value:'boolean'},{label:'列表',value:'sequence'},{label:'字典',value:'mapping'},{label:'配置对象（属性访问）',value:'namespace'},{label:'空值 None',value:'null'}]);
        const inner=node('div');host.append(select,inner);let read:Reader=()=>{throw Error(`请选择 ${f.name} 的数据类型。`);};
        select.setAttribute('aria-label',f.name+' · 数据类型');
        select.onchange=()=>{inner.replaceChildren();read=!select.value?()=>{throw Error(`请选择 ${f.name} 的数据类型。`);}:select.value==='null'?()=>null:editor(inner,{...f,kind:select.value,value:undefined},depth+1);};
        if(kind==='none'){select.value='null';read=()=>null;}
        return()=>read();
    }
    if(kind==='boolean'||kind==='choice'||kind==='enum'){
        const values=kind==='boolean'?[true,false]:f.choices??[];
        const select=choice(values.map((v,i)=>({label:kind==='boolean'?(v?'是':'否'):String(v),value:String(i)})));
        const current=kind==='enum'?(f.value as {name?:string})?.name:f.value;
        select.value=String(Math.max(0,values.indexOf(current)));host.append(select);
        return()=>kind==='enum'?{$enum:f.class,name:values[Number(select.value)]}:values[Number(select.value)];
    }
    if(kind==='tensor'){
        const wrapped=f.value as {$tensor?:unknown}|undefined;
        const value=(wrapped?.$tensor??f.value??{}) as {shape?:number[];dtype?:string;fill?:string;high?:number;values?:unknown};
        const grid=node('div');grid.className='import-tensor';
        const shape=node('input');shape.placeholder='例如 1, 3, 32, 32';shape.value=value.shape?.join(', ')??'';shape.required=true;shape.setAttribute('aria-label',f.name+' · 张量尺寸');
        const scalar=node('input');scalar.type='checkbox';scalar.checked=!!value.shape&&value.shape.length===0;const scalarLabel=node('label','标量张量（0 维）');scalarLabel.prepend(scalar);scalar.onchange=()=>shape.disabled=scalar.checked;shape.disabled=scalar.checked;
        const dtype=choice(['float32','float64','float16','bfloat16','int64','int32','bool'].map(v=>({label:v,value:v})));dtype.value=value.dtype??'float32';dtype.setAttribute('aria-label',f.name+' · 数据类型');
        const fill=choice([{label:'随机示例值',value:'random'},{label:'全部为零',value:'zeros'},{label:'全部为一',value:'ones'},{label:'填写数值',value:'values'}]);fill.value=value.values!==undefined?'values':value.fill??'random';fill.setAttribute('aria-label',f.name+' · 示例值');
        const values=node('textarea');values.rows=2;values.placeholder='按顺序填写，用逗号或空格分隔';values.setAttribute('aria-label',f.name+' · 数值');values.hidden=fill.value!=='values';
        if(value.values!==undefined)values.value=String((value.values as unknown[]).flat?.(8)??value.values);
        const high=node('input');high.type='number';high.min='1';high.step='1';high.value=String(value.high??2);high.setAttribute('aria-label',f.name+' · 随机整数上限');
        const highLabel=node('label','随机整数上限（不含）');highLabel.append(high);
        const sync=()=>{values.hidden=fill.value!=='values';highLabel.hidden=fill.value!=='random'||!dtype.value.startsWith('int');};fill.onchange=dtype.onchange=sync;sync();
        const label=node('label','尺寸');label.append(shape,scalarLabel);grid.append(label,dtype,fill,values,highLabel);host.append(grid);
        return()=>{const dims=scalar.checked?[]:shape.value.split(/[,，×x\s]+/).filter(Boolean).map(Number);
            if(!scalar.checked&&!dims.length||dims.some(d=>!Number.isInteger(d)||d<1))throw Error(`${f.name} 的尺寸须为正整数。`);
            if(dims.reduce((a,b)=>a*b,1)>2_000_000)throw Error(`${f.name} 超过 200 万元素，请减小示例尺寸。`);
            const result:Record<string,unknown>={shape:dims,dtype:dtype.value,fill:fill.value};
            if(fill.value==='values'){const vs=values.value.split(/[,，\s]+/).filter(Boolean).map(v=>v==='true'?1:v==='false'?0:Number(v));if(vs.some(v=>!Number.isFinite(v))||vs.length!==dims.reduce((a,b)=>a*b,1))throw Error(`${f.name} 的数值数量应与尺寸相符。`);result.values=vs;}
            if(dtype.value.startsWith('int')&&fill.value==='random'){const n=Number(high.value);if(!Number.isInteger(n)||n<1)throw Error('随机整数上限须大于零。');result.high=n;}
            return wrapped?.$tensor?{$tensor:result}:result;};
    }
    if(['sequence','mapping','namespace','object'].includes(kind)){
        const rows=node('div');rows.className='import-nested';host.append(rows);const readers:{key:HTMLInputElement;read:Reader;row:HTMLElement;enabled:HTMLInputElement}[]=[];
        const objectValue=(f.value??{}) as Record<string,unknown>;
        const source=kind==='object'?(objectValue.kwargs??{}):kind==='namespace'?(objectValue.$namespace??{}):objectValue;
        const children=f.children??Object.entries(source as Record<string,unknown>).map(([name,value])=>({name,value,kind:typeof value==='number'?'number':typeof value==='string'?'text':typeof value==='boolean'?'boolean':'any',required:true}));
        const add=(child:ImportField)=>{
            const row=node('fieldset');const legend=node('legend');const key=node('input');key.value=child.name;key.placeholder='字段名';key.setAttribute('aria-label','字段名');
            if(kind==='sequence'){legend.textContent='列表项';}else legend.append(key);
            const enabled=node('input');enabled.type='checkbox';enabled.checked=!!child.required||child.value!==undefined;const enableLabel=node('label','提供此字段');enableLabel.prepend(enabled);row.append(legend,enableLabel);
            const inner=node('div');row.append(inner);const read=editor(inner,child,depth+1);inner.hidden=!enabled.checked;enabled.onchange=()=>inner.hidden=!enabled.checked;
            const remove=node('button','移除此项');remove.type='button';remove.onclick=()=>{row.remove();readers.splice(readers.findIndex(r=>r.row===row),1);};row.append(remove);rows.append(row);readers.push({key,read,row,enabled});
        };
        for(const c of children)add(c);
        const more=node('button',kind==='sequence'?'＋ 添加列表项':'＋ 添加字段');more.type='button';more.onclick=()=>add({name:'',kind:'any',required:true});host.append(more);
        return()=>{const active=readers.filter(r=>r.enabled.checked);if(kind==='sequence')return active.map(r=>r.read());
            const values:Record<string,unknown>={};for(const r of active){const k=r.key.value.trim();if(!k)throw Error('请填写配置字段名称。');if(Object.hasOwn(values,k))throw Error('配置字段名称重复：'+k);Object.defineProperty(values,k,{value:r.read(),enumerable:true});}
            return kind==='object'?{$class:f.class,kwargs:values}:kind==='namespace'?{$namespace:values}:values;};
    }
    const input=node('input');input.type=['integer','number'].includes(kind)?'number':'text';input.step=kind==='integer'?'1':'any';input.required=!!f.required;input.setAttribute('aria-label',f.name);
    input.value=f.value===undefined?'':String(f.value);host.append(input);
    return()=>{if(kind==='text')return input.value;if(!input.value.trim())throw Error('请填写 '+f.name+'。');const n=Number(input.value);if(!Number.isFinite(n)||kind==='integer'&&!Number.isInteger(n))throw Error(f.name+' 需要有效的'+(kind==='integer'?'整数。':'数字。'));return n;};
}

export function installImportAssistant(getConnection:()=>LiveConnection|undefined){
    const strip=node('section');strip.id='import-notice';strip.hidden=true;strip.setAttribute('aria-live','polite');
    const summary=node('div'),title=node('strong'),message=node('p'),actions=node('div');summary.append(title,message);strip.append(summary,actions);document.body.append(strip);
    const settingsButton=node('button','导入排查');settingsButton.id='import-settings';settingsButton.hidden=true;document.querySelector('.live-panel')!.append(settingsButton);
    const dialog=node('dialog');dialog.id='import-assistant';dialog.setAttribute('aria-labelledby','import-heading');
    const head=node('div');head.className='import-heading';const heading=node('h2','补充导入信息');heading.id='import-heading';const close=node('button','关闭');close.type='button';close.onclick=()=>dialog.close();head.append(heading,close);
    const description=node('p'),body=node('div'),form=node('form'),feedback=node('p'),foot=node('div');feedback.setAttribute('role','alert');feedback.id='import-feedback';foot.className='import-actions';
    // Conditional fields validate only when enabled, including hidden optional tensors.
    form.noValidate=true;
    const submit=node('button','应用并重新捕获');submit.type='submit';const retry=node('button','直接重试');retry.type='button';foot.append(feedback,submit,retry);form.append(body,foot);dialog.append(head,description,form);document.body.append(dialog);
    let schema:ImportSchema|null|undefined,issue:ImportIssue|null|undefined,key='',read:()=>Record<string,unknown>=()=>({}),dirty=false;
    form.addEventListener('input',()=>dirty=true);form.addEventListener('change',()=>dirty=true);
    const send=async(settings?:Record<string,unknown>)=>{submit.disabled=retry.disabled=true;feedback.textContent='';try{await getConnection()!.configure(settings);dirty=false;dialog.close();title.textContent='正在重新捕获';message.textContent='设置已保存，完成后模型会自动更新。';}catch(e){feedback.textContent=e instanceof Error?e.message:String(e);}finally{submit.disabled=retry.disabled=false;}};
    retry.onclick=()=>void send();
    form.onsubmit=e=>{e.preventDefault();try{void send(read());}catch(err){feedback.textContent=err instanceof Error?err.message:String(err);}};
    function render(){
        body.replaceChildren();feedback.textContent='';heading.textContent=issue?'补充未确定的信息':'导入排查';description.textContent=issue?.code==='constructor_required'?'应用已读取源码构造示例与配置默认值，只需补充下方仍未确定的参数。':issue?.code==='entry_required'?'应用无法从源码唯一确定入口，请确认一个模型，后续参数仍由应用继续解析。':'应用优先自动解析。此处用于排查失败或主动覆盖自动识别结果，日常导入无需填写。';
        const s=schema;if(!s){body.append(node('p','当前入口未提供可编辑参数。可修复依赖或代码后直接重试。'));read=()=>({});submit.hidden=true;return;}
        submit.hidden=false;const selected=node('select');selected.setAttribute('aria-label','模型入口');selected.add(new Option('请选择模型入口',''));
        for(const c of s.candidates??[])selected.add(new Option(`${c.name} · ${c.file.split(/[\\/]/).pop()}:${c.line}`,c.id));selected.value=s.entry??String(s.settings.entry??'');
        const label=node('label','模型入口');label.append(selected);
        if(issue?.code==='entry_required'||!s.entry)body.append(label);
        else {body.append(node('p','已确定模型：'+(s.label??s.entry)));const change=node('details');change.append(node('summary','手动更换入口'),label);body.append(change);}
        const originalEntry=selected.value;const parameters=node('div');body.append(parameters);selected.onchange=()=>{parameters.hidden=selected.value!==originalEntry;submit.textContent=selected.value!==originalEntry?'选择入口并继续':'应用并重新捕获';};submit.textContent='应用并重新捕获';
        const fields=(title:string,items:ImportField[],parent:HTMLElement)=>{const group=node('section');group.append(node('h3',title));parent.append(group);const reads:{field:ImportField;enabled:HTMLInputElement;read:Reader}[]=[];
            for(const f of items){const row=node('fieldset');const legend=node('legend',f.name+(f.required?' · 必填':''));row.append(legend);const enabled=node('input');enabled.type='checkbox';enabled.checked=!!f.required||!!f.enabled;const enableLabel=node('label',f.required?'提供值':'覆盖默认值');enableLabel.prepend(enabled);if(f.required)enabled.disabled=true;row.append(enableLabel);const inner=node('div');row.append(inner);const reader=editor(inner,f);inner.hidden=!enabled.checked;enabled.onchange=()=>inner.hidden=!enabled.checked;group.append(row);reads.push({field:f,enabled,read:reader});}return reads;};
        const unresolved=issue?.code==='constructor_required';
        const editableConstructors=unresolved?(s.constructor??[]).filter(f=>issue?.fields?.includes(f.name)??(f.required&&!Object.hasOwn((s.settings.init??{}) as object,f.name))):issue?.code==='inputs_invalid'?[]:s.constructor??[];
        const constructors=editableConstructors.length?fields(unresolved?'尚未确定的构造参数':'构造参数',editableConstructors,parameters):[];
        if(issue?.code==='entry_required'||!s.entry)parameters.hidden=true;
        const inputMode=choice([{label:'自动推断；尺寸未定时先看结构',value:'auto'},{label:'提供示例输入，验证真实计算',value:'explicit'}]);inputMode.setAttribute('aria-label','输入方式');inputMode.value=('inputs'in s.settings||'kwargs'in s.settings)?'explicit':'auto';parameters.append(node('h3','模型输入'),inputMode);
        const inputHost=node('div');parameters.append(inputHost);const inputs=fields('forward 参数',s.inputs??[],inputHost);inputHost.hidden=inputMode.value!=='explicit';inputMode.onchange=()=>inputHost.hidden=inputMode.value!=='explicit';
        const showInputs=!['entry_required','constructor_required'].includes(issue?.code??'');
        if(!showInputs){inputMode.hidden=true;inputMode.previousElementSibling!.setAttribute('hidden','');inputHost.hidden=true;}
        const more=node('details');more.append(node('summary','捕获与内存设置'));parameters.append(more);
        const backend=choice(['auto','export','execution','fx'].map(value=>({value,label:value==='auto'?'自动选择接口':value})));backend.value=String(s.settings.backend??'auto');const backendLabel=node('label','捕获接口');backendLabel.append(backend);more.append(backendLabel);
        const budget=node('input');budget.type='number';budget.min='1';budget.step='1';budget.value=String(s.settings.max_parameters??25000000);const budgetLabel=node('label','参数与缓冲元素预算');budgetLabel.append(budget);more.append(budgetLabel,node('p','只对符合预算的模型分配权重内存；大模型建议使用仓库的小尺寸研究配置。'));
        const ignore=node('input');ignore.type='checkbox';ignore.checked=s.settings.ignore_project_config===true;const ignoreLabel=node('label','忽略项目原有导入配置');ignoreLabel.prepend(ignore);more.append(ignoreLabel);
        read=()=>{if(!selected.value)throw Error('请先选择模型入口。');if(selected.value!==originalEntry)return {entry:selected.value,ignore_project_config:ignore.checked};
            const result:Record<string,unknown>={...s.settings,entry:selected.value,init:{},backend:backend.value,ignore_project_config:ignore.checked};
            const init:Record<string,unknown>=unresolved||issue?.code==='inputs_invalid'?{...((s.settings.init??{}) as Record<string,unknown>)}:{};for(const r of constructors)if(r.enabled.checked)init[r.field.name]=r.read();result.init=init;
            const count=Number(budget.value);if(!Number.isSafeInteger(count)||count<1)throw Error('请输入有效的参数预算。');result.max_parameters=count;
            if(showInputs){delete result.inputs;delete result.kwargs;result.input_mode=inputMode.value;
                if(inputMode.value==='explicit'){const args:unknown[]=[],kwargs:Record<string,unknown>={};for(const r of inputs)if(r.enabled.checked){if(r.field.parameterKind==='positional_only')args.push(r.read());else kwargs[r.field.name]=r.read();}result.inputs=args;result.kwargs=kwargs;}}
            return result;};
        if(issue?.detail){const details=node('details');details.append(node('summary','技术详情'));const pre=node('pre',issue.detail);details.append(pre);body.append(details);}
    }
    const open=()=>{if(!dirty)render();if(!dialog.open)dialog.showModal();};settingsButton.onclick=open;
    return (s:LiveStatus)=>{
        schema=s.issue?.schema??s.importOptions;issue=s.issue;settingsButton.hidden=!s.controlToken;
        document.body.classList.toggle('awaiting-import',!s.ready);
        const signature=JSON.stringify([s.issue,s.ready,s.loading,s.importOptions]);
        if(signature===key)return;key=signature;
        if(dialog.open&&!dirty)render();
        strip.hidden=!s.error&&!s.loading;
        title.textContent=issue?.title??(s.loading?'正在捕获模型':'模型尚未就绪');
        message.textContent=issue?.message??(s.loading?(s.ready?'更新期间保留上一份模型。':'正在自动识别入口、构造参数与模型输入，请稍候。'):s.error??'');
        actions.replaceChildren();
        if(s.error){if(schema&&s.controlToken){const b=node('button',issue?.code==='entry_required'?'确认未确定的入口':'补充未确定信息');b.onclick=open;actions.append(b);}
            if(s.controlToken){const b=node('button','重试');b.onclick=()=>void send();actions.append(b);}
            if(issue?.module){const p=node('p',`缺失模块：${issue.module}。请在启动器选择项目所用的 Python 环境。`);actions.append(p);}
            if(issue?.file)actions.append(node('p',`${issue.file}${issue.line?':'+issue.line:''}`));
            if(issue?.detail){const d=node('details');d.append(node('summary','技术详情'),node('pre',issue.detail));actions.append(d);}
        }
        if(s.ready&&!s.error&&!dirty&&dialog.open)dialog.close();
    };
}
