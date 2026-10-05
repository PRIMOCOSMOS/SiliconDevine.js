import { validateModel, type Model } from './model.js';
export type ArchitectureScopes = NonNullable<Model['architecture']>['scopes'];
/** Compose bounded, source-audited functional templates. Repetition denotes independent instances. */
export function defineArchitecture(name:string, scopes:ArchitectureScopes, entry='root'):Model {
    if(!Object.prototype.hasOwnProperty.call(scopes,entry))throw Error('入口模板不存在。');
    const model:Model={...scopes[entry],name,format:'silicondevine',version:1,architecture:{mode:'structure',entry,scopes,parameterCount:'0'},
        notes:['结构通路演示；未捕获数值。重复模板表示独立实例。']};
    validateModel(model);
    const memo=new Map<string,bigint>();
    const count=(key:string):bigint=>{
        if(memo.has(key))return memo.get(key)!;
        const scope=scopes[key];let total=0n;
        for(const t of scope.tensors)if(t.role==='parameter'){
            if(t.shape.some(n=>typeof n!=='number'))throw Error('参数计数需要确定的维度。');
            total+=t.shape.reduce<bigint>((n,d)=>n*BigInt(d),1n);
        }
        for(const n of scope.nodes)if(n.attrs?.scopeRef)total+=count(String(n.attrs.scopeRef))*BigInt(Number(n.attrs.repeat??1));
        memo.set(key,total);return total;
    };
    model.architecture!.parameterCount=count(entry).toString();
    return validateModel(model);
}
