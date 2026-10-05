import {coordinates,flatIndex,valueAt,type Tensor} from './model.js';
import type {OperatorVisual,Dependency} from './operators.js';
const evenRound=(x:number)=>{const n=Math.floor(x),r=x-n;return r===.5?n+(Math.abs(n)%2):Math.round(x);};
export function installSpatialOperators(register:(name:string,visual:OperatorVisual)=>void){
    register('interpolate',{label:'插值重采样',color:'#8fbddd',formula:'y[p] = Σ 邻域权重 · x[q]；坐标由缩放与 align_corners 决定',detail:'exact',dependencies:(n,out,index,t)=>{
        const x=t.get(n.inputs[0])!,xs=x.shape as number[],os=out.shape as number[],c=coordinates(index,os),a=n.attrs??{},mode=String(a.mode??'nearest'),rank=xs.length-2;
        const factors=Array.isArray(a.scale_factor)?a.scale_factor:Array(rank).fill(a.scale_factor);
        const axis:Array<Array<[number,number]>>=[];
        for(let j=0;j<rank;j++){
            const input=xs[j+2],output=os[j+2],at=c[j+2],f=Number(factors[j]),ratio=!a.recompute_scale_factor&&f>0?1/f:input/output;
            if(mode.startsWith('nearest'))axis.push([[Math.min(input-1,Math.floor((at+(mode==='nearest-exact'?.5:0))*ratio)),1]]);
            else{
                const pos=a.align_corners?(output>1?at*(input-1)/(output-1):0):Math.max(0,(at+.5)*ratio-.5),lo=Math.min(input-1,Math.floor(pos)),hi=Math.min(input-1,lo+1),fraction=pos-Math.floor(pos);
                axis.push([[lo,1-fraction],[hi,fraction]]);
            }
        }
        let terms:{coords:number[],weight:number}[]=[{coords:c.slice(0,2),weight:1}];
        for(const choices of axis)terms=terms.flatMap(term=>choices.map(([v,w])=>({coords:[...term.coords,v],weight:term.weight*w})));
        return terms.filter(v=>v.weight>0).map(v=>({tensor:x.id,index:flatIndex(v.coords,xs),weight:v.weight}));
    }});
    register('grid_sample2d',{label:'坐标网格采样',color:'#8fd2ce',formula:'grid[n,h,w] → 源图连续坐标 → 邻域加权采样',detail:'exact',dependencies:(n,out,index,t)=>{
        const x=t.get(n.inputs[0])!,grid=t.get(n.inputs[1])!,xs=x.shape as number[],os=out.shape as number[],[b,ch,y,z]=coordinates(index,os),a=n.attrs??{},gi=flatIndex([b,y,z,0],grid.shape as number[]);
        const normalized=[valueAt(grid,gi+1),valueAt(grid,gi)],sizes=xs.slice(-2),align=Boolean(a.align_corners),padding=a.padding_mode??'zeros';
        if(normalized.some(v=>!Number.isFinite(v)))return []; // No invented coordinates outside captured grid windows.
        const source=normalized.map((v,j)=>{
            const size=sizes[j];let p=align?(v+1)*(size-1)/2:((v+1)*size-1)/2;
            if(padding==='reflection'){
                const low=align?0:-.5,span=align?size-1:size;
                if(span===0)p=0;else{const q=Math.abs(p-low),r=q%span;p=Math.floor(q/span)%2?span-r+low:r+low;}
            }
            if(padding==='reflection'||padding==='border')p=Math.max(0,Math.min(size-1,p));
            return p;
        });
        const choices=source.map(p=>a.mode==='nearest'?[[evenRound(p),1]]:[[Math.floor(p),1-(p-Math.floor(p))],[Math.floor(p)+1,p-Math.floor(p)]]);
        const deps:Dependency[]=[{tensor:grid.id,index:gi,weight:0},{tensor:grid.id,index:gi+1,weight:0}];
        for(const [row,wy] of choices[0])for(const [col,wx] of choices[1])if(row>=0&&row<sizes[0]&&col>=0&&col<sizes[1]&&wx*wy>0)deps.push({tensor:x.id,index:flatIndex([b,ch,row,col],xs),weight:wx*wy});
        return deps;
    }});
    for(const op of ['gather','index_select'])register(op,{label:op==='gather'?'按索引收集':'沿轴选取',color:'#93c9d4',formula:'y 的目标坐标沿 dim 替换为 index 指定的源坐标',detail:'exact',dependencies:(n,out,index,t)=>{
        const x=t.get(n.inputs[0])!,indices=t.get(n.inputs[1])!,s=x.shape as number[],c=coordinates(index,out.shape as number[]),axis=(Number(n.attrs?.dim??0)+s.length)%s.length,ii=op==='gather'?index:c[axis],at=valueAt(indices,ii);
        if(!Number.isInteger(at)||at<0||at>=s[axis])return [];
        c[axis]=at;return [{tensor:indices.id,index:ii,weight:0},{tensor:x.id,index:flatIndex(c,s),weight:1}];
    }});
    for(const op of ['sum','amax','amin'])register(op,{label:{sum:'求和归约',amax:'最大值归约',amin:'最小值归约'}[op]!,color:'#c4ade2',formula:{sum:'y = Σ x[归约轴]',amax:'y = max x[归约轴]',amin:'y = min x[归约轴]'}[op]!,detail:'exact',dependencies:(n,out,index,t)=>{
        const x=t.get(n.inputs[0])!,s=x.shape as number[],a=n.attrs??{},raw=a.dims as number[]|null,axes=(raw?.length?raw:s.map((_,i)=>i)).map(i=>(i+s.length)%s.length),oc=coordinates(index,out.shape as number[]);let k=0;
        const base=s.map((_,j)=>axes.includes(j)?0:oc[a.keepdim?j:k++]),reduced=axes.map(j=>s[j]),count=reduced.reduce((p,v)=>p*v,1);
        if(count>65536)return [];
        return Array.from({length:count},(_,i)=>{const c=[...base],rc=coordinates(i,reduced);axes.forEach((j,k)=>c[j]=rc[k]);return {tensor:x.id,index:flatIndex(c,s),weight:op==='sum'?1:undefined};});
    }});
}
