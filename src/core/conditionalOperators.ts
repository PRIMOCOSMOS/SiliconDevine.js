import {coordinates,flatIndex,valueAt,type Tensor} from './model.js';
import type {OperatorVisual,Dependency} from './operators.js';
export function installConditionalOperators(register:(name:string,visual:OperatorVisual)=>void){
 register('kernel_mix',{label:'条件核混合',color:'#ddb879',formula:'W_b = Σ_e α[b,e] W_e',detail:'exact',dependencies:(n,out,index,ts)=>{
  const a=ts.get(n.inputs[0])!,w=ts.get(n.inputs[1])!,stride=(w.shape as number[]).slice(1).reduce((a,b)=>a*b,1),b=Math.floor(index/stride),j=index%stride;
  return Array.from({length:Number(a.shape[1])},(_,e)=>({tensor:w.id,index:e*stride+j,weight:valueAt(a,b*Number(a.shape[1])+e),parameter:a.id,parameterIndex:b*Number(a.shape[1])+e}));
 }});
 register('dynamic_conv2d',{label:'逐样本动态卷积',color:'#84bce6',formula:'Y[b,o,p] = Σ W[b,o,c,k] X[b,c,p·s−pad+k·d]',detail:'exact',dependencies:(n,out,index,ts)=>{
  const x=ts.get(n.inputs[0])!,w=ts.get(n.parameters!.weight)!,xs=x.shape as number[],ws=w.shape as number[],os=out.shape as number[],c=coordinates(index,os),arr=(v:unknown,d:number)=>Array.isArray(v)?v as number[]:Array(2).fill(typeof v==='number'?v:d),a=n.attrs??{},s=arr(a.stride,1),p=arr(a.padding,0),d=arr(a.dilation,1),group=Math.floor(c[1]/(os[1]/Number(a.groups??1))),deps:Dependency[]=[];
  for(let i=0;i<ws[2];i++)for(let u=0;u<ws[3];u++)for(let v=0;v<ws[4];v++){const y=c[2]*s[0]-p[0]+u*d[0],z=c[3]*s[1]-p[1]+v*d[1];if(y<0||z<0||y>=xs[2]||z>=xs[3])continue;const wi=flatIndex([c[0],c[1],i,u,v],ws);deps.push({tensor:x.id,index:flatIndex([c[0],group*ws[2]+i,y,z],xs),weight:valueAt(w,wi),parameter:w.id,parameterIndex:wi});}return deps;
 }});
 register('rope_pair',{label:'RoPE 配对旋转',color:'#b9a3e3',formula:'(x₀ cosθ − x₁ sinθ, x₀ sinθ + x₁ cosθ)',detail:'exact',dependencies:(n,out,i,ts)=>{
  const s=out.shape as number[],d=s.at(-1)!,rd=Number(n.attrs?.rotaryDim??d);if(i%d>=rd)return [{tensor:n.inputs[0],index:i}];const p=Math.floor((i%d)/2),row=Math.floor(i/d)%s.at(-2)!,ci=row*rd/2+p,cs=ts.get(n.inputs[1])!,sn=ts.get(n.inputs[2])!,base=i-i%2;
  return [{tensor:n.inputs[0],index:base,weight:valueAt(i%2?sn:cs,ci)},{tensor:n.inputs[0],index:base+1,weight:(i%2?1:-1)*valueAt(i%2?cs:sn,ci)},{tensor:cs.id,index:ci},{tensor:sn.id,index:ci}];
 }});
 register('state_update',{label:'递归状态写入',color:'#8ed4b1',formula:'Sₜ = λ Sₜ₋₁ + kₜᵀvₜ',detail:'exact',dependencies:(n,_,i,ts)=>[{tensor:n.inputs[0],index:i,weight:valueAt(ts.get(n.inputs[2])!,0)},{tensor:n.inputs[1],index:i,weight:1}]});
 const rowDeps=(t:Tensor,row:number)=>Array.from({length:Number(t.shape.at(-1))},(_,i)=>({tensor:t.id,index:row*Number(t.shape.at(-1))+i}));
 register('route_select',{label:'分组 Top-k 选择',color:'#e3a2bb',formula:'按组筛选，再取 Top-k；校正项只参与选择',detail:'exact',dependencies:(n,o,i,ts)=>[...rowDeps(ts.get(n.inputs[0])!,Math.floor(i/Number(o.shape.at(-1)))),...(n.inputs[1]?rowDeps(ts.get(n.inputs[1])!,0):[])]});
 register('route_weights',{label:'路由权重归一化',color:'#ddb879',formula:'w = s[selected] / Σ s[selected] × scale',detail:'exact',dependencies:(n,o,i,ts)=>{
  const indices=ts.get(n.inputs[1])!,s=ts.get(n.inputs[0])!,k=Number(o.shape.at(-1)),row=Math.floor(i/k);return Array.from({length:k},(_,j)=>({tensor:s.id,index:row*Number(s.shape.at(-1))+valueAt(indices,row*k+j)}));
 }});
 register('expert_combine',{label:'稀疏专家汇聚',color:'#efe0ae',formula:'y[t,d] = Σ_j w[t,j] expert[index[t,j]][t,d]',detail:'exact',dependencies:(n,o,i,ts)=>{
  const idx=ts.get(n.inputs[0])!,w=ts.get(n.inputs[1])!,row=Math.floor(i/Number(o.shape.at(-1))),k=Number(idx.shape.at(-1));return Array.from({length:k},(_,j)=>({tensor:n.inputs[2+valueAt(idx,row*k+j)],index:i,weight:valueAt(w,row*k+j),parameter:w.id,parameterIndex:row*k+j}));
 }});
}
