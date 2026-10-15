import {build} from 'esbuild';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
await build({entryPoints:['src/index.ts'],bundle:true,platform:'node',format:'esm',outfile:'.qa/math-library.mjs',packages:'external'});
const {moduleView,operatorVisual,topologicalNodes,valueAt,validateModel}=await import('../.qa/math-library.mjs');
let checked=0;
for(const file of ['design-DesignerModel','design-OtherNames','contract-attention','contract-conv','contract-conv-fill','contract-nested']){
 const g=validateModel(JSON.parse(await fs.readFile(`.qa/${file}.json`,'utf8'))),t=new Map(g.tensors.map(v=>[v.id,v]));
 const view=moduleView(g,'','mathematical'),raw=moduleView(g,'',true);
 topologicalNodes({...g,nodes:view});
 assert.ok(view.length<raw.length);
 assert.ok(!view.some(n=>['routing_not','routing_masked_fill','cast','identity','reshape','transpose','permute','expand','layout'].includes(n.op)),`${file}: no preparation layers`);
 assert.ok(g.outputs.every(id=>view.some(n=>n.outputs.includes(id))),`${file}: returned outputs retained`);
 assert.ok(raw.some(n=>n.op==='routing_not'));
 for(const n of view)for(const id of n.outputs){
  const out=t.get(id);for(let i=0;i<Math.min(32,out.data?.values.length??0);i++){
   const deps=operatorVisual(n.op).dependencies(n,out,i,t);
   assert.ok(deps.every(d=>n.inputs.includes(d.tensor)||Object.values(n.parameters??{}).includes(d.tensor)),`${file} ${n.name} must reach actual boundary inputs`);
   if(n.op==='conv2d'){
    assert.ok(deps.length>0);
    const sum=deps.reduce((s,d)=>s+valueAt(t.get(d.tensor),d.index)*(d.weight??1),0);
    assert.ok(Math.abs(sum-valueAt(out,i))<2e-5,`${sum} / ${valueAt(out,i)}`);checked++;
   }
   if(n.op==='matmul'&&n.attrs?.attentionRole==='score'){
    const sum=deps.filter(d=>d.weight!==undefined).reduce((s,d)=>s+valueAt(t.get(d.tensor),d.index)*d.weight,0);
    assert.ok(Math.abs(sum-valueAt(out,i))<2e-5);checked++;
   }
  }
 }
 if(file==='contract-attention'){
  assert.ok(view.some(n=>n.attrs?.attentionRole==='probability'));
  assert.ok(view.some(n=>String(n.attrs?.displayLabel).includes('QKᵀ')));
  const prob=view.find(n=>n.attrs?.attentionRole==='probability'),p=t.get(prob.outputs[0]);
  for(let row=0;row<3;row++){let sum=0;for(let col=0;col<3;col++){const v=valueAt(p,row*3+col);if(col>row)assert.equal(v,0);sum+=v;}assert.ok(Math.abs(sum-1)<1e-6);}
 }
}
const g=JSON.parse(await fs.readFile('.qa/expnet-graph.json','utf8')),t=new Map(g.tensors.map(v=>[v.id,v]));
let scopes=0;
for(const unit of g.composition.units){
 const nodes=moduleView(g,unit.id,'mathematical');topologicalNodes({...g,nodes});scopes++;
 assert.ok(!nodes.some(n=>n.op==='layout'||['routing_not','cast','identity','routing_masked_fill'].includes(n.op)),`${unit.name}: engineering layers omitted`);
 for(const n of nodes)for(const id of n.outputs.slice(-1)){
  const out=t.get(id),deps=operatorVisual(n.op).dependencies(n,out,0,t);
  assert.ok(deps.every(d=>n.inputs.includes(d.tensor)||Object.values(n.parameters??{}).includes(d.tensor)),`${unit.name} ${n.name}: no invisible dependency`);
 }
}
assert.equal(g.inputContract.parameters.length,6);
assert.equal(g.inputContract.parameters.find(p=>p.name==='valid').status,'default');
assert.ok(g.inputContract.parameters.filter(p=>p.required).every(p=>p.tensors.length&&p.symbolicShape));
const prep=g.composition.units.find(u=>u.name==='张量准备与变换'),prepared=moduleView(g,prep.id,'mathematical');
const features=prepared.find(n=>n.attrs?.designKind==='feature_construction');
assert.ok(features);
assert.ok(!features.inputs.includes(g.inputs[0]),'A shared validity predicate does not make unrelated signal data part of the feature function');
assert.ok(!features.outputs.some(id=>t.get(id).dtype==='bool'),'A predicate is not a feature output');
console.log(`Mathematical presentation: ${checked} exact score/convolution values, causal probabilities, ${scopes} real scopes, and complete input contract verified.`);
const scalar=validateModel(JSON.parse(await fs.readFile('.qa/contract-scalar.json','utf8'))),st=new Map(scalar.tensors.map(t=>[t.id,t]));
const n=scalar.nodes[0],out=st.get(n.outputs[0]);
for(let i=0;i<6;i++){const deps=operatorVisual(n.op).dependencies(n,out,i,st);assert.equal(deps.length,2);assert.equal(valueAt(out,i),valueAt(st.get(n.inputs[0]),i)*valueAt(st.get(n.inputs[1]),0));}
const invalid=structuredClone(scalar);invalid.inputContract.parameters[0].structure.children=[null];assert.throws(()=>validateModel(invalid),/输入结构无效/);
const norm=validateModel(JSON.parse(await fs.readFile('.qa/contract-normalization.json','utf8'))),nt=new Map(norm.tensors.map(t=>[t.id,t]));
const normalized=moduleView(norm,'','mathematical');assert.equal(normalized.length,1);
assert.equal(normalized[0].op,'rmsnorm');assert.ok(normalized[0].parameters.weight);
assert.ok(moduleView(norm,'',true).some(n=>n.op==='rsqrt'));
for(let i=0;i<6;i++){
 const n=normalized[0],out=nt.get(n.outputs[0]),x=nt.get(n.inputs[0]),gamma=nt.get(n.parameters.weight),row=Math.floor(i/3);
 const rms=Math.sqrt([0,1,2].reduce((s,j)=>s+valueAt(x,row*3+j)**2,0)/3+n.attrs.eps);
 assert.ok(Math.abs(valueAt(out,i)-valueAt(gamma,i%3)*valueAt(x,i)/rms)<2e-6);
 assert.ok(operatorVisual(n.op).dependencies(n,out,i,nt).every(d=>n.inputs.includes(d.tensor)||Object.values(n.parameters).includes(d.tensor)));
}
const large=validateModel(JSON.parse(await fs.readFile('.qa/contract-large-mask.json','utf8'))),lt=new Map(large.tensors.map(t=>[t.id,t]));
const visible=moduleView(large,'','mathematical');assert.equal(visible.length,1);assert.equal(visible[0].op,'conv2d');
for(const i of [0,1,2,4100,4200,4224]){
 const n=visible[0],out=lt.get(n.outputs[0]),deps=operatorVisual(n.op).dependencies(n,out,i,lt);
 assert.ok(deps.every(d=>n.inputs.includes(d.tensor)||Object.values(n.parameters).includes(d.tensor)));
 const actual=deps.reduce((s,d)=>s+valueAt(lt.get(d.tensor),d.index)*(d.weight??1),0);
 assert.ok(Math.abs(actual-valueAt(out,i))<2e-5);
}
