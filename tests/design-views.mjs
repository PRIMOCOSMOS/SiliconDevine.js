import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {moduleView,topologicalNodes,operatorVisual,valueAt,validateModel} from '../dist/silicondevine.js';
for(const name of ['DesignerModel','OtherNames']){
 const g=validateModel(JSON.parse(await readFile(`.qa/design-${name}.json`,'utf8'))),t=new Map(g.tensors.map(t=>[t.id,t]));
 const view=moduleView(g,'','mathematical'),raw=moduleView(g,'',true);
 topologicalNodes({...g,nodes:view});
 const dot=view.find(n=>n.attrs?.designKind==='dot_product');assert.ok(dot,'Multiply + last-axis sum is dot');
 const norm=view.filter(n=>n.op==='multiply');assert.ok(norm.length,'Unreduced product remains a product');
 assert.ok(!view.some(n=>n.attrs?.designKind==='valid_selection'),'Mask preparation is not a standalone architecture');
 assert.ok(view.some(n=>n.attrs?.engineeringAbsorbed&&n.attrs?.inputConditions?.length),'Validity remains a condition on the mathematical unit');
 assert.ok(view.some(n=>n.attrs?.designKind==='feature_construction'));
 assert.ok(view.some(n=>n.attrs?.displayLabel==='残差连接'));
 assert.ok(!view.some(n=>['routing_not','routing_unbind','routing_masked_fill'].includes(n.op)));
 assert.ok(raw.some(n=>n.op==='routing_not'));assert.ok(view.length<raw.length*.65);
 // Boundary outputs always keep the exact captured tensor and value.
 for(const n of view)for(const id of n.outputs)assert.ok(t.has(id));
 const out=t.get(dot.outputs[0]);
 for(let i=0;i<out.data.values.length;i++){
  const deps=operatorVisual(dot.op).dependencies(dot,out,i,t);
  assert.ok(deps.every(d=>dot.inputs.includes(d.tensor)));
  const a=t.get(dot.inputs[0]),b=t.get(dot.inputs[1]);
  assert.equal(valueAt(out,i),valueAt(a,2*i)*valueAt(b,2*i)+valueAt(a,2*i+1)*valueAt(b,2*i+1));
 }
}
const g=JSON.parse(await readFile('.qa/expnet-graph.json','utf8'));
const prep=g.composition.units.find(u=>u.name==='张量准备与变换'),view=moduleView(g,prep.id),raw=moduleView(g,prep.id,true);
assert.ok(view.length<raw.length/2,`Preparation: ${view.length}/${raw.length}`);
assert.ok(view.some(n=>n.attrs?.designKind==='feature_construction'));
assert.ok(!view.some(n=>['routing_not','routing_unbind','routing_masked_fill'].includes(n.op)));
console.log(`Design views verified; actual preparation ${raw.length} calls → ${view.length} mathematical units.`);
const p=JSON.parse(await readFile('.qa/design-projected.json','utf8')),t=new Map(p.tensors.map(t=>[t.id,t])),ps=moduleView(p,'','mathematical');
assert.equal(ps.length,1);
const projection=ps[0];assert.equal(projection.attrs.designKind,'projection');
const out=t.get(projection.outputs[0]),w=t.get(projection.parameters.weight),bias=t.get(projection.parameters.bias);
for(let i=0;i<out.data.values.length;i++){
 const deps=operatorVisual('semantic').dependencies(projection,out,i,t),row=Math.floor(i/6)*2+i%2;
 assert.equal(deps.length,5);
 const links=deps.filter(d=>d.parameter===w.id);assert.equal(links.length,4);
 assert.ok(links.every(d=>Math.floor(d.parameterIndex/4)===row));
 const actual=links.reduce((s,d)=>s+valueAt(t.get(d.tensor),d.index)*valueAt(w,d.parameterIndex),valueAt(bias,row));
 assert.ok(Math.abs(actual-valueAt(out,i))<1e-6);
}
console.log('Projected head vectors preserve W indices, bias and exact values.');
