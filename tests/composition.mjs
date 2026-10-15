import {build} from 'esbuild';
import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
await build({entryPoints:['src/core/hierarchy.ts'],outfile:'.qa/hierarchy-check.mjs',bundle:true,platform:'node',format:'esm'});
await build({entryPoints:['src/core/operators.ts'],outfile:'.qa/operators-check.mjs',bundle:true,platform:'node',format:'esm'});
await build({entryPoints:['src/core/layout.ts'],outfile:'.qa/orbital-check.mjs',bundle:true,platform:'node',format:'esm'});
const {moduleView}=await import('../.qa/hierarchy-check.mjs');
const {operatorVisual}=await import('../.qa/operators-check.mjs');
const {layoutModel}=await import('../.qa/orbital-check.mjs');
const model=JSON.parse(await readFile('.qa/mmhvae-composition.json','utf8'));
const overview=moduleView(model);assert.ok(overview.length<40);
assert.ok(overview.some(n=>n.attrs?.compositionVisual?.stages));
assert.ok(!overview.some(n=>n.op==='opaque'));
const heads=overview.filter(n=>n.attrs?.compositionVisual?.family&&n.attrs?.compositionVisual?.distribution);
assert.equal(heads.length,4);assert.deepEqual(new Set(heads.map(n=>n.attrs.compositionVisual.branch)),new Set(['us','t2','cet1','flair']));
const ids=new Set(overview.flatMap(n=>[...n.inputs,...n.outputs,...Object.values(n.parameters??{})]));
const view={...model,nodes:overview,tensors:model.tensors.filter(t=>ids.has(t.id))};
const planes=layoutModel(view,24);
const towers=overview.filter(n=>n.attrs?.compositionVisual?.stages?.length===7);
const centers=towers.map(n=>planes.get(n.outputs[0]).center);
assert.equal(new Set(centers.map(p=>[p[0],p[2]].map(v=>v.toFixed(1)).join(','))).size,4);
for(const n of towers){const ps=n.attrs.compositionVisual.stages.map(s=>planes.get(s.tensor));assert.equal(ps.length,7);assert.equal(new Set(ps.map(p=>p.center[1])).size,7);assert.equal(new Set(ps.map(p=>p.center[0])).size,1);}
for(const h of heads){const d=h.attrs.compositionVisual.distribution;assert.ok(model.tensors.find(t=>t.id===d.mean));assert.ok(model.tensors.find(t=>t.id===d.std));}
let scopes=0;
for(const unit of model.composition.units.filter(u=>u.nodes.length>1)){
 const nodes=moduleView(model,unit.id);
 if(!nodes.length){
  const raw=moduleView(model,unit.id,true),t=new Map(model.tensors.map(t=>[t.id,t]));
  assert.ok(raw.every(n=>n.op==='cast'||n.outputs.every(id=>t.get(id).dtype==='bool')),`${unit.name}: only engineering/guard scopes may disappear`);
 }
 if(unit.visual?.kind==='gaussian_parameters')assert.ok(nodes.some(n=>n.attrs?.designKind==='gaussian_distribution'),'A distribution retains its mathematical meaning');
 // Every visible port is a real captured tensor; no placeholder models.
 const tensors=new Set(model.tensors.map(t=>t.id));
 for(const n of nodes)for(const t of [...n.inputs,...n.outputs])assert.ok(tensors.has(t));
 scopes++;
}
const scalar={id:'s',shape:[],role:'constant',dtype:'float32'},out={id:'o',shape:[2,3],role:'activation',dtype:'float32'};
const broadcast={id:'b',op:'broadcast_tensors',inputs:['s'],outputs:['o']};
assert.deepEqual(operatorVisual('broadcast_tensors').dependencies(broadcast,out,5,new Map([['s',scalar]])),[{tensor:'s',index:0}]);
console.log(JSON.stringify({overview:overview.length,validatedScopes:scopes}));

for(const count of [2,3,5])for(const cls of ['Branches','RenamedBranches']){
 const model=JSON.parse(await readFile(`.qa/generic-${cls}-${count}.json`,'utf8'));
 const nodes=moduleView(model),ids=new Set(nodes.flatMap(n=>[...n.inputs,...n.outputs,...Object.values(n.parameters??{})]));
 const view={...model,nodes,tensors:model.tensors.filter(t=>ids.has(t.id))};
 const planes=layoutModel(view,24),towers=nodes.filter(n=>n.attrs?.compositionVisual?.stages);
 assert.equal(towers.length,count);
 const centers=towers.map(n=>planes.get(n.attrs.compositionVisual.stages[0].tensor).center);
 assert.equal(new Set(centers.map(p=>`${p[0].toFixed(2)},${p[2].toFixed(2)}`)).size,count);
 for(const n of towers){const ps=n.attrs.compositionVisual.stages.map(s=>planes.get(s.tensor));
  assert.equal(ps.length,3);assert.ok(ps.every((p,i)=>i===0||ps[i-1].center[1]>p.center[1]));
  assert.ok(ps.every(p=>p.center.every(Number.isFinite)));
 }
 const owner=new Map(nodes.flatMap(n=>n.outputs.map(t=>[t,n])));
 for(const n of nodes)for(const t of n.inputs){const from=owner.get(t);if(from&&from.id!==n.id){
  const input=planes.get(t),output=planes.get(n.outputs[0]);assert.ok(input.center[1]>output.center[1]);
 }}
}
console.log('Generic parallel geometry: 2 / 3 / 5 branches, renamed sources, descending real dependencies.');

for(const parent of model.composition.units){
 const visible=moduleView(model,parent.id),families=new Map();
 for(const n of visible){const v=n.attrs?.compositionVisual;if(v?.family)families.set(v.family,[...families.get(v.family)??[],v.branch]);}
 for(const keys of families.values())assert.equal(new Set(keys).size,keys.length,'Independent calls must not share one spatial lane');
}
