import {build} from 'esbuild';
import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
await build({entryPoints:['src/index.ts'],bundle:true,platform:'node',format:'esm',outfile:'.qa/atomic-library.mjs',external:['three','three/*']});
const {atomicComposition,resolveAtomicComposition,compositionCatalog,validateModel,registerOperator}=await import('../.qa/atomic-library.mjs');
for(const family of ['deepseek_v3','glm45','minimax_m1']){
 const m=validateModel(JSON.parse(await readFile(`demo/public/models/${family}.sd.json`,'utf8'))),a=m.architecture;
 const catalog=compositionCatalog(m);assert.equal(catalog.units.length,Object.keys(a.scopes).length);
 for(const s of Object.values(a.scopes))for(const n of s.nodes){
  const unit=resolveAtomicComposition(m,n);
  if(n.attrs.kind==='repeat'&&unit)assert.equal(unit.basis,'captured','Repeated decoder must use original execution');
  if(unit){assert.ok(['captured','demonstration'].includes(unit.basis));assert.equal(unit.boundaries.length,0);assert.equal(unit.atoms.length,unit.model.nodes.length);}
 }
 const attention=atomicComposition(a.mechanisms.attention);for(const op of ['linear','matmul','softmax','rope_pair'])assert.ok(attention.vocabulary.some(v=>v.op===op));
 const expert=atomicComposition(a.mechanisms.expert);for(const op of ['linear','silu','multiply'])assert.ok(expert.vocabulary.some(v=>v.op===op));
 assert.strictEqual(expert.model,a.mechanisms.expert,'Preserve actual tensors and parameters');
 assert.ok(catalog.units.flatMap(u=>u.children).some(c=>c.repeat>1));
}
const captured=validateModel(JSON.parse(await readFile('demo/public/models/llama.sd.json','utf8')));
assert.equal(atomicComposition(captured).basis,'captured');assert.strictEqual(resolveAtomicComposition(captured).model,captured);
const custom={...captured,nodes:[{...captured.nodes[0],op:'custom_probe'}]};
assert.equal(atomicComposition(custom).boundaries.length,1);
registerOperator('custom_probe',{label:'Plugin',color:'#fff',formula:'x',detail:'exact',dependencies:()=>[]});
assert.equal(atomicComposition(custom).boundaries.length,0);
console.log('Atomic composition: three model families, captured graph, repetition, source identity and plugins passed.');
