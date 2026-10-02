import {selectExample,selectModule} from './browser-shell.mjs';
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try {
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[],report={};page.on('pageerror',e=>errors.push(e.message));await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5184');
 for(const name of ['sab','isab']) {
  await selectExample(page,name);await page.waitForFunction(n=>window.siliconDevine?.model.name===n,name);
  await page.evaluate(()=>window.siliconDevine.viewer.setPlaying(false));await page.waitForTimeout(100);
  assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().representation),'architecture');
  await page.screenshot({path:`.qa/${name}-architecture.png`,fullPage:true});
  const scope=name==='sab'?'attention':'to_inducing';await selectModule(page,scope);await page.waitForTimeout(100);
  report[name]=await page.evaluate(()=>{const v=window.siliconDevine.viewer;return {nodes:v.nodes.map(n=>({op:n.op,name:n.name})),stats:v.getStats()}});
  assert.ok(report[name].nodes.some(n=>n.op==='layout'));
  for(const op of ['softmax','layernorm','add']) {
   await page.evaluate(op=>{const v=window.siliconDevine.viewer;v.focus(v.nodes.find(n=>n.op===op).id);v.setProgress(.24);},op);await page.waitForTimeout(150);
   const state=await page.evaluate(()=>window.siliconDevine.viewer.getStats().mechanism);
   assert.ok(state?.visibleItems>0,`${name} ${op}: ${JSON.stringify(state)}`);if(op==='softmax')assert.ok(Math.abs(state.sum-1)<1e-6);
   report[name][op]=state;await page.screenshot({path:`.qa/${name}-${op}.png`,fullPage:true});
  }
  if(name==='isab') {
   await selectModule(page,'to_set');await page.waitForTimeout(100);
   const shapes=await page.evaluate(()=>{const v=window.siliconDevine.viewer;return v.nodes.filter(n=>n.op==='softmax').map(n=>v.tensors.get(n.outputs[0]).shape)});
   assert.deepEqual(shapes,[[1,4,6,3]]);report[name].secondPass=shapes;
  }
 }
 await selectExample(page,'llama');await page.waitForFunction(()=>window.siliconDevine?.model.name==='llama');await selectModule(page,'blocks.0.ffn');await page.waitForTimeout(100);
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setPlaying(false);v.focus(v.nodes.find(n=>n.op==='multiply').id);v.setProgress(.24)});await page.waitForTimeout(100);
 assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.getStats().mechanism.visibleItems>0));await page.screenshot({path:'.qa/llama-gate.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});await page.click('#fit');await page.waitForTimeout(100);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/mechanism-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);report.errors=errors;await writeFile('.qa/mechanism-browser.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close()}
