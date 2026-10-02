import {selectExample,selectModule} from './browser-shell.mjs';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
await mkdir('.qa',{recursive:true});
try{
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[],report={};page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5184');
 for(const name of ['tinygpt','llama']){
  await selectExample(page,name);await page.waitForFunction(n=>window.siliconDevine?.model.name===n,name);
  await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setPlaying(false);v.clearFocus();});await page.waitForTimeout(150);
  const overview=await page.evaluate(()=>({stats:window.siliconDevine.viewer.getStats(),nav:window.siliconDevine.viewer.getNavigation()}));assert.equal(overview.nav.representation,'architecture');assert.ok(overview.nav.nodes.some(n=>n.op==='module'));assert.equal(overview.stats.visibleNodes,overview.stats.totalNodes);
  await page.screenshot({path:`.qa/${name}-overview.png`,fullPage:true});
  await page.evaluate(()=>window.siliconDevine.viewer.focus('module:blocks.0'));await page.waitForTimeout(100);
  // The pointer hits a 3D ownership region; hover switches immediately, click enters its module.
  const point=await page.evaluate(()=>{const v=window.siliconDevine.viewer,n=v.nodes.find(n=>n.id==='module:blocks.0.attention'),r=v.regions.find(r=>r.node===n.id),p=r.box.getCenter(v.camera.position.clone());p.z=r.box.max.z;const projected=p.project(v.camera),rect=document.querySelector('#stage').getBoundingClientRect();return{x:rect.left+(projected.x+1)*rect.width/2,y:rect.top+(1-projected.y)*rect.height/2,id:n.id};});
  await page.mouse.move(point.x,point.y);await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getStats().active),point.id);await page.mouse.click(point.x,point.y);await page.waitForTimeout(150);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().scope),'blocks.0.attention');
  await page.mouse.move(10,10);await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.focus(v.nodes.find(n=>n.op==='attention_mask').id);});await page.waitForTimeout(100);await page.screenshot({path:`.qa/${name}-causal.png`,fullPage:true});
  const mask=await page.evaluate(()=>{const v=window.siliconDevine.viewer,n=v.nodes.find(n=>n.op==='attention_mask');return{op:n.op,values:v.tensors.get(n.outputs[0]).specialValues,fabrics:v.fabrics.size};});assert.ok(Object.values(mask.values).includes('-inf'));assert.equal(mask.fabrics,1);
  if(name==='llama'){
   await selectModule(page,'blocks.0.attention.rope');await page.waitForTimeout(120);await page.screenshot({path:'.qa/llama-rope.png',fullPage:true});
   await selectModule(page,'blocks.0.ffn');await page.waitForTimeout(150);await page.screenshot({path:'.qa/llama-swiglu.png',fullPage:true});assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.nodes.some(n=>n.op==='silu')));
  }
  await selectModule(page,'');await page.selectOption('#representation','operators');await page.waitForTimeout(150);const raw=await page.evaluate(()=>window.siliconDevine.viewer.getStats());
  await page.selectOption('#representation','architecture');await page.waitForTimeout(150);const compact=await page.evaluate(()=>window.siliconDevine.viewer.getStats());
  assert.ok(compact.cells<raw.cells);report[name]={rawFirstPage:raw,completeArchitecture:compact,hoverAndDrilldown:true};
 }
 await page.setViewportSize({width:390,height:844});await selectModule(page,'blocks.0.ffn');await page.click('#fit');await page.waitForTimeout(150);await page.screenshot({path:'.qa/llama-mobile.png',fullPage:true});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 assert.deepEqual(errors,[]);report.errors=errors;await writeFile('.qa/llm-browser.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();}
