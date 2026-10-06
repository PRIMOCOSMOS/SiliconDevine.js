import {selectExample,selectModule} from './browser-shell.mjs';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[],report=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5195');await page.waitForFunction(()=>window.siliconDevine?.viewer.nodes.length>0);
 async function check(name,op){
  await page.mouse.move(0,0);await page.evaluate(op=>{const v=window.siliconDevine.viewer;v.setPlaying(false);v.setLabels('auto');v.focus(v.nodes.find(n=>n.op===op).id);},op);await page.waitForTimeout(850);
  const state=await page.evaluate(()=>{
   const v=window.siliconDevine.viewer,a=v.getAssembly(v.selected),frame=v.assemblyBoundaries.bounds.get(a.id);
   return {unit:a,visible:v.cards.filter(c=>c.group.visible).map(c=>c.id),selected:v.selected,frame:v.assemblyBoundaries.selected.visible,
    owned:a.owned.map(id=>{const p=v.planes.get(id);return {id,inside:frame.containsPoint(v.camera.position.clone().set(...p.center)),opacity:v.planeOutlines.get(id).material.opacity};}),
    related:a.related.map(id=>({id,opacity:v.planeOutlines.get(id).material.opacity})),
    background:[...v.planeOutlines].filter(([id])=>!a.related.includes(id)).map(([,p])=>p.material.opacity),
    anchors:v.cards.map(c=>({id:c.id,p:c.group.position.toArray(),q:c.group.quaternion.toArray()})),camera:v.camera.position.toArray(),target:v.controls.target.toArray(),flight:v.cameraFlight,stats:v.getStats()};
  });
  assert.ok(state.frame);assert.deepEqual(new Set(state.visible),new Set(state.unit.related),'Selection reveals all associated captions');
  assert.ok(state.owned.every(p=>p.inside));assert.ok(state.related.every(p=>p.opacity>.9));assert.ok(state.background.every(o=>o<.2));
  await page.screenshot({path:`.qa/assembly-${name}.png`});
  // Moving across another layer does not steal an explicitly selected assembly.
  await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.hovered=v.nodes.find(n=>!v.getAssembly(v.selected).nodes.includes(n.id))?.id;v.dirty=true;});await page.waitForTimeout(100);
  assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.assemblyBoundaries.active),state.unit.id);
  await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.hovered=undefined;v.orbit(.45,.1);});await page.waitForTimeout(200);
  assert.deepEqual(await page.evaluate(()=>window.siliconDevine.viewer.cards.map(c=>({id:c.id,p:c.group.position.toArray(),q:c.group.quaternion.toArray()}))),state.anchors);
  await page.evaluate(()=>window.siliconDevine.viewer.setLabels('none'));await page.waitForTimeout(80);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.cards.filter(c=>c.group.visible).length),0);
  await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setLabels('auto');v.clearFocus();});await page.waitForTimeout(850);
  assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.assemblyBoundaries.selected.visible),false);
  report.push({name,tensors:state.unit.related.length,nodes:state.unit.nodes.length,drawCalls:state.stats.drawCalls,camera:state.camera,target:state.target,flight:state.flight});
 }
 await check('mlp','linear');
 await page.locator('#file').setInputFiles('.qa/layout-user.json');await page.waitForFunction(()=>window.siliconDevine.model.name==='HybridVisionNetwork');await selectModule(page,'res_block1');
 await check('cnn','conv2d');await check('norm','batchnorm');
 await selectExample(page,'deepseek_v3');await selectModule(page,'moe_block');await selectModule(page,'decoder.self_attn');await page.selectOption('#representation','operators');
 await check('attention','softmax');
 await page.setViewportSize({width:390,height:844});await check('mobile-attention','softmax');
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);
 await writeFile('.qa/assembly-browser.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}finally{await browser.close();}
