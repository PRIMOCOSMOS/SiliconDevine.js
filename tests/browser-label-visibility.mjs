import {selectExample,selectModule} from './browser-shell.mjs';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
const report=[];
try{
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5195');
 await page.waitForFunction(()=>window.siliconDevine?.viewer.nodes.length>0);
 const inspect=()=>page.evaluate(()=>{
  const v=window.siliconDevine.viewer;
  return {total:v.cards.length,visible:v.cards.filter(c=>c.group.visible).map(c=>{
   const m=c.group.children[0],b=m.geometry.boundingBox;
   const ps=[[b.min.x,b.min.y],[b.max.x,b.max.y],[b.min.x,b.max.y],[b.max.x,b.min.y]].map(([x,y])=>m.localToWorld(v.camera.position.clone().set(x,y,0)).project(v.camera));
   const onScreen=Math.max(...ps.map(p=>p.x))>=-1&&Math.min(...ps.map(p=>p.x))<=1&&Math.max(...ps.map(p=>p.y))>=-1&&Math.min(...ps.map(p=>p.y))<=1&&ps.every(p=>p.z>=-1&&p.z<=1);
   const front=v.camera.position.clone().set(0,0,1).transformDirection(m.matrixWorld);
   const cam=v.camera.position.clone().set(0,0,1).applyQuaternion(v.camera.quaternion);
   return {id:c.id,owner:c.owner,onScreen,facing:front.dot(cam),depthTest:m.material.depthTest};
  }),anchors:v.cards.map(c=>({id:c.id,p:c.group.position.toArray(),s:c.group.scale.toArray(),q:c.group.quaternion.toArray()}))};
 });
 async function checkView(name){
  await page.mouse.move(0,0);
  await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setPlaying(false);v.clearFocus();v.setLabels('auto');});await page.waitForTimeout(850);
  const baseline=await inspect();
  assert.ok(baseline.visible.some(c=>c.onScreen),`${name}: auto captions cannot all disappear`);
  for(let step=0;step<8;step++){
   await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.stopFlight();const d=v.camera.position.clone().sub(v.controls.target);d.applyAxisAngle(d.clone().set(0,1,0),Math.PI/4);v.camera.position.copy(v.controls.target).add(d);v.controls.update();});await page.waitForTimeout(90);
   const state=await inspect();assert.deepEqual(state.anchors,baseline.anchors,`${name}: fixed anchors`);
   assert.ok(state.visible.some(c=>c.onScreen),`${name}: orbit ${step} needs a visible caption`);
   assert.ok(state.visible.every(c=>c.facing>.999&&!c.depthTest),`${name}: readable lettering`);
  }
  await page.evaluate(()=>window.siliconDevine.viewer.setLabels('all'));await page.waitForTimeout(100);
  const all=await inspect();assert.equal(all.visible.length,all.total,`${name}: all must mean all`);
  await page.evaluate(()=>window.siliconDevine.viewer.setLabels('none'));await page.waitForTimeout(100);assert.equal((await inspect()).visible.length,0);
  const selected=await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setLabels('auto');const c=v.cards.find(c=>{const a=v.nodes.find(n=>n.id===c.owner)?.attrs;return c.priority>0&&!a?.scopeRef&&!a?.mechanismRef;});if(!c)return null;v.focus(c.owner);return c.owner;});await page.waitForTimeout(850);
  if(selected){const focus=await inspect();assert.ok(focus.visible.some(c=>c.owner===selected&&c.onScreen),`${name}: focused primary caption must be on screen`);}
  await page.screenshot({path:`.qa/visibility-${name}.png`});
  report.push({name,auto:baseline.visible.length,all:all.visible.length,fullOrbit:true,focus:!!selected});
 }
 await checkView('mlp');
 for(const family of ['deepseek_v3','glm45','minimax_m1']){
  await selectExample(page,family);await checkView(`${family}-overview`);
  await selectModule(page,'moe_block');await selectModule(page,'decoder.self_attn');await checkView(`${family}-attention`);
 }
 await page.setViewportSize({width:390,height:844});await checkView('mobile-attention');
 assert.deepEqual(errors,[]);await writeFile('.qa/label-visibility.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}finally{await browser.close();}
