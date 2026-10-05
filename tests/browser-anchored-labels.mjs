import {selectExample,selectModule} from './browser-shell.mjs';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5195');await page.waitForFunction(()=>window.siliconDevine?.viewer.nodes.length>0);
 await page.waitForTimeout(800);
 const snap=()=>page.evaluate(()=>{const v=window.siliconDevine.viewer;return v.cards.map(c=>({id:c.id,owner:c.owner,p:c.group.position.toArray(),s:c.group.scale.toArray(),q:c.group.quaternion.toArray()}));});
 const before=await snap();
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setPlaying(false);v.zoom(1.2);const axis=v.camera.position.clone().set(0,1,0),d=v.camera.position.clone().sub(v.controls.target).applyAxisAngle(axis,.38);v.camera.position.copy(v.controls.target).add(d);v.controls.update();});await page.waitForTimeout(200);
 assert.deepEqual(await snap(),before,'Orbit/zoom must never relocate, rotate or rescale world-space labels');
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setLabels('all');v.setProgress(.8);});await page.waitForTimeout(100);assert.deepEqual(await snap(),before,'Animation and label modes must preserve anchors');
 await page.evaluate(()=>window.siliconDevine.viewer.setLabels('none'));await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.cards.filter(c=>c.group.visible).length),0);
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setLabels('auto');const n=v.nodes.find(n=>v.activationCurves.has(n.id));v.focus(n.id);v.setProgress(.4);});await page.waitForTimeout(850);
 const plot=await page.evaluate(()=>{const v=window.siliconDevine.viewer,p=v.activationCurves.get(v.active),s=v.getStats().sample;return {input:s.input,output:s.output,inputPoint:p.inputMarker.position.toArray(),outputPoint:p.marker.position.toArray(),extent:p.extent,yExtent:p.yExtent,w:p.halfWidth,h:p.halfHeight,values:p.points.map(p=>p.toArray()),visible:p.group.visible};});
 assert.ok(plot.visible);assert.equal(plot.inputPoint[1],0);assert.ok(Math.abs(plot.outputPoint[0]-plot.input/plot.extent*plot.w)<1e-8);assert.ok(Math.abs(plot.outputPoint[1]-plot.output/plot.yExtent*plot.h)<1e-8);assert.ok(plot.values.every(p=>p.every(Number.isFinite)));
 await page.screenshot({path:'.qa/anchored-function-desktop.png'});
 const label=await page.evaluate(()=>{const v=window.siliconDevine.viewer,c=v.cards.find(c=>c.group.visible&&c.owner===v.active);if(!c)return null;const r=v.renderer.domElement.getBoundingClientRect(),p=c.group.getWorldPosition(v.camera.position.clone()).project(v.camera);return {x:r.left+(p.x+1)*r.width/2,y:r.top+(1-p.y)*r.height/2,owner:c.owner};});
 assert.ok(label,'Selected operator needs a readable local caption');await page.mouse.move(label.x,label.y);await page.waitForTimeout(80);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.hovered),label.owner);
 await selectExample(page,'deepseek_v3');await selectModule(page,'moe_block');await selectModule(page,'decoder.self_attn');await page.waitForTimeout(850);
 const attentionBefore=await snap();await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.zoom(.8);v.controls.target.x+=.7;v.controls.update();});await page.waitForTimeout(150);assert.deepEqual(await snap(),attentionBefore);
 await page.click('#fit');await page.waitForTimeout(800);await page.screenshot({path:'.qa/anchored-attention-desktop.png'});
 await page.setViewportSize({width:390,height:844});await page.click('#fit');await page.waitForTimeout(800);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/anchored-mobile.png'});
 await page.setViewportSize({width:1480,height:1000});
 for(const name of ['official_vae','conv_vae','conditional_vae']){
  await selectExample(page,name);await page.waitForTimeout(150);
  await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.selected=v.nodes.find(n=>n.op==='gaussian_sample').id;v.setProgress(.35);});await page.waitForTimeout(100);
  const g=await page.evaluate(()=>{const v=window.siliconDevine.viewer,m=v.gaussianMotion,s=m.state;return {state:s,segments:m.curve.count,peak:m.points[32*3+1],marker:m.marker.position.toArray()};});
  assert.ok(g.state&&g.segments===64);assert.ok(Math.abs(g.state.mean+g.state.std*g.state.epsilon-g.state.sample)<1e-5);
  const width=1+(g.state.std-1)*g.state.phase;assert.ok(Math.abs(g.peak*width-.95*Math.min(1,g.state.std))<1e-6,'Gaussian height must follow the changing standard deviation');
 }
 assert.deepEqual(errors,[]);await writeFile('.qa/anchored-labels.json',JSON.stringify({fixedAnchors:true,hoverAssociation:true,plot},null,2));console.log('Fixed caption transforms, hover ownership, aligned function samples, label modes and mobile layout passed.');
}finally{await browser.close();}
