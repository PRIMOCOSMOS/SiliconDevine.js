import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import('file:///C:/Users/30246/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1520,height:1080}}),errors=[],report={};page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5200');await page.waitForFunction(()=>window.siliconDevine?.model);await page.waitForTimeout(900);
 for(const family of ['deepseek_v3','glm45','minimax_m1']){
  await page.selectOption('#example',family,{force:true});await page.waitForFunction(f=>window.siliconDevine.model.architecture?.family===f,family);await page.waitForTimeout(900);
  await page.screenshot({path:`.qa/v07-${family}-root.png`});
  await page.evaluate(()=>window.siliconDevine.viewer.showModule('attention'));await page.waitForTimeout(900);await page.locator('#mechanism-toggle').click();await page.waitForTimeout(900);
  assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.isMechanism),true);assert.match(await page.locator('#structure-badge').textContent(),/实算/);
  report[family]={};
  for(const key of await page.evaluate(()=>Object.keys(window.siliconDevine.model.architecture.mechanisms))){
   await page.evaluate(k=>window.siliconDevine.viewer.showMechanism(k),key);await page.waitForTimeout(800);
   const stats=await page.evaluate(()=>window.siliconDevine.viewer.getStats());report[family][key]=stats;
   const stageCount=await page.evaluate(()=>window.siliconDevine.viewer.computationStages.length);
   for(let si=0;si<stageCount;si++){await page.evaluate(i=>window.siliconDevine.viewer.setComputationStage(i),si);await page.waitForTimeout(90);assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.getStats().cells>0));}
   if(stageCount)await page.evaluate(()=>window.siliconDevine.viewer.setComputationStage(0));await page.waitForTimeout(800);
   assert.ok(stats.cells>0&&stats.cells<=8192);assert.ok(stats.drawCalls<500,`${family}/${key} draw calls ${stats.drawCalls}`);assert.equal(stats.unknownTensors,0);assert.ok(stats.connections>0,`${family}/${key} needs actual links`);
   if(['attention','moe','lightning'].includes(key))await page.screenshot({path:`.qa/v07-${family}-${key}.png`});
  }
  await page.evaluate(()=>window.siliconDevine.viewer.showStructure());assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().scope),'attention');
 }
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.showMechanism('attention');v.setTour(true);v.tourClock=12;});await page.mouse.move(5,5);await page.waitForTimeout(1100);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().stage),1);await page.evaluate(()=>window.siliconDevine.viewer.setTour(false));
 // Real pointer hover selects a numeric operation; selecting it begins a smooth, interruptible camera flight.
 await page.evaluate(()=>window.siliconDevine.viewer.showMechanism('expert'));await page.waitForTimeout(850);
 const point=await page.evaluate(()=>{const v=window.siliconDevine.viewer,n=v.nodes.find(n=>n.op==='linear'),p=v.planes.get(n.outputs[0]).center;const q=v.controls.target.clone().set(...p).project(v.camera),r=v.renderer.domElement.getBoundingClientRect();return {id:n.id,x:r.x+(q.x+1)*r.width/2,y:r.y+(1-q.y)*r.height/2};});
 await page.mouse.move(point.x,point.y);await page.waitForTimeout(120);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getStats().active),point.id);
 await page.mouse.click(point.x,point.y);await page.waitForTimeout(80);assert.ok(await page.evaluate(()=>!!window.siliconDevine.viewer.cameraFlight));
 await page.mouse.wheel(0,100);await page.waitForTimeout(150);assert.equal(await page.evaluate(()=>!!window.siliconDevine.viewer.cameraFlight),false);
 await page.waitForTimeout(800);const pose=await page.evaluate(()=>window.siliconDevine.viewer.camera.position.toArray());await page.waitForTimeout(900);const stable=await page.evaluate(()=>window.siliconDevine.viewer.camera.position.toArray());assert.ok(pose.every((x,i)=>Math.abs(x-stable[i])<.01));
 for(const name of ['dynamic_conv','conditional_attention']){
  await page.selectOption('#example',name,{force:true});await page.waitForFunction(()=>!window.siliconDevine.model.architecture);await page.waitForTimeout(850);
  await page.screenshot({path:`.qa/v07-${name}.png`});report[name]=await page.evaluate(()=>window.siliconDevine.viewer.getStats());
  const op=name==='dynamic_conv'?'dynamic_conv2d':'matmul';await page.evaluate(op=>{const v=window.siliconDevine.viewer;v.selected=v.nodes.find(n=>n.op===op).id;},op);await page.waitForTimeout(400);
  if(name==='dynamic_conv')assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.getStats().receptiveField?.visibleSamples>0));
 }
 await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.siliconDevine.viewer.fit());await page.waitForTimeout(850);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/v07-mobile.png',fullPage:true});
 await page.emulateMedia({reducedMotion:'reduce'});await page.evaluate(()=>window.siliconDevine.viewer.fit());assert.equal(await page.evaluate(()=>!!window.siliconDevine.viewer.cameraFlight),false);
 assert.deepEqual(errors,[]);await writeFile('.qa/v07-browser.json',JSON.stringify({report,errors},null,2));console.log('Numerical LLM mechanisms, real hover, camera interruption, dynamic kernels, conditioning and mobile passed.');
}finally{await browser.close();}
