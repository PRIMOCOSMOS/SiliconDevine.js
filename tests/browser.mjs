import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
await mkdir('.qa',{recursive:true});
const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL??'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
const errors=[];page.on('pageerror',e=>errors.push(e.message));const result=[];
await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5180/');await page.waitForFunction(()=>window.siliconDevine?.viewer.getStats().connections>100);
for(const example of ['declarative','conv2d','conv3d','attention']){
  await page.selectOption('#example',example);await page.waitForTimeout(800);
  const stats=await page.evaluate(()=>window.siliconDevine.viewer.getStats());
  assert.ok(stats.cells>0);assert.ok(stats.connections>0);assert.ok(stats.drawCalls<300);
  await page.screenshot({path:`.qa/${example}-desktop.png`,fullPage:true});result.push({example,...stats});
}
// Hovering a region must immediately take over from the automatic scheduler.
const region=await page.evaluate(()=>{const v=window.siliconDevine.viewer,r=v.regions[1],p=r.box.getCenter(v.camera.position.clone());p.z=2;p.project(v.camera);const b=document.querySelector('#stage').getBoundingClientRect();return{id:r.node,x:b.left+(p.x+1)*b.width/2,y:b.top+(1-p.y)*b.height/2};});
await page.mouse.move(region.x,region.y);await page.waitForTimeout(300);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getStats().active),region.id);
await page.mouse.move(10,10);await page.click('#play');
await page.click('#zoom-in');await page.waitForTimeout(150);const camera=await page.evaluate(()=>window.siliconDevine.viewer.camera.position.toArray());
await page.waitForTimeout(650);const after=await page.evaluate(()=>window.siliconDevine.viewer.camera.position.toArray());assert.ok(Math.max(...camera.map((v,i)=>Math.abs(v-after[i])))<1e-7,'camera drift');
await page.selectOption('#example','declarative');await page.fill('#batch','3');await page.fill('#channels','10');await page.click('#rebuild');assert.deepEqual(await page.evaluate(()=>window.siliconDevine.model.tensors[0].shape),[3,10]);
await page.setInputFiles('#file',{name:'bad.json',mimeType:'application/json',buffer:Buffer.from('{"format":"wrong"}')});await page.waitForTimeout(100);assert.ok(await page.locator('#error').isVisible());
assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getStats().totalNodes),3,'bad import destroyed valid model');
await page.setViewportSize({width:390,height:844});await page.click('#rebuild');await page.click('#fit');await page.waitForTimeout(600);await page.screenshot({path:'.qa/mobile.png',fullPage:true});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
assert.deepEqual(errors,[]);await writeFile('.qa/browser-results.json',JSON.stringify({result,hover:region.id,cameraStable:true,errors},null,2));await browser.close();console.log(JSON.stringify({result,hover:region.id,cameraStable:true,errors},null,2));
