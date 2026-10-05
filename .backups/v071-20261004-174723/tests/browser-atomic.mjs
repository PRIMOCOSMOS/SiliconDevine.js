import {selectExample,selectModule} from './browser-shell.mjs';
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[],report={};page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5193');
 for(const family of ['deepseek_v3','glm45','minimax_m1']){
  await selectExample(page,family);await page.waitForFunction(f=>window.siliconDevine?.model.architecture?.family===f,family);
  await selectModule(page,'moe_block');await page.waitForTimeout(200);
  await page.evaluate(()=>window.siliconDevine.viewer.setPlaying(false));
  const point=await page.evaluate(()=>{const v=window.siliconDevine.viewer,r=v.regions.find(r=>r.node==='attention'),p=r.box.getCenter(v.camera.position.clone());p.z=r.box.max.z;const q=p.project(v.camera),rect=document.querySelector('#stage').getBoundingClientRect();return{x:rect.left+(q.x+1)*rect.width/2,y:rect.top+(1-q.y)*rect.height/2};});
  await page.mouse.move(point.x,point.y);await page.waitForFunction(()=>window.siliconDevine.viewer.getStats().atomic?.node==='attention');
  const preview=await page.evaluate(()=>{const v=window.siliconDevine.viewer;return {stats:v.getStats(),geometry:v.renderer.info.memory.geometries,oldHidden:!v.structureMotion.units.get('attention').visible};});
  assert.ok(preview.oldHidden);assert.ok(preview.stats.atomic.atoms>=20);assert.ok(preview.stats.atomic.connections>100);assert.ok(preview.stats.atomic.cells<3072);
  await page.screenshot({path:`.qa/atomic-${family}-preview.png`});
  // Repeated hover switches release resources instead of accumulating geometry.
  for(let i=0;i<6;i++)await page.evaluate(i=>{const v=window.siliconDevine.viewer;v.structureMotion.update(i%2?'attention':'norm1',2);},i);
  await page.evaluate(()=>window.siliconDevine.viewer.structureMotion.update('attention',2));await page.waitForTimeout(100);
  assert.ok(await page.evaluate(b=>window.siliconDevine.viewer.renderer.info.memory.geometries<=b+4,preview.geometry));
  await page.mouse.click(point.x,point.y);await page.waitForTimeout(150);
  assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().scope),'attention');
  await page.click('#mechanism-toggle');await page.mouse.move(5,5);await page.waitForTimeout(200);
  const full=await page.evaluate(()=>{const v=window.siliconDevine.viewer;return {nav:v.getNavigation(),stats:v.getStats(),total:v.getDisplayedModel().nodes.length};});
  assert.equal(full.nav.stage,-1);assert.equal(full.nav.nodes.length,full.total);
  for(const op of ['linear','matmul','softmax'])assert.ok(full.nav.nodes.some(n=>n.op===op));
  await page.screenshot({path:`.qa/atomic-${family}-full.png`});report[family]={preview:preview.stats,full:full.stats};
 }
 await page.setViewportSize({width:390,height:844});await page.click('#fit');await page.waitForTimeout(100);
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/atomic-mobile.png'});
 assert.deepEqual(errors,[]);await writeFile('.qa/atomic-browser.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}finally{await browser.close();}
