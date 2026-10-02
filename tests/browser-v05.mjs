import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'file:///C:/Users/30246/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1520,height:1080}}),errors=[],report={};page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5186');await page.waitForFunction(()=>window.siliconDevine?.model);
 assert.equal(await page.locator('aside').isVisible(),false);
 await page.waitForTimeout(850);
 await page.screenshot({path:'.qa/v05-workbench.png',fullPage:true});
 await page.click('#catalog-toggle');await page.waitForTimeout(500);await page.screenshot({path:'.qa/v05-library.png',fullPage:true});
 await page.click('#library-return');await page.click('#inspector-toggle');
 await page.fill('#channels','6');await page.fill('#batch','3');await page.click('#rebuild');
 assert.deepEqual(await page.evaluate(()=>window.siliconDevine.model.tensors.find(t=>t.id==='x').shape),[3,6]);
 await page.waitForTimeout(900);
 const cameraBefore=await page.evaluate(()=>window.siliconDevine.viewer.camera.position.toArray());
 await page.screenshot({path:'.qa/v05-inspector.png',fullPage:true});await page.click('#inspector-close');
 await page.waitForTimeout(400);assert.deepEqual(await page.evaluate(()=>window.siliconDevine.viewer.camera.position.toArray()),cameraBefore);
 for(const name of ['official_vae','conv_vae','conditional_vae']){
  await page.click('#catalog-toggle');await page.click(`[data-model="${name}"]`);await page.waitForFunction(n=>window.siliconDevine?.model.name===n,name);
  await page.waitForTimeout(700);
  assert.equal(await page.locator('#representation').inputValue(),'architecture');
  const id=await page.evaluate(()=>window.siliconDevine.viewer.nodes.find(n=>n.op==='gaussian_sample').id);
  // Hover lock uses the same selected region as all other nodes without entering it.
  await page.evaluate(id=>{const v=window.siliconDevine.viewer;v.selected=id;v.setProgress(.5)},id);
  await page.waitForTimeout(150);report[name]=await page.evaluate(()=>window.siliconDevine.viewer.getStats());assert.ok(report[name].gaussian);
  assert.ok(Math.abs(report[name].gaussian.mean+report[name].gaussian.std*report[name].gaussian.epsilon-report[name].gaussian.sample)<1e-5);
  await page.screenshot({path:`.qa/v05-${name}.png`,fullPage:true});
  await page.evaluate(id=>window.siliconDevine.viewer.focus(id),id);await page.waitForTimeout(120);
  assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().function));assert.ok(await page.locator('#formula .katex').count());
  await page.click('#parent-module');
 }
 await page.click('#presentation');assert.equal(await page.locator('#presentation').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('aside').isVisible(),false);await page.screenshot({path:'.qa/v05-presentation.png',fullPage:true});await page.keyboard.press('Escape');assert.equal(await page.locator('#presentation').getAttribute('aria-pressed'),'false');
 await page.click('#catalog-toggle');assert.equal(await page.locator('aside').isVisible(),true);await page.click('#catalog-toggle');assert.equal(await page.locator('aside').isVisible(),false);
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(1000);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/v05-mobile.png',fullPage:true});
 await page.click('#catalog-toggle');await page.waitForTimeout(500);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/v05-mobile-library.png',fullPage:true});
 await page.click('#library-return');await page.waitForTimeout(1000);await page.screenshot({path:'.qa/v05-mobile-return.png',fullPage:true});await page.click('#inspector-toggle');assert.ok(await page.locator('#model-inspector').isVisible());assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.emulateMedia({reducedMotion:'reduce'});await page.reload();await page.waitForFunction(()=>window.siliconDevine?.model);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.isPlaying),false);
 assert.deepEqual(errors,[]);report.errors=errors;await writeFile('.qa/v05-browser.json',JSON.stringify(report,null,2));console.log('VAE hierarchy, real sample motion, presentation, mobile and reduced-motion checks passed.');
}finally{await browser.close();}
