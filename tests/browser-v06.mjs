import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'file:///C:/Users/30246/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try {
 const page=await browser.newPage({viewport:{width:1520,height:1080}}),errors=[],report={};page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5196');await page.waitForFunction(()=>window.siliconDevine?.model);
 for(const family of ['deepseek_v3','glm45','minimax_m1']) {
  await page.click('#catalog-toggle');const button=page.locator(`[data-model="${family}"]`);await button.evaluate(b=>b.closest('details').open=true);await button.click();
  await page.waitForFunction(f=>window.siliconDevine.model.architecture?.family===f,family);await page.waitForTimeout(800);
  assert.equal(await page.locator('#structure-badge').isVisible(),true);
  report[family]={root:await page.evaluate(()=>window.siliconDevine.viewer.getStats()),scopes:{}};
  await page.screenshot({path:`.qa/v06-${family}.png`,fullPage:true});
  const position=await page.evaluate(()=>window.siliconDevine.viewer.camera.position.toArray());
  await page.click('#catalog-toggle');await page.click('#library-return');await page.waitForTimeout(500);
  assert.deepEqual(await page.evaluate(()=>window.siliconDevine.viewer.camera.position.toArray()),position);
  // Every compact scope renders without expanding repeat or expert counts.
  const scopes=await page.evaluate(()=>Object.keys(window.siliconDevine.model.architecture.scopes));
  for(const scope of scopes) {
   await page.evaluate(s=>window.siliconDevine.viewer.showModule(s),scope);await page.waitForTimeout(100);
   const stats=await page.evaluate(()=>window.siliconDevine.viewer.getStats());
   report[family].scopes[scope]=stats;assert.ok(stats.cells>0&&stats.cells<700);assert.ok(stats.drawCalls<180);
  }
  await page.evaluate(()=>window.siliconDevine.viewer.showModule('moe_block'));await page.waitForTimeout(200);
  // Project an actual region point to screen and use a real pointer movement.
  const point=await page.evaluate(()=>{const v=window.siliconDevine.viewer,p=v.planes.get('attention:out').center;const vec=v.controls.target.clone().set(...p).project(v.camera),r=v.renderer.domElement.getBoundingClientRect();return {x:r.x+(vec.x+1)*r.width/2,y:r.y+(1-vec.y)*r.height/2};});
  await page.mouse.move(point.x,point.y);await page.waitForTimeout(100);
  assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getStats().active),'attention');
  await page.mouse.click(point.x,point.y);await page.waitForTimeout(250);
  assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().scope),'attention');
  await page.screenshot({path:`.qa/v06-${family}-attention.png`,fullPage:true});
  await page.evaluate(()=>window.siliconDevine.viewer.parentModule());
  assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().scope),'moe_block');
 }
 await page.evaluate(()=>window.siliconDevine.viewer.showModule('lightning'));await page.waitForTimeout(200);await page.screenshot({path:'.qa/v06-lightning.png',fullPage:true});
 await page.click('#inspector-toggle');await page.evaluate(()=>window.siliconDevine.viewer.focus('recurrence'));await page.waitForTimeout(200);assert.ok(await page.locator('#formula .katex').count());await page.screenshot({path:'.qa/v06-inspector.png',fullPage:true});
 await page.click('#inspector-close');await page.setViewportSize({width:390,height:844});await page.waitForTimeout(200);await page.evaluate(()=>window.siliconDevine.viewer.fit());await page.waitForTimeout(200);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/v06-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);report.errors=errors;await writeFile('.qa/v06-browser.json',JSON.stringify(report,null,2));console.log('All large-model scopes, bounded rendering, real pointer hover/click, return, math and mobile passed.');
} finally {await browser.close();}
