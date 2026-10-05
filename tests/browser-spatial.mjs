import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import('file:///C:/Users/30246/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],report={};page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5200');await page.waitForFunction(()=>window.siliconDevine?.model);
 await page.click('#catalog-toggle');await page.locator('.model-family.dense>summary').click();await page.click('[data-model="spatial"]');
 await page.waitForFunction(()=>window.siliconDevine.model.name==='空间采样与分块重组');await page.waitForTimeout(900);
 for(const op of ['interpolate','grid_sample2d','unfold2d','fold2d']){
  await page.evaluate(op=>{const v=window.siliconDevine.viewer;v.selected=v.nodes.find(n=>n.op===op).id;v.setPlaying(false);v.setProgress(.3);},op);await page.waitForTimeout(200);
  const stats=await page.evaluate(()=>window.siliconDevine.viewer.getStats());report[op]=stats;assert.ok(stats.connections>0&&stats.cells>0);assert.ok(stats.drawCalls<500);
  if(op!=='fold2d')assert.ok(stats.receptiveField?.visibleSamples>0);
  if(op==='grid_sample2d')await page.screenshot({path:'.qa/v071-spatial.png'});
 }
 await page.click('#inspector-toggle');await page.locator('#operator-support>summary').click();assert.match(await page.locator('#operator-support pre').textContent(),/4 \/ 4/);
 assert.deepEqual(errors,[]);await writeFile('.qa/spatial-browser.json',JSON.stringify(report,null,2));console.log('Spatial library entry, sampling neighbourhoods, weighted motion and support report passed.');
}finally{await browser.close();}
