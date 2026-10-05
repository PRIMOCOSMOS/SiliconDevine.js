import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import('file:///C:/Users/30246/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
const report={};
try{
 const p=await browser.newPage({viewport:{width:1520,height:1080}}),errors=[];p.on('pageerror',e=>errors.push(e.message));
 await p.goto(process.env.DEMO_URL??'http://127.0.0.1:5200');await p.waitForFunction(()=>window.siliconDevine?.model);
 for(const family of ['deepseek_v3','glm45']){
  await p.selectOption('#example',family,{force:true});await p.waitForFunction(f=>window.siliconDevine.model.architecture?.family===f,family);
  await p.evaluate(()=>window.siliconDevine.viewer.setPlaying(false));
  await p.waitForTimeout(1000);
  for(const scope of ['root','attention']){
   await p.evaluate(s=>window.siliconDevine.viewer.showModule(s),scope);await p.waitForTimeout(900);
   report[`${family}-${scope}`]=await p.evaluate(()=>{const v=window.siliconDevine.viewer;return {stats:v.getStats(),cards:v.cards.map(c=>({visible:c.group.visible,composition:c.group.userData.composition,position:c.group.position.toArray()}))};});
   assert.ok(report[`${family}-${scope}`].cards.some(c=>c.visible));
   await p.screenshot({path:`.qa/v071-${family}-${scope}.png`});
  }
 }
 await p.evaluate(()=>window.siliconDevine.viewer.showModule('root'));await p.waitForTimeout(900);
 const layout=await p.evaluate(()=>{const r=id=>{const b=document.getElementById(id).getBoundingClientRect();return {top:b.top,bottom:b.bottom,left:b.left,right:b.right};};return {active:r('active'),badge:r('structure-badge'),stage:r('stage')};});
 assert.ok(layout.badge.top>=layout.active.bottom);assert.ok(layout.stage.top>=layout.badge.bottom);report.layout=layout;
 await p.click('#inspector-toggle');await p.waitForTimeout(500);assert.ok(await p.locator('#operator-support').isVisible());await p.screenshot({path:'.qa/v071-inspector.png'});await p.click('#inspector-close');
 await p.setViewportSize({width:390,height:844});await p.waitForTimeout(600);
 assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await p.screenshot({path:'.qa/v071-mobile.png',fullPage:true});
 await p.click('#inspector-toggle');assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await p.emulateMedia({reducedMotion:'reduce'});await p.reload();await p.waitForFunction(()=>window.siliconDevine?.model);assert.equal(await p.evaluate(()=>window.siliconDevine.viewer.isPlaying),false);
 assert.deepEqual(errors,[]);report.errors=errors;await writeFile('.qa/v071-browser.json',JSON.stringify(report,null,2));console.log('Structure labels, responsive header, support panel and reduced-motion checks passed.');
}finally{await browser.close();}
