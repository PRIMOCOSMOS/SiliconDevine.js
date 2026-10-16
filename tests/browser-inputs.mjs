import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||undefined,headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1600,height:1050}}),errors=[],report=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error'&&/THREE|WebGL|shader/i.test(m.text()))errors.push(m.text());});
 await page.goto('http://127.0.0.1:5199');await page.waitForFunction(()=>window.siliconDevine?.viewer);
 async function capture(name){await page.waitForTimeout(350);report.push(await page.evaluate(name=>{const v=window.siliconDevine.viewer;return {name,stats:v.getStats(),cards:v.cards.filter(c=>c.group.visible).length,nodes:v.nodes.length,badge:document.querySelector('#structure-badge').textContent};},name));await page.screenshot({path:`.qa/inputs-${name}.png`});}
 await page.locator('#file').setInputFiles('.qa/expnet-graph.json');await page.waitForFunction(()=>window.siliconDevine.model.name==='ExpNet');await capture('expnet');
 assert.ok(report[0].nodes>2&&report[0].nodes<15,'Nested return should expose native module composition');
 assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.cards.every(c=>c.group.children.length===1&&c.group.userData.annotationStyle==='world-text')),'Only fixed text; no backing slabs');
 const anchor=await page.evaluate(()=>{const v=window.siliconDevine.viewer;return v.cards.map(c=>({p:c.group.position.toArray(),q:c.group.quaternion.toArray(),text:c.group.children[0].quaternion.toArray(),plaque:c.group.userData.integratedPlaque}));});
 assert.ok(anchor.length&&anchor.every(c=>c.plaque));
 await page.evaluate(()=>window.siliconDevine.viewer.orbit(.3,.1));await page.waitForTimeout(150);
 const after=await page.evaluate(()=>window.siliconDevine.viewer.cards.map(c=>({p:c.group.position.toArray(),q:c.group.quaternion.toArray(),text:c.group.children[0].quaternion.toArray(),plaque:c.group.userData.integratedPlaque})));
 assert.deepEqual(after,anchor);await capture('orbit');
 for(const type of ['symbolic','inventory']){
  await page.locator('#file').setInputFiles(`.qa/unknown-${type}.json`);
  await page.waitForFunction(mode=>window.siliconDevine.model.provenance?.topology===mode,type==='symbolic'?'fx-dependencies':'module-inventory');
  await capture(type);assert.ok(await page.locator('#structure-badge').isVisible());
  const m=await page.evaluate(()=>window.siliconDevine.model);assert.ok(m.tensors.every(t=>!t.data));
  if(type==='inventory')assert.ok(m.nodes.every(n=>!n.inputs.length));
 }
 await page.setViewportSize({width:430,height:900});await page.waitForTimeout(100);await page.evaluate(()=>window.siliconDevine.viewer.fit());await capture('mobile');
 assert.ok(report.every(r=>r.nodes>0&&r.stats.cells>0));assert.deepEqual(errors,[]);
 await writeFile('.qa/inputs-browser.json',JSON.stringify({report,errors},null,2));console.log(JSON.stringify(report));
}finally{await browser.close();}
