import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||undefined,headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1600,height:1050}}),errors=[],report=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('console',m=>{if(m.type()==='error'&&/THREE|WebGL|shader/i.test(m.text()))errors.push(m.text());});
 await page.goto('http://127.0.0.1:5199');await page.waitForFunction(()=>window.siliconDevine?.viewer);
 async function capture(name){
  await page.waitForTimeout(300);
  report.push(await page.evaluate(name=>{const v=window.siliconDevine.viewer;const metal=[];v.root.traverse(o=>{if(o.userData.metalScaffold)metal.push({count:o.count,metalness:o.material.metalness,roughness:o.material.roughness,transparent:o.material.transparent});});return {name,stats:v.getStats(),metal,environment:!!v.scene.environment,planes:v.planes.size};},name));
  await page.screenshot({path:`.qa/metal-${name}.png`});
 }
 await page.locator('#file').setInputFiles('.qa/generic-RenamedBranches-3.json');
 await page.waitForFunction(()=>window.siliconDevine.model.name==='RenamedBranches');
 await capture('towers');
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.focus(v.nodes.find(n=>n.attrs?.compositionVisual?.stages).id);});
 await capture('details');
 await page.evaluate(()=>window.siliconDevine.viewer.orbit(.55,.16));await capture('reflections');
 await page.setViewportSize({width:430,height:900});await page.waitForTimeout(100);
 await page.evaluate(()=>window.siliconDevine.viewer.fit());await capture('mobile');
 await page.setViewportSize({width:1600,height:1050});
 await page.locator('#file').setInputFiles('.qa/mmhvae-composition.json');
 await page.waitForFunction(()=>window.siliconDevine.model.name==='MHVAE2D');await capture('large');
 for(const r of report){assert.equal(r.metal.length,1);assert.equal(r.metal[0].count,r.planes*8);assert.ok(r.metal[0].metalness>.9);assert.equal(r.metal[0].transparent,false);assert.ok(r.environment);assert.ok(r.stats.cells>0);}
 assert.deepEqual(errors,[]);await writeFile('.qa/metal-browser.json',JSON.stringify({report,errors},null,2));
 console.log(JSON.stringify(report));
 await page.evaluate(()=>window.siliconDevine.viewer.dispose());
}finally{await browser.close();}
