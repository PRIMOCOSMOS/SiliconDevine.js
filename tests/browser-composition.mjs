import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1600,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('console',m=>{if(m.type()==='error'&&/THREE|WebGL|shader/i.test(m.text()))errors.push(m.text());});
 await page.goto('http://127.0.0.1:5199');
 await page.waitForFunction(()=>window.siliconDevine?.viewer);
 await page.locator('#file').setInputFiles('.qa/mmhvae-composition.json');
 await page.waitForFunction(()=>window.siliconDevine?.model.name==='MHVAE2D');
 const report=[];
 async function capture(name){
   await page.waitForTimeout(600);
   report.push(await page.evaluate(name=>{const v=window.siliconDevine.viewer;return {name,scope:v.getNavigation(),stats:v.getStats(),camera:v.camera.position.toArray()};},name));
   await page.screenshot({path:`.qa/composition-${name}.png`});
 }
 await capture('overview');
 const distributions=await page.evaluate(()=>{const ds=[];window.siliconDevine.viewer.root.traverse(o=>{if(o.userData.distribution)ds.push(o.userData.distribution);});return ds;});
 assert.ok(distributions.length>=5);assert.ok(distributions.every(d=>Number.isFinite(d.mean)&&d.variance>0));
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.focus(v.nodes.find(n=>n.attrs?.compositionVisual?.stages?.length===7).id);});
 await capture('encoder');
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.focus(v.nodes.find(n=>n.name.includes('BlockEncoder')).id);});
 await capture('block');
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.parentModule();v.parentModule();v.focus(v.nodes.find(n=>n.name==='循环计算 · 1/6').id);});
 await capture('latent');
 for(const r of report){assert.ok(r.stats.cells>0);assert.ok(r.camera.every(Number.isFinite));}
 assert.equal(report[0].scope.nodes.some(n=>n.op==='opaque'),false);
 assert.ok(report[2].scope.nodes.some(n=>n.op==='conv2d'));
 await page.evaluate(()=>window.siliconDevine.viewer.showModule(''));
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;const heads=v.nodes.filter(n=>n.attrs?.compositionVisual?.family&&n.attrs?.compositionVisual?.distribution);if(heads.length!==4)throw Error('Expected four independent heads');v.focus(heads[0].id);});
 await capture('probability');
 await page.evaluate(()=>window.siliconDevine.viewer.showModule(''));
 await page.locator('#file').setInputFiles('.qa/generic-RenamedBranches-3.json');
 await page.waitForFunction(()=>window.siliconDevine.model.name==='RenamedBranches');
 await capture('generic');
 const generic=await page.evaluate(()=>{const v=window.siliconDevine.viewer;return {mode:v.getNavigation().representation,towers:v.nodes.filter(n=>n.attrs?.compositionVisual?.stages).length};});
 assert.equal(generic.mode,'architecture');assert.equal(generic.towers,3);
 const before=await page.evaluate(()=>window.siliconDevine.viewer.cards.map(c=>c.group.position.toArray()));
 await page.evaluate(()=>window.siliconDevine.viewer.orbit(.5,.1));
 await page.waitForTimeout(100);
 const after=await page.evaluate(()=>window.siliconDevine.viewer.cards.map(c=>c.group.position.toArray()));
 assert.deepEqual(after,before,'Card anchors must not move while orbiting');

 await page.setViewportSize({width:430,height:900});
 await capture('mobile');
 assert.deepEqual(errors,[]);
 await writeFile('.qa/composition-browser.json',JSON.stringify({report,errors},null,2));
 console.log(JSON.stringify(report.map(r=>({name:r.name,scope:r.scope.scopeName,nodes:r.scope.nodes.length,stats:r.stats}))));
}finally{await browser.close();}
