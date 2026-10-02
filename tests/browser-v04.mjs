import {selectExample,selectModule} from './browser-shell.mjs';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1520,height:1050}}),errors=[],report={};page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5186');
 for(const name of ['conv2d','conv3d','official_sab','official_isab','official_llama']){
  await selectExample(page,name);await page.waitForFunction(n=>window.siliconDevine?.model.name===n,name);await page.evaluate(()=>window.siliconDevine.viewer.setPlaying(false));
  if(name.startsWith('conv')){
   await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.focus(v.nodes.find(n=>n.op.startsWith('conv')).id)});await page.waitForTimeout(50);await page.evaluate(()=>window.siliconDevine.viewer.setProgress(.3));await page.waitForTimeout(100);
   const s=await page.evaluate(()=>window.siliconDevine.viewer.getStats());assert.ok(s.receptiveField.visibleSamples>0);report[name]=s;await page.screenshot({path:`.qa/v04-${name}.png`,fullPage:true});
   if(name==='conv2d'){assert.ok(s.residuals>0);await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.focus(v.nodes.find(n=>n.attrs?.residualInput).id)});await page.waitForTimeout(120);await page.screenshot({path:'.qa/v04-residual.png',fullPage:true});}
  }else{
   const scope=name==='official_sab'?'mab':name==='official_isab'?'mab0':'model.layers.0.self_attn';await selectModule(page,scope);await page.waitForTimeout(120);
   const projections=await page.evaluate(()=>window.siliconDevine.viewer.nodes.filter(n=>n.attrs?.projectionRoles).flatMap(n=>n.attrs.projectionRoles));for(const r of ['Q','K','V'])assert.ok(projections.includes(r));
   for(const role of ['probability','context']){
    await page.evaluate(r=>{const v=window.siliconDevine.viewer;v.focus(v.nodes.find(n=>n.attrs?.attentionRole===r).id)},role);await page.waitForTimeout(70);await page.evaluate(()=>window.siliconDevine.viewer.setProgress(.23));await page.waitForTimeout(120);
    const s=await page.evaluate(()=>window.siliconDevine.viewer.getStats());report[name+role]=s;if(role==='context')assert.ok(s.attention?.contributors>0);else assert.ok(s.mechanism?.visibleItems>0);
    await page.screenshot({path:`.qa/v04-${name}-${role}.png`,fullPage:true});
   }
   if(name==='official_llama'){await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.focus(v.nodes.find(n=>n.op==='function').id)});await page.waitForTimeout(80);assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().function));await page.click('#parent-module');await page.waitForTimeout(80);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().function),undefined);}
   if(name==='official_isab'){await selectModule(page,'mab1');await page.waitForTimeout(80);assert.deepEqual(await page.evaluate(()=>{const v=window.siliconDevine.viewer,n=v.nodes.find(n=>n.attrs?.attentionRole==='probability');return v.tensors.get(n.outputs[0]).shape.slice(-2)}),[6,3]);}
  }
 }
 await page.setViewportSize({width:390,height:844});await page.click('#fit');await page.waitForTimeout(150);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/v04-mobile.png',fullPage:true});assert.deepEqual(errors,[]);report.errors=errors;await writeFile('.qa/v04-browser.json',JSON.stringify(report,null,2));console.log('Official attention, receptive fields, residuals, mobile and console passed.');
}finally{await browser.close()}
