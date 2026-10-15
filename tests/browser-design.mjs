import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[],report=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error'&&/THREE|WebGL|shader/i.test(m.text()))errors.push(m.text());});
 await page.goto('http://127.0.0.1:5199');await page.waitForFunction(()=>window.siliconDevine?.viewer);
 const shot=async(name)=>{await page.waitForTimeout(300);report.push(await page.evaluate(name=>({name,stats:window.siliconDevine.viewer.getStats(),labels:window.siliconDevine.viewer.getNavigation().nodes.map(n=>n.attrs?.displayLabel??n.name)}),name));await page.screenshot({path:`.qa/design-${name}.png`});};
 await page.locator('#file').setInputFiles('.qa/expnet-graph.json');
 await page.waitForFunction(()=>window.siliconDevine.model.name==='ExpNet');
 await page.evaluate(()=>{const s=window.siliconDevine;const prep=s.model.composition.units.find(u=>u.name==='张量准备与变换');s.viewer.showModule(prep.id);});
 await shot('preparation');assert.equal(report.at(-1).labels.length,1);
 await page.selectOption('#representation','raw');
 assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().nodes.length),29);
 await page.selectOption('#representation','operators');await shot('mathematical');
 await page.locator('#file').setInputFiles('.qa/design-DesignerModel.json');
 await page.waitForFunction(()=>window.siliconDevine.model.name==='DesignerModel');
 await page.selectOption('#representation','operators');await shot('generic');
 const nav=await page.evaluate(()=>window.siliconDevine.viewer.getNavigation());
 const dot=nav.nodes.find(n=>n.attrs?.designKind==='dot_product');assert.ok(dot);
 await page.evaluate(id=>window.siliconDevine.viewer.focus(id),dot.id);await shot('dot-focus');
 assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().function),undefined,'Click focuses math without unfolding implementation calls');
 await page.setViewportSize({width:430,height:900});await page.evaluate(()=>window.siliconDevine.viewer.clearFocus());await shot('mobile');
 assert.deepEqual(errors,[]);await writeFile('.qa/design-browser.json',JSON.stringify({report,errors},null,2));console.log(JSON.stringify(report));
}finally{await browser.close();}
