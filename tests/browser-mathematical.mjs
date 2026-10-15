import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[],report=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error'&&/THREE|WebGL|shader/i.test(m.text()))errors.push(m.text());});
 await page.goto('http://127.0.0.1:5199');await page.waitForFunction(()=>window.siliconDevine?.viewer);
 const load=async(file,name)=>{
  const expected=JSON.parse(await readFile(`.qa/${file}.json`,'utf8'));
  await page.locator('#file').setInputFiles(`.qa/${file}.json`);
  await page.waitForFunction(e=>{const m=window.siliconDevine.model;return m?.name===e.name&&m.producer?.backend===e.backend&&m.nodes.length===e.count;},
   {name,backend:expected.producer?.backend,count:expected.nodes.length});
 };
 const shot=async(name)=>{
  await page.waitForTimeout(450);
  report.push(await page.evaluate(name=>({name,stats:window.siliconDevine.viewer.getStats(),labels:window.siliconDevine.viewer.getNavigation().nodes.map(n=>n.attrs?.displayLabel??n.name),inputText:document.querySelector('#model-inputs').textContent,overflow:document.documentElement.scrollWidth>innerWidth}),name));
  await page.screenshot({path:`.qa/math-${name}.png`});
 };
 await load('expnet-graph','ExpNet');
 await page.evaluate(()=>{const s=window.siliconDevine;const prep=s.model.composition.units.find(u=>u.name==='张量准备与变换');s.viewer.showModule(prep.id);});
 await shot('preparation');assert.equal(report.at(-1).labels.length,1);
 await page.selectOption('#representation','raw');assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().nodes.length),29);
 await page.selectOption('#representation','operators');await page.locator('#inputs-toggle').click();
 await shot('inputs');assert.equal(await page.locator('#model-inputs .input-parameter').count(),6);
 assert.match(report.at(-1).inputText,/signal/);assert.match(report.at(-1).inputText,/batch/);assert.match(report.at(-1).inputText,/ones_like/);
 await page.locator('#inspector-close').click();
 await load('contract-conv-fill','ConstrainedConvolution');await page.selectOption('#representation','operators');
 const conv=await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().nodes[0]);assert.equal(conv.op,'conv2d');
 await page.evaluate(id=>window.siliconDevine.viewer.focus(id),conv.id);await shot('conv-focus');
 assert.ok(report.at(-1).stats.receptiveField?.visibleSamples>0);
 assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().function),undefined);
 await load('contract-attention','MaskedAttention');await page.selectOption('#representation','operators');await shot('attention');
 assert.equal(report.at(-1).labels.length,3);assert.ok(report.at(-1).labels.some(n=>n.includes('QKᵀ')));
 await load('contract-nested','InputModel');await page.locator('#inputs-toggle').click();await shot('nested-inputs');
 assert.match(report.at(-1).inputText,/samples\['images'\]\[0\]/);assert.match(report.at(-1).inputText,/gain/);
 await page.setViewportSize({width:430,height:900});await page.locator('#inputs-toggle').click();await shot('mobile-inputs');
 assert.equal(report.at(-1).overflow,false);
 await load('contract-unknown','InputModel');await page.locator('#inputs-toggle').click();
 await shot('unknown-inputs');assert.match(report.at(-1).inputText,/尺寸待定|形状与类型待定/);
 assert.ok(!report.at(-1).labels.some(n=>/invert|masked_fill/.test(n)));
 assert.equal(report.at(-1).overflow,false);
 await page.locator('#presentation').click();await page.locator('#inputs-toggle').click();
 assert.ok(await page.locator('#model-inspector').isVisible(),'Input control opens the panel from presentation mode');
 assert.deepEqual(errors,[]);await writeFile('.qa/math-browser.json',JSON.stringify({report,errors},null,2));
 console.log(JSON.stringify(report.map(r=>({name:r.name,labels:r.labels,overflow:r.overflow,drawCalls:r.stats.drawCalls,cells:r.stats.cells}))));
}finally{await browser.close();}
