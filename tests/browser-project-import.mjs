import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5197');
 await page.waitForFunction(()=>window.siliconDevine?.viewer.nodes.length>0);
 await page.locator('#file').setInputFiles('.qa/import-model.json');
 await page.waitForFunction(()=>window.siliconDevine?.model.name==='VisionModel');
 assert.match(await page.locator('#source').textContent(),/自动推断的示例输入/);
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setPlaying(false);v.focus(v.nodes.find(n=>n.op==='conv2d').id);});
 await page.waitForTimeout(1100);
 const result=await page.evaluate(()=>{const v=window.siliconDevine.viewer;return {nodes:v.nodes.length,assembly:v.getAssembly(v.selected),cards:v.cards.filter(c=>c.group.visible).map(c=>c.id),stats:v.getStats(),camera:v.camera.position.toArray(),source:window.siliconDevine.model.discovery};});
 assert.ok(result.stats.cells>0);assert.ok(result.camera.every(Number.isFinite));
 assert.ok(result.assembly.related.every(id=>result.cards.includes(id)));
 assert.ok(result.source.files.some(f=>f.module.endsWith('.blocks')));
 await page.screenshot({path:'.qa/project-import-browser.png'});
 assert.deepEqual(errors,[]);
 await writeFile('.qa/project-import-browser.json',JSON.stringify(result,null,2));
 console.log(JSON.stringify({nodes:result.nodes,cells:result.stats.cells,drawCalls:result.stats.drawCalls,cards:result.cards.length}));
}finally{await browser.close();}
