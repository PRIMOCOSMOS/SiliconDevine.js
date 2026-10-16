import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {selectModule} from './browser-shell.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||undefined,headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const start=Date.now();
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5198/?live=1');
 await page.waitForFunction(()=>window.siliconDevine?.model.name==='MHVAE2D',{},{timeout:120000});
 const overview=await page.evaluate(()=>({stats:window.siliconDevine.viewer.getStats(),nodes:window.siliconDevine.model.nodes.length,source:document.querySelector('#source').textContent}));
 await page.screenshot({path:'.qa/mmhvae-overview.png'});
 const options=await page.locator('#modules option').evaluateAll(options=>options.map(o=>o.value));
 const scope=options.find(s=>s==='conv_blocks_context.us.0')??options.find(s=>s.startsWith('conv_blocks_context.us.'));
 assert.ok(scope,JSON.stringify(options.slice(0,30)));
 await selectModule(page,scope);
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setPlaying(false);const n=v.nodes.find(n=>n.op==='conv2d');if(n)v.focus(n.id);});
 await page.waitForTimeout(900);
 const detail=await page.evaluate(()=>({stats:window.siliconDevine.viewer.getStats(),scope:window.siliconDevine.viewer.getNavigation().scope,camera:window.siliconDevine.viewer.camera.position.toArray()}));
 assert.ok(detail.stats.cells>0);assert.ok(detail.camera.every(Number.isFinite));
 assert.deepEqual(errors,[]);
 await page.screenshot({path:'.qa/mmhvae-detail.png'});
 const report={elapsedMs:Date.now()-start,overview,detail,errors};
 await writeFile('.qa/mmhvae-browser.json',JSON.stringify(report,null,2));
 console.log(JSON.stringify(report));
}finally{await browser.close();}
