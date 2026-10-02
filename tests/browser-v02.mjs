import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
await mkdir('.qa/ide_project',{recursive:true});
const source=(outputs)=>`import torch\nfrom torch import nn\ndef build_model():\n torch.manual_seed(7)\n return nn.Sequential(nn.Linear(64,${outputs}),nn.GELU()),(torch.randn(2,64),)\n`;
const file=resolve('.qa/ide_project/model.py');await writeFile(file,source(8));
const child=spawn('python',['-m','silicondevine.watch',file,'--port','5183','--static-dir',resolve('demo-dist')],{env:{...process.env,PYTHONPATH:resolve('python')},windowsHide:true,stdio:['ignore','pipe','pipe']});
let log='';child.stdout.on('data',v=>log+=v);child.stderr.on('data',v=>log+=v);
const wait=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn){let error;for(let i=0;i<120;i++){try{const value=await fn();if(value)return value;}catch(e){error=e;}await wait(300);}throw error??Error('Timed out: '+log);}
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 await until(async()=>{const r=await fetch('http://127.0.0.1:5183/api/status');return(await r.json()).ready;});
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:5183/?live=1');await page.waitForFunction(()=>window.siliconDevine?.model.producer?.backend==='pytorch-fx');
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.focus('_0');v.setPlaying(false);});await wait(400);
 const linked=await page.evaluate(()=>{const v=window.siliconDevine.viewer,s=v.getStats(),f=v.fabrics.get('_0');return{state:s.weight,shader:f.mesh.material.uniforms.selectedParameter.value,parameterIndices:Array.from(f.mesh.geometry.attributes.parameterIndex.array),positions:[...v.planes.values()].map(p=>p.center),cards:v.cards.filter(c=>c.priority).map(c=>c.group.position.toArray()),edges:[...v.planes.values()].map(p=>p.center[0]+p.width/2)};});
 assert.ok(linked.state);assert.equal(Math.floor(linked.shader),linked.state.index);assert.ok(linked.parameterIndices.includes(linked.state.index));assert.ok(linked.cards.every(p=>p[0]>Math.max(...linked.edges)));
 await page.screenshot({path:'.qa/v02-linear-desktop.png',fullPage:true});
 // Pick a weight cell by its actual 3D location, not by an artificial overlay.
 const pick=await page.evaluate(()=>{const v=window.siliconDevine.viewer,n=v.nodes.find(n=>n.id==='_0'),p=v.planes.get(n.parameters.weight),index=p.indices.length-4,point=v.camera.position.clone().set(...p.positions[index]).project(v.camera),r=document.querySelector('#stage').getBoundingClientRect();return{index:p.indices[index],x:r.left+(point.x+1)*r.width/2,y:r.top+(1-point.y)*r.height/2};});
 await page.mouse.move(pick.x,pick.y);await wait(200);const hovered=await page.evaluate(()=>window.siliconDevine.viewer.hoveredWeight);assert.equal(hovered?.index,pick.index);
 await page.mouse.move(10,10);
 await page.evaluate(()=>window.siliconDevine.viewer.focus('_1'));await wait(250);const sample=await page.evaluate(()=>window.siliconDevine.viewer.getStats().sample);assert.ok(sample&&Number.isFinite(sample.input)&&Number.isFinite(sample.output));
 await page.screenshot({path:'.qa/v02-activation-desktop.png',fullPage:true});
 const input=await page.evaluate(()=>window.siliconDevine.model.inputs[0]);await page.evaluate(async id=>{await window.siliconDevine.viewer.setTensorWindow(id,[0,32]);},input);
 const windowed=await page.evaluate(id=>{const v=window.siliconDevine.viewer,p=v.planes.get(id);return{origin:p.origin,indices:p.indices,samples:v.model.tensors.find(t=>t.id===id).samples};},input);assert.equal(windowed.origin[1],32);assert.ok(Object.keys(windowed.samples).length>0);
 const camera=await page.evaluate(()=>window.siliconDevine.viewer.camera.position.toArray());
 await writeFile(file,source(12));await until(async()=>{const s=await(await fetch('http://127.0.0.1:5183/api/status')).json();return s.revision===2;});await page.waitForFunction(()=>window.siliconDevine.model.tensors.some(t=>t.role==='activation'&&t.shape.at(-1)===12));
 assert.deepEqual(await page.evaluate(()=>window.siliconDevine.viewer.camera.position.toArray()),camera);
 await writeFile(file,'def broken(:');await until(async()=>(await(await fetch('http://127.0.0.1:5183/api/status')).json()).error);await page.waitForFunction(()=>document.querySelector('#live-status').textContent.includes('捕获失败'));assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getStats().totalNodes),2);
 await writeFile(file,source(6));await until(async()=>(await(await fetch('http://127.0.0.1:5183/api/status')).json()).revision===3);await page.waitForFunction(()=>window.siliconDevine.model.tensors.some(t=>t.role==='activation'&&t.shape.at(-1)===6));
 await page.click('#disconnect');await page.selectOption('#example','conv3d');await wait(600);await page.screenshot({path:'.qa/v02-conv3d-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});await page.selectOption('#example','declarative');await page.click('#fit');await wait(400);await page.screenshot({path:'.qa/v02-mobile.png',fullPage:true});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 assert.deepEqual(errors,[]);const result={liveReload:true,errorRecovery:true,cameraPreserved:true,weightLinked:true,weightHover:true,coordinatePaging:true,activationSample:sample,errors};await writeFile('.qa/v02-results.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();child.kill();}

