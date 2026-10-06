import {selectModule} from './browser-shell.mjs';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5195');await page.waitForFunction(()=>window.siliconDevine?.viewer.nodes.length>0);
 await page.locator('#file').setInputFiles('.qa/layout-user.json');await page.waitForFunction(()=>window.siliconDevine.model.name==='HybridVisionNetwork');
 await selectModule(page,'res_block1');await page.waitForTimeout(850);
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setPlaying(false);v.setLabels('all');});await page.waitForTimeout(150);
 const layout=await page.evaluate(()=>{const v=window.siliconDevine.viewer;return {planes:[...v.planes.values()].map(p=>({id:p.tensor.id,owner:p.owner,c:p.center,w:p.width,d:p.depth,h:p.height})),cards:v.cards.map(c=>{const p=v.planes.get(c.id),w=c.group.children[0].geometry.parameters.width;return {id:c.id,gap:Math.abs(c.group.position.x-p.center[0])-p.width/2-w/2};})};});
 for(const c of layout.cards)assert.ok(Math.abs(c.gap-.18)<1e-6,'Trimmed lettering must stay next to the tensor');
 await page.screenshot({path:'.qa/user-residual-all.png'});
 await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setLabels('auto');v.focus(v.nodes.find(n=>n.op==='batchnorm').id);});await page.waitForTimeout(850);await page.screenshot({path:'.qa/user-norm-focused.png'});
 await page.click('#fit');await page.waitForTimeout(850);
 const canvas=page.locator('#stage canvas'),rect=await canvas.boundingBox(),cx=rect.x+rect.width*.5,cy=rect.y+rect.height*.5;
 const state=()=>page.evaluate(()=>{const v=window.siliconDevine.viewer;return {p:v.camera.position.toArray(),t:v.controls.target.toArray(),selected:v.selected??null,scope:v.getNavigation().scope,flight:!!v.cameraFlight};});
 const dist=(a,b)=>Math.hypot(...a.map((x,i)=>x-b[i]));
 async function drag(button,dx,dy){await page.mouse.move(cx,cy);await page.mouse.down({button});await page.mouse.move(cx+dx,cy+dy,{steps:12});await page.mouse.up({button});await page.waitForTimeout(900);}
 let a=await state();await drag('left',100,25);let b=await state();assert.ok(dist(a.p,b.p)>1);assert.ok(dist(a.t,b.t)<1e-5);assert.equal(b.selected,null);
 a=b;await drag('right',75,25);b=await state();assert.ok(dist(a.t,b.t)>.1);assert.equal(b.selected,null);assert.ok(Math.abs(dist(a.p,a.t)-dist(b.p,b.t))<1e-4);
 await page.click('#pan-mode');assert.equal(await page.locator('#pan-mode').getAttribute('aria-pressed'),'true');a=await state();await drag('left',-70,-25);b=await state();assert.ok(dist(a.t,b.t)>.1);assert.equal(b.selected,null);await page.click('#pan-mode');
 await page.locator('#stage').focus();a=await state();await page.keyboard.press('ArrowRight');await page.waitForTimeout(150);b=await state();assert.ok(dist(a.t,b.t)>.1);
 a=b;await page.keyboard.press('Shift+ArrowLeft');await page.waitForTimeout(150);b=await state();assert.ok(dist(a.p,b.p)>.1);assert.ok(dist(a.t,b.t)<1e-5);
 await page.mouse.move(cx,cy);a=await state();await page.mouse.wheel(0,-180);await page.waitForTimeout(800);b=await state();assert.ok(dist(b.p,b.t)<dist(a.p,a.t));assert.ok(dist(a.t,b.t)<1e-5,'Wheel must not cause target drift');
 await page.selectOption('#camera-view','front');await page.waitForTimeout(850);a=await state();assert.ok(Math.abs(a.p[0]-a.t[0])<1e-5&&Math.abs(a.p[1]-a.t[1])<1e-5);
 await page.selectOption('#camera-view','oblique');await page.waitForTimeout(850);
 // Interrupt a smooth camera flight with an actual drag; it must not pull back.
 await page.click('#fit');await drag('right',50,15);assert.equal((await state()).flight,false);
 await selectModule(page,'attention');await page.waitForTimeout(850);await page.screenshot({path:'.qa/user-attention-layout.png'});
 await page.setViewportSize({width:390,height:844});await page.click('#fit');await page.waitForTimeout(850);
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.ok(await page.locator('#pan-mode').isVisible());await page.screenshot({path:'.qa/user-navigation-mobile.png'});
 assert.deepEqual(errors,[]);await writeFile('.qa/spatial-navigation.json',JSON.stringify({layout,mouseRotate:true,rightPan:true,leftPan:true,keyboard:true,stableZoom:true,presets:true,interruptFlight:true},null,2));
 console.log('User model, close captions, mouse/keyboard navigation, stable zoom and mobile layout passed.');
}finally{await browser.close();}
