import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:900,height:900},hasTouch:true});
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5195');await page.waitForFunction(()=>window.siliconDevine?.viewer.nodes.length>0);await page.waitForTimeout(900);
 const cdp=await page.context().newCDPSession(page),r=await page.locator('#stage canvas').boundingBox(),x=r.x+r.width/2,y=r.y+r.height/2;
 const state=()=>page.evaluate(()=>{const v=window.siliconDevine.viewer;return {p:v.camera.position.toArray(),t:v.controls.target.toArray(),selected:v.selected??null};});
 const dist=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
 const touch=(type,points)=>cdp.send('Input.dispatchTouchEvent',{type,touchPoints:points.map(([id,x,y])=>({id,x,y}))});
 let a=await state();await touch('touchStart',[[1,x,y]]);for(let i=1;i<=8;i++)await touch('touchMove',[[1,x+i*8,y+i*2]]);await touch('touchEnd',[]);await page.waitForTimeout(800);let b=await state();assert.ok(dist(a.p,b.p)>.1);assert.equal(b.selected,null);
 await page.click('#pan-mode');a=await state();await touch('touchStart',[[1,x,y]]);for(let i=1;i<=8;i++)await touch('touchMove',[[1,x+i*5,y]]);await touch('touchEnd',[]);await page.waitForTimeout(800);b=await state();assert.ok(dist(a.t,b.t)>.1);assert.equal(b.selected,null);
 a=b;await touch('touchStart',[[1,x-35,y],[2,x+35,y]]);for(let i=1;i<=8;i++)await touch('touchMove',[[1,x-35-i*3,y+i],[2,x+35+i*3,y+i]]);await touch('touchEnd',[]);await page.waitForTimeout(800);b=await state();assert.ok(dist(b.p,b.t)<dist(a.p,a.t));assert.equal(b.selected,null);
 console.log('Single-finger rotate/pan and two-finger pinch/pan passed without accidental focus.');
}finally{await browser.close();}
