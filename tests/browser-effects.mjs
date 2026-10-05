import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const {chromium}=await import('file:///C:/Users/30246/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1520,height:1080}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5198/demo-dist/');await page.waitForFunction(()=>window.siliconDevine?.model);await page.evaluate(()=>document.fonts.ready);await page.waitForTimeout(900);
 await page.screenshot({path:'.qa/effects-stage.png',fullPage:true});
 await page.click('#catalog-toggle');await page.waitForTimeout(1000);await page.screenshot({path:'.qa/effects-library.png',fullPage:true});
 const art=page.locator('.keynote-object canvas');assert.ok(await art.isVisible());
 const first=await art.screenshot();await page.waitForTimeout(700);assert.equal(first.equals(await art.screenshot()),false,'Art should move');
 await page.click('#library-return');await page.click('#inspector-toggle');await page.locator('#visual-effects').scrollIntoViewIfNeeded();await page.selectOption('#visual-effects','quiet');await page.screenshot({path:'.qa/effects-controls.png',fullPage:true});
 await page.click('#catalog-toggle');await page.waitForTimeout(250);const still=await art.screenshot();await page.waitForTimeout(250);assert.equal(still.equals(await art.screenshot()),true,'Quiet art must be static');
 await page.reload();await page.waitForFunction(()=>window.siliconDevine?.model);assert.equal(await page.locator('body').getAttribute('data-effects'),'quiet');
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(400);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/effects-mobile-stage.png',fullPage:true});
 await page.click('#catalog-toggle');await page.waitForTimeout(450);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/effects-mobile-library.png',fullPage:true});
 await page.emulateMedia({reducedMotion:'reduce'});await page.click('#library-return');await page.click('#inspector-toggle');await page.selectOption('#visual-effects','full');await page.click('#catalog-toggle');await page.waitForTimeout(250);const reduced=await art.screenshot();await page.waitForTimeout(200);assert.equal(reduced.equals(await art.screenshot()),true);
 assert.deepEqual(errors,[]);await writeFile('.qa/effects-browser.json',JSON.stringify({errors,animated:true,quiet:true,persistent:true,mobile:true,reducedMotion:true},null,2));console.log('Effects, static preference, persistence, reduced motion and mobile passed.');
}finally{await browser.close();}
