import {chromium} from 'file:///C:/Users/30246/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
const browser=await chromium.launch({channel:'msedge',headless:true});
try{
 const page=await browser.newPage({viewport:{width:1280,height:900}});
 await page.goto('file:///D:/Personpage/tmp/silicondevine-v07/docs/QUICKSTART.html');
 await page.getByRole('link',{name:'代码接入',exact:true}).click();
 assert.ok(page.url().endsWith('#2'));
 await page.screenshot({path:'.qa/guide-desktop.png'});
 await page.setViewportSize({width:390,height:844});
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.screenshot({path:'.qa/guide-mobile.png'});
 await page.getByRole('link',{name:'参考目录',exact:true}).click();
 await page.getByRole('link',{name:'源码组合',exact:true}).click();
 assert.ok(page.url().endsWith('ATOMIC-COMPOSITION.html'));
 console.log('Browser documentation: chapter anchors, mobile layout and reference navigation passed.');
}finally{await browser.close();}
