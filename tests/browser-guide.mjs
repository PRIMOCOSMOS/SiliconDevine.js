import {chromium} from 'playwright';
import assert from 'node:assert/strict';
const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||undefined,headless:true});
try{
 const page=await browser.newPage({viewport:{width:1280,height:900}});
 await page.goto(new URL('../docs/QUICKSTART.html',import.meta.url).href);
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
