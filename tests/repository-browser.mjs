/** Optional end-to-end smoke test against a trusted local model repository. */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {parseArgs} from 'node:util';
import {chromium} from 'playwright';

const {values}=parseArgs({options:{source:{type:'string'},entry:{type:'string'}}});
if(!values.source)throw Error('Usage: node tests/repository-browser.mjs --source PATH [--entry module:Class]');
await mkdir('.qa',{recursive:true});
const profile=path.resolve('.qa/repository-browser-profile.json');
await writeFile(profile,JSON.stringify({input_mode:'auto',...(values.entry?{entry:values.entry}:{})}));
const child=spawn(process.env.PYTHON??'python',['-u','-m','silicondevine.watch',values.source,'--port','0','--settings',profile,'--static-dir',path.resolve('demo-dist')],{
  env:{...process.env,PYTHONPATH:[path.resolve('python'),process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8',OMP_NUM_THREADS:'2',MKL_NUM_THREADS:'2'},
  windowsHide:true,stdio:['ignore','pipe','pipe']
});
let log='',url,browser,startupError;
child.stdout.on('data',chunk=>{log+=chunk;url??=log.match(/http:\/\/127\.0\.0\.1:\d+\/\?live=1/)?.[0];});
child.stderr.on('data',chunk=>log+=chunk);
child.on('error',error=>{startupError=error;log+=error.message;});
try{
  const deadline=Date.now()+120000;
  let status;
  while(Date.now()<deadline){
    if(startupError)throw startupError;
    if(child.exitCode!==null)throw Error(log);
    if(url){status=await(await fetch(new URL('/api/status',url))).json();if(status.ready)break;if(status.error)throw Error(JSON.stringify(status.issue??status));}
    await new Promise(resolve=>setTimeout(resolve,200));
  }
  assert.ok(status?.ready,log);
  const snapshot=await(await fetch(new URL('/api/model',url))).json();
  browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL||undefined,headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto(url);
  await page.waitForFunction(name=>window.siliconDevine?.model.name===name,snapshot.model.name);
  await page.evaluate(()=>window.siliconDevine.viewer.setPlaying(false));
  const stats=await page.evaluate(()=>window.siliconDevine.viewer.getStats());
  assert.ok(stats.cells>0,JSON.stringify(stats));
  assert.equal(await page.locator('#import-notice').isVisible(),false);
  assert.deepEqual(errors,[]);
  await page.screenshot({path:'.qa/repository-browser.png'});
  const report={entry:snapshot.model.discovery?.entry,model:snapshot.model.name,stats,errors};
  await writeFile('.qa/repository-browser.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify(report));
}finally{
  await browser?.close();
  if(child.pid&&child.exitCode===null&&child.signalCode===null){const exited=once(child,'exit');child.kill();await exited;}
  await writeFile('.qa/repository-browser.log',log);
}
