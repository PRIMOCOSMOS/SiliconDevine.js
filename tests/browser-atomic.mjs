import {selectExample,selectModule} from './browser-shell.mjs';
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{
 const page=await browser.newPage({viewport:{width:1480,height:1000}}),errors=[],report={};page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL??'http://127.0.0.1:5195');
 for(const family of ['deepseek_v3','glm45','minimax_m1']){
  await selectExample(page,family);await page.waitForFunction(f=>window.siliconDevine?.model.architecture?.family===f,family);
  await selectModule(page,'moe_block');await page.waitForTimeout(180);
  const block=await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.setPlaying(false);return {source:v.isSourceExecution,nodes:v.nodes.map(n=>({id:n.id,op:n.op,inputs:n.inputs,outputs:n.outputs})),stats:v.getStats()};});
  assert.ok(block.source);assert.ok(block.nodes.some(n=>n.id==='module:decoder.self_attn'));assert.ok(!block.nodes.some(n=>n.op==='structure'));assert.equal(block.stats.atomic,undefined);
  await page.screenshot({path:`.qa/source-${family}-block.png`});
  await page.evaluate(()=>window.siliconDevine.viewer.focus('module:decoder.self_attn'));await page.waitForTimeout(200);
  const attention=await page.evaluate(()=>{const v=window.siliconDevine.viewer;return {scope:v.getNavigation().scope,nodes:v.nodes.map(n=>({op:n.op,outputs:n.outputs,role:n.attrs?.attentionRole})),stats:v.getStats()};});
  assert.equal(attention.scope,'decoder.self_attn');for(const op of ['linear','matmul','semantic'])assert.ok(attention.nodes.some(n=>n.op===op),`${family}: ${op}`);
  assert.ok(attention.nodes.some(n=>n.role==='probability'));
  const parentOut=block.nodes.find(n=>n.id==='module:decoder.self_attn').outputs;
  assert.ok(parentOut.every(id=>attention.nodes.some(n=>n.outputs.includes(id))),'Parent outputs must be the identical captured child tensors');
  await page.click('#fit');await page.waitForTimeout(750);
  await page.screenshot({path:`.qa/semantic-${family}-attention.png`});
  const cardCheck=await page.evaluate(()=>{const v=window.siliconDevine.viewer;const rects=v.cards.filter(c=>c.group.visible).map(c=>{const w=c.group.children[0],b=w.geometry.boundingBox;const ps=[[b.min.x,b.min.y],[b.max.x,b.max.y],[b.min.x,b.max.y],[b.max.x,b.min.y]].map(([x,y])=>w.localToWorld(v.camera.position.clone().set(x,y,0)).project(v.camera));return {x:Math.min(...ps.map(p=>p.x)),X:Math.max(...ps.map(p=>p.x)),y:Math.min(...ps.map(p=>p.y)),Y:Math.max(...ps.map(p=>p.y))};});return {count:rects.length,overlap:rects.some((a,i)=>rects.some((b,j)=>j>i&&a.x<b.X&&a.X>b.x&&a.y<b.Y&&a.Y>b.y))};});
  assert.ok(cardCheck.count<=4&&!cardCheck.overlap,'Compact automatic captions must not overlap');
  const coordinate=await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.selected=v.nodes.find(n=>n.op==='layout').id;v.setProgress(.4);return v.selected;});
  await page.waitForTimeout(100);assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.getStats().mechanism?.visibleItems>0),coordinate);
  await page.evaluate(()=>{const v=window.siliconDevine.viewer;v.focus(v.nodes.find(n=>n.op==='linear').id);v.setProgress(.4);});await page.waitForTimeout(150);
  assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.getStats().weight));
  await page.screenshot({path:`.qa/source-${family}-projection.png`});
  await page.click('#parent-module');await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>window.siliconDevine.viewer.getNavigation().scope),'decoder');
  const moe=family==='minimax_m1'?'decoder.block_sparse_moe':'decoder.mlp';await selectModule(page,moe);await page.waitForTimeout(100);
  assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.nodes.some(n=>n.op==='routing_index_add')));
  await page.screenshot({path:`.qa/source-${family}-moe.png`});report[family]={block:block.stats,attention:attention.stats};
 }
 await page.evaluate(()=>window.siliconDevine.viewer.showStructure());
 await selectModule(page,'lightning_block');await page.waitForTimeout(100);
 assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.isSourceExecution));
 await selectModule(page,'decoder.self_attn');await page.waitForTimeout(100);
 assert.ok(await page.evaluate(()=>window.siliconDevine.viewer.nodes.some(n=>n.op==='matmul')),'Lightning must expose captured matrix operations');
 await page.setViewportSize({width:390,height:844});await page.click('#fit');await page.waitForTimeout(100);
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:'.qa/source-mobile.png'});
 assert.deepEqual(errors,[]);await writeFile('.qa/source-browser.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}finally{await browser.close();}
