import {build} from 'esbuild';
import {writeFile} from 'node:fs/promises';
await build({stdin:{contents:`import {KeynoteArtwork} from '../demo/keynoteStage';const art=new KeynoteArtwork(document.querySelector('canvas'));art.resize(1000,1000);art.render(0);window.art=art;`,resolveDir:process.cwd()+'/.qa',loader:'ts'},bundle:true,format:'esm',outfile:'.qa/art.js'});
await writeFile('.qa/art.html','<!doctype html><meta charset=utf-8><style>html,body{margin:0;background:transparent}canvas{width:1000px;height:1000px}</style><canvas></canvas><script type=module src=art.js></script>');
const {chromium}=await import('file:///C:/Users/30246/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
try{const page=await browser.newPage({viewport:{width:1000,height:1000},deviceScaleFactor:1});await page.goto('http://127.0.0.1:5198/.qa/art.html');await page.waitForFunction(()=>window.art);
 const data=await page.evaluate(()=>{window.art.render(0);return {image:window.art.renderer.domElement.toDataURL(),path:Array.from({length:240},(_,i)=>window.art.point(i*Math.PI*2/240))};});
 await writeFile('assets/keynote-source.png',Buffer.from(data.image.split(',')[1],'base64'));await writeFile('assets/keynote-path.json',JSON.stringify(data.path));console.log('Rendered shared optical sculpture and projected motion path.');
}finally{await browser.close();}
