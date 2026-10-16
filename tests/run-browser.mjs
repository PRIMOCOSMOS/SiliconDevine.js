import {preview} from 'vite';
import {spawn} from 'node:child_process';
import {mkdir} from 'node:fs/promises';
await mkdir('.qa',{recursive:true});
const server=await preview({configFile:'vite.demo.config.ts',preview:{host:'127.0.0.1',port:0,open:false}});
try{
  const base=server.resolvedUrls.local[0];
  for(const file of ['browser-import-assistant.mjs','browser-v071.mjs']){
    const code=await new Promise((resolve,reject)=>{
      const child=spawn(process.execPath,[`tests/${file}`],{stdio:'inherit',env:{...process.env,DEMO_URL:base}});
      child.on('error',reject);child.on('exit',resolve);
    });
    if(code!==0)throw Error(`${file} failed (${code})`);
  }
}finally{await new Promise(resolve=>server.httpServer.close(resolve));}
