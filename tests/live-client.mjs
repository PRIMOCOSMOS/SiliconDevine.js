import assert from 'node:assert/strict';
import {build} from 'esbuild';
await build({entryPoints:['src/core/liveClient.ts'],outfile:'.qa/live-client.mjs',bundle:true,format:'esm',platform:'node'});
const {LiveConnection}=await import('../.qa/live-client.mjs');
globalThis.location={href:'http://127.0.0.1:5182/'};
const originalFetch=globalThis.fetch;
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(predicate){for(let i=0;i<150;i++){if(predicate())return;await pause(5);}assert.fail('Timed out waiting for client state');}
const reply=value=>new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json'}});
const status={revision:1,ready:true,loading:false,error:null,controlToken:'test-token'};
try{
  for(const url of ['https://example.com','ftp://localhost/','http://user:secret@localhost/'])assert.throws(()=>new LiveConnection(url,{onModel(){}}));
  let calls=0,errors=[],models=[],aborted=0;
  globalThis.fetch=async(url,options)=>{
    if(url.endsWith('/api/status')){
      if(++calls===1)return new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>{aborted++;reject(options.signal.reason);},{once:true}));
      return reply(status);
    }
    return reply({revision:1,model:{name:'recovered'}});
  };
  const client=new LiveConnection('http://127.0.0.1:5182',{onModel:m=>models.push(m),onError:e=>errors.push(e.message)},{requestTimeoutMs:30,pollIntervalMs:5}).start();
  await until(()=>models.length===1);
  assert.equal(aborted,1);assert.match(errors[0],/超时/);assert.equal(models[0].name,'recovered');
  globalThis.fetch=async()=>new Response('<html>Bad gateway</html>',{status:502});
  await assert.rejects(client.configure({}),/502/);
  client.stop();await assert.rejects(client.configure({}),/启动器/);
  let resolveModel;
  globalThis.fetch=async url=>url.endsWith('/api/status')?reply(status):new Promise(resolve=>resolveModel=resolve);
  models=[];
  const stopped=new LiveConnection('http://localhost',{onModel:m=>models.push(m)},{requestTimeoutMs:100}).start();
  await until(()=>resolveModel);
  stopped.stop();resolveModel(reply({revision:1,model:{name:'stale'}}));await pause(10);
  assert.deepEqual(models,[]);
  globalThis.fetch=async url=>url.includes('/api/tensor')?reply({revision:2,values:[1]}):url.endsWith('/api/status')?reply(status):reply({revision:1,model:{name:'valid'}});
  const windows=new LiveConnection('http://localhost',{onModel(){}},{pollIntervalMs:100}).start();
  await until(()=>windows.currentRevision===1);
  await assert.rejects(windows.tensorWindow('x',[0]),/模型已更新/);windows.stop();
  console.log('Live client: timeout recovery, invalid responses, stopped requests and stale tensor windows passed.');
}finally{globalThis.fetch=originalFetch;}
