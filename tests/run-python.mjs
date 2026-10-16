import {spawnSync} from 'node:child_process';
import {mkdirSync} from 'node:fs';
mkdirSync('.qa',{recursive:true});
const python=process.env.PYTHON??'python';
const args=process.argv.slice(2);
const result=spawnSync(python,args.length?args:['-m','unittest','discover','-s','tests','-p','test_*.py'],{
  stdio:'inherit',env:{...process.env,OMP_NUM_THREADS:process.env.OMP_NUM_THREADS??'2',MKL_NUM_THREADS:process.env.MKL_NUM_THREADS??'2',PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8'}
});
if(result.error){console.error(`Cannot start ${python}: ${result.error.message}`);process.exit(1);}
process.exit(result.status??1);
