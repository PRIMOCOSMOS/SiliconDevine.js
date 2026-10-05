"""Fetch small public source/config files only, never model weights or remote execution."""
from pathlib import Path
import urllib.request,json,hashlib
root=Path(__file__).resolve().parents[1]/'examples/upstream/large_models'
root.mkdir(parents=True,exist_ok=True)
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'SiliconDevine-source-verification'})
    with urllib.request.urlopen(req,timeout=45) as r:return r.read()
manifest={}
for key,repo,paths in [
 ('deepseek_v3','deepseek-ai/DeepSeek-V3',['inference/model.py','inference/configs/config_671B.json','LICENSE-CODE']),
 ('minimax_m1','MiniMax-AI/MiniMax-M1',['modeling_minimax_m1.py','configuration_minimax_m1.py','LICENSE'])]:
    sha=json.loads(get('https://api.github.com/repos/'+repo+'/commits/main'))['sha']
    entries=[]
    for path in paths:
        url=f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'
        data=get(url);dest=root/key/Path(path).name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
        entries.append({'path':path,'url':url,'sha256':hashlib.sha256(data).hexdigest()})
    manifest[key]={'repository':'https://github.com/'+repo,'commit':sha,'files':entries}
    print(key,sha,flush=True)
for key,repo in [('glm45','zai-org/GLM-4.5'),('minimax_m1','MiniMaxAI/MiniMax-M1-80k')]:
    info=json.loads(get('https://huggingface.co/api/models/'+repo));sha=info['sha']
    url=f'https://huggingface.co/{repo}/resolve/{sha}/config.json';data=get(url)
    folder=root/key;folder.mkdir(exist_ok=True);(folder/'config.json').write_bytes(data)
    manifest.setdefault(key,{})['config']={'repository':'https://huggingface.co/'+repo,'commit':sha,'url':url,'sha256':hashlib.sha256(data).hexdigest()}
    print(key,'config',sha,flush=True)
    (root/'SOURCE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
import transformers,inspect,shutil
from transformers.models.glm4_moe import modeling_glm4_moe
file=Path(inspect.getfile(modeling_glm4_moe));shutil.copy2(file,root/'glm45/modeling_glm4_moe.py')
manifest['glm45']['implementation']={'repository':'https://github.com/huggingface/transformers','version':transformers.__version__,'path':'src/transformers/models/glm4_moe/modeling_glm4_moe.py','sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
(root/'SOURCE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Sources pinned; no weights downloaded.')
