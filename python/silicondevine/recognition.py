"""Conservative graph recognition. Metadata enriches, never replaces, execution."""
import hashlib,inspect
from pathlib import Path


def annotate_model(graph,model):
    nodes=graph['nodes'];tensors={t['id']:t for t in graph['tensors']};producer={o:n for n in nodes for o in n['outputs']}
    class_sources={};file_hashes={}
    for entry in graph.get('modules',[]):
        part=model.get_submodule(entry['path']);cls=type(part)
        entry['qualified']=cls.__module__+'.'+cls.__name__
        try:
            if cls not in class_sources:
                file=Path(inspect.getsourcefile(cls))
                if file not in file_hashes:file_hashes[file]=hashlib.sha256(file.read_bytes()).hexdigest()
                class_sources[cls]={'file':str(file),'line':inspect.getsourcelines(cls)[1],'sha256':file_hashes[file]}
            entry['code']=class_sources[cls]
        except (OSError,TypeError):pass
    if hasattr(model,'_silicondevine_source'):graph['provenance']=model._silicondevine_source
    units=[]
    def ancestors(t,limit=150):
        result=set();queue=[t]
        while queue and len(result)<limit:
            key=queue.pop()
            if key in result:continue
            result.add(key);n=producer.get(key)
            if n:queue.extend(n['inputs'])
        return result
    def find_score(t,depth=0):
        n=producer.get(t)
        if not n or depth>15:return None
        if n['op']=='matmul':return n
        if n['op'] not in ('identity','cast','multiply','divide','add','attention_mask','slice'):return None
        found={}
        for i in n['inputs']:
            p=find_score(i,depth+1)
            if p:found[p['id']]=p
        return next(iter(found.values())) if len(found)==1 else None
    def projection(t):
        queue=[t];visited=set();found=[]
        while queue and len(visited)<100:
            key=queue.pop()
            if key in visited:continue
            visited.add(key);p=producer.get(key)
            if not p:continue
            if p['op']=='linear':found.append(p);continue
            if p['op'] in ('reshape','identity','cast','permute','transpose','slice','select','split','chunk','concat','expand','multiply','add','negative'):queue.extend(p['inputs'])
        unique={p['id']:p for p in found}
        return list(unique.values())
    for n in nodes:
        if n['op']=='softmax':
            score=find_score(n['inputs'][0])
            if not score:continue
            weighted=next((p for p in nodes if p['op']=='matmul' and n['outputs'][0] in ancestors(p['inputs'][0],15)),None)
            if not weighted:continue
            q,k=score['inputs'];v=weighted['inputs'][1];unit={'kind':'attention','id':n['id'],'path':n['group'],'query':q,'keyTransposed':k,'value':v,'score':score['outputs'][0],'probability':n['outputs'][0],'context':weighted['outputs'][0],'qk':score['id'],'weighted':weighted['id']}
            unit['evidence']='matmul → scale/mask → softmax → matmul with values'
            units.append(unit)
            for node,role in [(score,'score'),(n,'probability'),(weighted,'context')]:node['attrs'].update({'attention':unit,'attentionRole':role})
            for tensor,role in [(q,'Q'),(k,'Kᵀ'),(v,'V'),(n['outputs'][0],'A'),(weighted['outputs'][0],'Context')]:tensors[tensor]['semantic']=role
            for tensor,role in [(q,'Q'),(k,'K'),(v,'V')]:
                candidates=projection(tensor)
                if len(candidates)==1:
                    candidates[0]['attrs'].setdefault('projectionRoles',[]).append(role)
                    candidates[0]['name']= '/'.join(dict.fromkeys(candidates[0]['attrs']['projectionRoles']))+' 投影 · '+candidates[0]['group']
            shape=tensors[n['outputs'][0]]['shape'];tensors[n['outputs'][0]]['axes']=['B','H','Query','Key'] if len(shape)==4 else ['Batch×Head','Query','Key'] if len(shape)==3 else ['Query','Key']
        if n['op']=='add' and len(n['inputs'])==2:
            a,b=n['inputs'];shape=tensors[n['outputs'][0]]['shape']
            if tensors[a]['shape']==tensors[b]['shape']==shape and tensors[a]['role'] in ('input','activation') and tensors[b]['role'] in ('input','activation'):
                skip=a if a in ancestors(b) else b if b in ancestors(a) else None
                if skip:
                    n['attrs']['residualInput']=skip;n['name']='残差汇合 · '+n['group'];units.append({'kind':'residual','id':n['id'],'input':skip,'output':n['outputs'][0],'evidence':'same shape and one operand is an ancestor of the other'})
    def scalar(t):
        item=tensors.get(t,{})
        values=item.get('data',{}).get('values',[])
        return values[0] if item.get('role')=='constant' and len(values)==1 else None
    # Only recognize the actual diagonal-Gaussian arithmetic graph, not a class name.
    for n in nodes:
        if n['op']!='add' or len(n['inputs'])!=2 or n['attrs'].get('alpha',1)!=1:continue
        for mean,product in (n['inputs'],n['inputs'][::-1]):
            p=producer.get(product)
            if not p or p['op']!='multiply' or len(p['inputs'])!=2:continue
            for noise,std in (p['inputs'],p['inputs'][::-1]):
                noise_node=producer.get(noise);std_node=producer.get(std)
                if not noise_node or noise_node['op']!='standard_normal' or not std_node or std_node['op']!='exp':continue
                half=producer.get(std_node['inputs'][0])
                if not half or half['op']!='multiply' or len(half['inputs'])!=2:continue
                logvar=next((half['inputs'][1-i] for i,t in enumerate(half['inputs']) if scalar(t)==.5 or half['attrs'].get('scalarOperands',{}).get(str(i))==.5),None)
                if not logvar or any(tensors[t]['shape']!=tensors[mean]['shape'] for t in (logvar,std,noise,n['outputs'][0])):continue
                unit={'kind':'gaussian_reparameterization','id':n['id'],'mean':mean,'logvar':logvar,'std':std,'noise':noise,'sample':n['outputs'][0],
                      'children':[half['id'],std_node['id'],noise_node['id'],p['id'],n['id']],
                      'evidence':'z = mu + exp(0.5 * logvar) * captured randn; equal tensor shapes'}
                units.append(unit)
                n['attrs']['gaussian']=unit;n['name']='高斯重参数采样'
                for t,label in ((mean,'均值 μ'),(logvar,'对数方差 log σ²'),(std,'标准差 σ'),(noise,'标准高斯 ε'),(n['outputs'][0],'潜变量 z')):
                    tensors[t]['semantic']=label
                    tensors[t]['axes']=['Batch']+[f'D{i}' for i in range(1,len(tensors[t]['shape'])-1)]+['Latent'] if len(tensors[t]['shape'])>1 else ['Latent']
                for t,label in ((mean,'均值头 μ'),(logvar,'对数方差头 log σ²')):
                    head=producer.get(t)
                    if head and head['op'] in ('linear','conv1d','conv2d','conv3d'):
                        head['attrs']['gaussianHead']=label;head['name']=label
    graph['functionalUnits']=units
