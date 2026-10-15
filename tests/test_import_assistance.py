"""Exercise import recovery through the same local API used by form controls."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from urllib.request import Request,urlopen
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from silicondevine.assistance import ImportProblem,exception_issue,dependency_issue,validate_settings,parameters
from silicondevine.live import LiveServer
from silicondevine.discovery import capture_project,_value


class AssistanceTests(unittest.TestCase):
    def test_config_objects_and_recovery_from_old_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);name='typed_'+uuid.uuid4().hex
            source=folder/(name+'.py')
            source.write_text('''from __future__ import annotations
from dataclasses import dataclass
from torch import nn
@dataclass
class Options:
 width:int=3
 enabled:bool=True
class Network(nn.Module):
 def __init__(self,config:Options):
  super().__init__();self.fc=nn.Linear(config.width,2)
 def forward(self,signal):return self.fc(signal)
''')
            config=folder/'silicondevine.json';config.write_text('{invalid')
            with self.assertRaises(ImportProblem) as error:capture_project([source],folder/'bad',config=config)
            self.assertEqual(error.exception.issue['code'],'config_invalid')
            settings={'ignore_project_config':True,'init':{'config':{'$class':name+':Options','kwargs':{'width':4}}}}
            graph=capture_project([source],folder/'object',backend='fx',overrides=settings)
            self.assertEqual(graph['discovery']['importOptions']['constructor'][0]['kind'],'object')
            config.write_text(json.dumps({'init':settings['init'],'inputs':[{'shape':[1,999]}]}))
            graph=capture_project([source],folder/'auto',backend='fx',overrides={'input_mode':'auto'})
            self.assertEqual(graph['discovery']['inputStatus'],'validated-example')
            source.write_text('class Broken(:\n pass\n')
            with self.assertRaises(ImportProblem) as error:capture_project([source],folder/'syntax')
            self.assertEqual(error.exception.issue['code'],'syntax_error')
            self.assertEqual(error.exception.issue['line'],1)

    def test_schema_types_and_literal_safety(self):
        from dataclasses import dataclass
        from typing import Literal
        @dataclass
        class Config:
            width:int=4
            enabled:bool=True
        def build(config:Config, mode:Literal['a','b']='a'):pass
        fields=parameters(build)
        self.assertEqual(fields[0]['kind'],'object')
        self.assertEqual(fields[0]['children'][0]['value'],4)
        self.assertEqual(fields[1]['choices'],['a','b'])
        def nested(batch:list[object],meta:'dict[str, object]',unknown):pass
        self.assertEqual([p['kind'] for p in parameters(nested,forward=True)],['sequence','mapping','any'])
        self.assertEqual(_value({'$namespace':{'width':4}}).width,4)
        for value in ({'script':'raise Error'}, {'inputs':[{'shape':[1,-2]}]}, {'init':{'bad':float('nan')}}):
            with self.assertRaises(ValueError):validate_settings(value)

    def test_clear_dependency_classification(self):
        e=ModuleNotFoundError("No module named 'einops'",name='einops')
        issue=exception_issue(e)
        self.assertEqual(issue['code'],'dependency_missing');self.assertEqual(issue['package'],'einops')
        unknown=dependency_issue('not_a_distribution_name')
        self.assertIsNone(unknown['package'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'local_module.py').write_text('');(p/'requirements.txt').write_text('einops')
            issue=dependency_issue('local_module',[p])
            self.assertEqual(issue['code'],'import_path');self.assertIsNone(issue['package'])
        self.assertEqual(exception_issue(TimeoutError())['code'],'capture_timeout')

    def test_readonly_server_and_control_origin(self):
        with tempfile.TemporaryDirectory() as d:
            for writable in (False,True):
                server=LiveServer(0,ROOT/'demo-dist',Path(d)/'profile.json' if writable else None)
                base=server.url.split('/?')[0]
                try:
                    def status(origin=None):
                        return json.load(urlopen(Request(base+'/api/status',headers={'Origin':origin} if origin else {})))
                    token=status()['controlToken']
                    self.assertEqual(bool(token),writable)
                    self.assertIsNone(status('http://localhost:9999')['controlToken'])
                    for origin,auth in [('https://untrusted.example',token),('http://localhost:9999',token),(None,'bad')]:
                        req=Request(base+'/api/import-settings',data=b'{}',headers={'Content-Type':'application/json','X-SiliconDevine-Token':auth or '',**({'Origin':origin} if origin else {})})
                        with self.assertRaises(HTTPError) as error:urlopen(req)
                        self.assertEqual(error.exception.code,403)
                    self.assertFalse((Path(d)/'profile.json').exists())
                finally:server.close()

    def test_live_recovery_without_editing_model(self):
        with tempfile.TemporaryDirectory(prefix='sd-assist-') as d:
            folder=Path(d);model=folder/'recovery_model.py'
            source='''import torch
from torch import nn
class First(nn.Module):
 def __init__(self,width:int):
  super().__init__();self.fc=nn.Linear(width,2)
 def forward(self,signal:torch.Tensor):return self.fc(signal).relu()
class Second(nn.Module):
 def __init__(self,width:int):
  super().__init__();self.fc=nn.Linear(width,3)
 def forward(self,signal):return self.fc(signal)
'''
            model.write_text(source,encoding='utf-8')
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            base=f'http://127.0.0.1:{port}'
            env=dict(os.environ,PYTHONPATH=str(ROOT/'python'),PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            with (folder/'log.txt').open('w',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-m','silicondevine.watch',str(model),'--port',str(port),'--settings',str(folder/'managed.json'),'--static-dir',str(ROOT/'demo-dist')],stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                try:
                    def get():return json.load(urlopen(base+'/api/status',timeout=2))
                    def wait(predicate):
                        deadline=time.monotonic()+45;value=None
                        while time.monotonic()<deadline:
                            try:
                                value=get()
                                if predicate(value):return value
                            except OSError:pass
                            if process.poll() is not None:break
                            time.sleep(.15)
                        self.fail(str(value)+'\n'+(folder/'log.txt').read_text(encoding='utf-8'))
                    def apply(value,route='import-settings'):
                        request=Request(base+'/api/'+route,data=json.dumps(value).encode(),headers={'Content-Type':'application/json','X-SiliconDevine-Token':get()['controlToken'],'Origin':base})
                        return json.load(urlopen(request))
                    state=wait(lambda s:(s.get('issue') or {}).get('code')=='entry_required')
                    options=state['issue']['schema']['candidates'];self.assertEqual(len(options),2)
                    entry=next(c['id'] for c in options if c['name']=='First')
                    apply({'entry':entry})
                    state=wait(lambda s:(s.get('issue') or {}).get('code')=='constructor_required')
                    self.assertEqual(state['issue']['schema']['constructor'][0]['kind'],'integer')
                    apply({'entry':entry,'init':{'width':3},'backend':'fx','kwargs':{'signal':{'shape':[1,3],'fill':'ones'}}})
                    state=wait(lambda s:s['ready'] and not s['loading']);self.assertEqual(state['revision'],1)
                    self.assertEqual(state['importOptions']['inputs'][0]['name'],'signal')
                    apply({'entry':entry,'init':{'width':3},'backend':'fx','kwargs':{'signal':{'shape':[1,8]}}})
                    state=wait(lambda s:(s.get('issue') or {}).get('code')=='inputs_invalid')
                    self.assertTrue(state['ready']);self.assertEqual(state['revision'],1)
                    apply({'entry':entry,'init':{'width':3},'backend':'fx','input_mode':'auto'})
                    state=wait(lambda s:s['revision']==2 and not s['loading']);self.assertIsNone(state['issue'])
                    self.assertEqual(model.read_text(encoding='utf-8'),source)
                    self.assertEqual(json.loads((folder/'managed.json').read_text())['init']['width'],3)
                    apply({},'retry');wait(lambda s:s['revision']==3 and not s['loading'])
                finally:
                    if process.poll() is None:
                        if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                        else:process.terminate()
                    process.wait(timeout=10)


if __name__=='__main__':unittest.main()
