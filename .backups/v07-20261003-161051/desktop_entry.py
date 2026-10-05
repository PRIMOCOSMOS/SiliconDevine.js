"""Windowed executable entry point and opt-in packaged-runtime diagnostic."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
import traceback
from urllib.request import urlopen
import app as launcher


def diagnose(destination, python=None, preset='MLP'):
    result = {'frozen':bool(getattr(sys,'frozen',False)), 'root':str(launcher.ROOT), 'ok':False}
    app = None
    window = None
    try:
        with tempfile.TemporaryDirectory(prefix='silicondevine-diagnostic-') as temp:
            launcher.CONFIG=Path(temp)/'settings.json'
            opened=[]
            launcher.webbrowser.open=opened.append
            launcher.messagebox.showerror=lambda title,message: (_ for _ in ()).throw(RuntimeError(title+': '+message))
            window=tk.Tk()
            window.withdraw()
            app=launcher.Application(window)
            result['detectedPython']=app.python.get()
            if python:app.python.set(python)
            app.preset.set(preset)
            app.preset.event_generate('<<ComboboxSelected>>')
            window.update()
            app.start()
            process=app.process
            assert process is not None,'No model process'
            deadline=time.monotonic()+150
            while not app.auto_opened and time.monotonic()<deadline:
                window.update()
                time.sleep(.08)
            assert app.auto_opened,app.status.get()+'\n'+app.log.get('1.0','end')
            with urlopen(app.url.split('?')[0]+'api/model',timeout=10) as response:
                data=json.load(response)
            with urlopen(app.url,timeout=10) as response:
                assert b'<html' in response.read().lower()
            result.update(model=data['model']['name'],nodes=len(data['model']['nodes']),browserHandoff=opened==[app.url],settingsSaved=launcher.CONFIG.exists(),provenance=data['model'].get('provenance'))
            assert result['browserHandoff'] and result['settingsSaved']
            app.stop()
            assert process.poll() is not None and app.process is None
            result['processStopped']=True
            result['ok']=True
    except Exception:
        result['error']=traceback.format_exc()
    finally:
        if app:app.close()
        elif window:window.destroy()
        destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if result['ok'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='SiliconDevine desktop launcher')
    parser.add_argument('--self-test',type=Path,help='Write a local startup diagnostic report and exit')
    parser.add_argument('--python',help='Python interpreter for the diagnostic')
    parser.add_argument('--preset',default='MLP',choices=['MLP','TinyGPT','LLaMA · Transformers','SAB · 作者实现','ISAB · 作者实现','VAE · PyTorch 官方','卷积 VAE','条件 VAE','DeepSeek-V3','GLM-4.5','MiniMax-M1'])
    args=parser.parse_args()
    launcher.enable_high_dpi()
    if args.self_test:
        raise SystemExit(diagnose(args.self_test,args.python,args.preset))
    window=tk.Tk()
    application=launcher.Application(window)
    window.mainloop()
