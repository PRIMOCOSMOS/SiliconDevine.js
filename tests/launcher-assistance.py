"""Native recovery controls: no hidden JSON editing or mandatory log reading."""
import json
from pathlib import Path
import sys
import tempfile
import tkinter as tk
import ctypes
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app as launcher
from desktop_runtime import startup_issue,install_command

def capture(window,path):
    import win32gui,win32ui
    from PIL import Image
    hwnd=win32gui.GetParent(window.winfo_id());left,top,right,bottom=win32gui.GetWindowRect(hwnd)
    dc=win32gui.GetWindowDC(hwnd);source=win32ui.CreateDCFromHandle(dc);memory=source.CreateCompatibleDC();bitmap=win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(source,right-left,bottom-top);memory.SelectObject(bitmap)
    try:
        assert ctypes.windll.user32.PrintWindow(hwnd,memory.GetSafeHdc(),2)
        data=bitmap.GetInfo();Image.frombuffer('RGB',(data['bmWidth'],data['bmHeight']),bitmap.GetBitmapBits(True),'raw','BGRX',0,1).save(path)
    finally:
        win32gui.DeleteObject(bitmap.GetHandle());memory.DeleteDC();source.DeleteDC();win32gui.ReleaseDC(hwnd,dc)

launcher.enable_high_dpi()
report=[]
with tempfile.TemporaryDirectory() as directory:
 launcher.CONFIG=Path(directory)/'launcher.json'
 for scale in (1.,1.5):
    window=tk.Tk();window.withdraw();window.tk.call('tk','scaling',scale*96/72)
    application=launcher.Application(window);errors=[];window.report_callback_exception=lambda *e:errors.append(str(e))
    try:
        window.geometry(f'{round(850*scale)}x{round(760*scale)}+4000+4000');window.deiconify();window.update()
        issue=startup_issue("ModuleNotFoundError: No module named 'torch'",application.python.get())
        application.show_issue(issue);window.update()
        assert application.state_title.cget('text')=='缺少依赖 · torch'
        assert 'pip install' in application.issue_command.get()
        assert application.install_row.winfo_ismapped()
        assert not application.log_panel.winfo_ismapped()
        assert not application.assist_button.winfo_ismapped()
        capture(window,Path(__file__).resolve().parents[1]/'.qa'/f'import-native-{scale}.png')
        application.url='http://127.0.0.1:5182/?live=1'
        application.show_issue({'title':'请选择要展示的模型','message':'选择入口后继续填写构造参数。','schema':{'candidates':[]}});window.update()
        # An empty schema is not actionable; real schemas always include candidates/settings.
        application.show_issue({'title':'请补充模型构造参数','message':'需要填写 width。','schema':{'settings':{},'constructor':[{'name':'width'}]}});window.update()
        assert application.assist_button.winfo_ismapped()
        assert not application.install_row.winfo_ismapped()
        application.clear_issue();window.update();assert not application.issue_panel.winfo_ismapped()
        assert not errors,errors
        report.append({'scale':scale,'missingDependency':True,'configurationRecovery':True})
    finally:application.close()
assert startup_issue('ImportError: DLL load failed while importing torch','python')['code']=='environment_broken'
assert install_command({'module':'unknown'},'python')==''
assert "a''b" in install_command({'package':'torch'},"D:/a'b/python.exe")

# A real capture failure reaches the same inline recovery controls.
with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);launcher.CONFIG=folder/'launcher.json'
    model=folder/'missing_dependency.py'
    model.write_text('from torch import nn\nimport silicondevine_absent_test_package\nclass Model(nn.Module):\n def forward(self,x):return x\n')
    (folder/'requirements.txt').write_text('torch')
    window=tk.Tk();window.withdraw();application=launcher.Application(window)
    try:
        application.file.set(str(model));application.factory.set('auto');application.start()
        deadline=time.monotonic()+35
        while not application.issue and time.monotonic()<deadline:window.update();time.sleep(.05)
        assert application.issue and application.issue['code']=='dependency_missing',application.status.get()
        assert application.issue['module']=='silicondevine_absent_test_package'
        assert '-r' in application.issue_command.get()
        application.retry()
        deadline=time.monotonic()+5
        while application.visual_state!='loading' and time.monotonic()<deadline:window.update();time.sleep(.03)
        assert application.visual_state=='loading','Native retry did not start a new capture'
    finally:application.close()
print(json.dumps(report))
