"""Briefly render native windows at 100% and 150% for bounded visual/layout QA."""
import ctypes
import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
import win32gui
import win32ui
from PIL import ImageGrab

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import app as launcher
launcher.enable_high_dpi()
out=root/'.qa'
out.mkdir(exist_ok=True)
report=[]

def capture(window,path):
    hwnd=win32gui.GetParent(window.winfo_id())
    left,top,right,bottom=win32gui.GetWindowRect(hwnd)
    # Capture only this app's window, including when another desktop covers it.
    from PIL import Image
    dc=win32gui.GetWindowDC(hwnd)
    source=win32ui.CreateDCFromHandle(dc)
    memory=source.CreateCompatibleDC()
    bitmap=win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(source,right-left,bottom-top)
    memory.SelectObject(bitmap)
    try:
        assert ctypes.windll.user32.PrintWindow(hwnd,memory.GetSafeHdc(),2), 'Window render capture failed'
        data=bitmap.GetInfo()
        im=Image.frombuffer('RGB',(data['bmWidth'],data['bmHeight']),bitmap.GetBitmapBits(True),'raw','BGRX',0,1)
        im.save(path)
    finally:
        win32gui.DeleteObject(bitmap.GetHandle())
        memory.DeleteDC()
        source.DeleteDC()
        win32gui.ReleaseDC(hwnd,dc)

with tempfile.TemporaryDirectory(prefix='sd-ui-') as folder:
 launcher.CONFIG=Path(folder)/'launcher.json'
 for scale in (1.,1.5):
    window=tk.Tk()
    callback_errors=[]
    window.report_callback_exception=lambda *exc: callback_errors.append(str(exc))
    window.withdraw()
    window.tk.call('tk','scaling',scale*96/72)
    app=launcher.Application(window)
    window.geometry('+4000+4000')
    # Keep diagnostic windows off-screen; PrintWindow captures only this HWND.
    window.deiconify()
    window.update()
    app.motion_button.invoke()
    assert app.keynote_stage.job is None
    app.motion_button.invoke()
    stage=app.keynote_stage
    item_count=len(stage.find_all())
    stage.pointer=(1.,-.5)
    stage.phase=1.
    start=time.perf_counter()
    for _ in range(100):stage.move_lights()
    frame_ms=(time.perf_counter()-start)*10
    assert len(stage.find_all())==item_count
    assert 0<stage.offset[0]<=4*app.ui_scale
    app.set_visual_state('loading')
    assert app.progress.active
    app.motion_button.invoke()
    assert stage.job is None and app.progress.job is None
    frozen=stage.phase
    window.update()
    assert stage.phase==frozen
    app.motion_button.invoke()
    window.withdraw();window.update()
    assert stage.job is None and app.progress.job is None
    window.deiconify();window.update()
    assert stage.job is not None and app.progress.job is not None
    stage.pointer=(0.,0.)
    for state in ('idle','ready','error'):
        app.set_visual_state(state)
        if state=='error':
            app.status.set('模型捕获失败。查看下方日志，修改代码后保存即可重试。')
            app.set_visual_state('error')
            app.log.configure(state='normal')
            app.log.insert('end','RuntimeError: 输入张量与模型不匹配。\n请检查模型输入后保存文件，服务会自动重新捕获。\n')
            app.log.configure(state='disabled')
            window.geometry(f'{round(700*scale)}x{round(540*scale)}')
        for _ in range(5):
            window.update()
            time.sleep(.08)
        for widget in (app.start_button,app.open_button,app.stop_button,app.log_toggle,app.state_title):
            assert widget.winfo_ismapped()
            x=widget.winfo_rootx()-window.winfo_rootx()
            y=widget.winfo_rooty()-window.winfo_rooty()
            assert 0<=x and x+widget.winfo_width()<=window.winfo_width(),(scale,state,widget,x)
            assert 0<=y and y+widget.winfo_height()<=window.winfo_height(),(scale,state,widget,y)
        capture(window,out/f'launcher-{scale}-{state}.png')
        report.append(dict(scale=scale,state=state,size=[window.winfo_width(),window.winfo_height()],effect_items=item_count,frame_ms=round(frame_ms,3)))
    stage.motion_allowed=False
    stage.set_motion(True)
    assert not stage.motion and stage.job is None
    app.close()
    assert not callback_errors,callback_errors
print(json.dumps(report))
