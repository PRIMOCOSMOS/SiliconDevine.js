"""Local desktop launcher. The browser is the 3D workspace; this window owns its server."""
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from urllib.request import urlopen
import webbrowser
from launcher_ui import build_ui, enable_high_dpi
from desktop_runtime import application_root, default_python, external_environment, external_libraries

ROOT = application_root()
CONFIG = Path(os.getenv('LOCALAPPDATA', Path.home())) / 'SiliconDevine' / 'launcher.json'


class Application:
    def __init__(self, window):
        self.window = window
        self.process = None
        self.url = None
        self.events = queue.Queue()
        self.generation = 0
        self.auto_opened = False
        self.closed = False
        try:
            self.saved = json.loads(CONFIG.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            self.saved = {}
        self.file = tk.StringVar(value=self.saved.get('file', str(ROOT / 'examples/live_model.py')))
        self.backend = tk.StringVar(value=self.saved.get('backend', 'fx'))
        self.factory = tk.StringVar(value=self.saved.get('factory', 'build_model'))
        self.status = tk.StringVar(value='选择模型文件，启动后在 IDE 中编辑并保存。')
        self.python = tk.StringVar(value=self.saved.get('python') or default_python())
        self.root_path = ROOT
        build_ui(self)
        window.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_job = window.after(150, self.poll)

    def choose(self):
        selected = filedialog.askopenfilename(title='选择 PyTorch 模型', filetypes=[('Python 模型', '*.py')])
        if selected:
            self.file.set(selected)

    def choose_python(self):
        selected = filedialog.askopenfilename(title='选择包含 PyTorch 的 Python 环境', filetypes=[('Python', '*.exe')])
        if selected:
            self.python.set(selected)

    def start(self):
        if self.process and self.process.poll() is None:
            return
        file = Path(self.file.get()).expanduser()
        if not file.is_file() or file.suffix.lower() != '.py':
            return messagebox.showerror('无法启动', '请选择存在的 Python 模型文件。')
        if not self.factory.get().isidentifier():
            return messagebox.showerror('无法启动', '模型工厂函数名必须是有效的 Python 标识符。')
        interpreter = Path(self.python.get()).expanduser()
        if not interpreter.is_file() or (getattr(sys, 'frozen', False) and interpreter.resolve() == Path(sys.executable).resolve()):
            return messagebox.showerror('请选择 Python 环境', '请选择已安装 PyTorch 的 python.exe，不能选择启动器自身。')
        if not (ROOT / 'demo-dist/index.html').exists():
            return messagebox.showerror('缺少工作台', '请先在项目目录运行 npm run build。')
        # Use an available local port so another workspace is never disturbed.
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = external_environment()
        env['PYTHONPATH'] = str(ROOT / 'python') + os.pathsep + env.get('PYTHONPATH', '')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        try:
            with external_libraries():
                self.process = subprocess.Popen([str(interpreter.resolve()), '-u', '-m', 'silicondevine.watch', str(file.resolve()), '--factory', self.factory.get(), '--backend', self.backend.get(), '--port', str(port), '--static-dir', str(ROOT / 'demo-dist')], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as exc:
            return messagebox.showerror('无法启动 Python', str(exc))
        self.url = f'http://127.0.0.1:{port}/?live=1'
        self.generation += 1
        generation, process = self.generation, self.process
        self.auto_opened = False
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.started_at=time.monotonic()
        self.status.set('首次加载大型依赖可能较慢，完成后会自动打开工作台。')
        self.set_visual_state('loading')
        self.lock_settings(True)
        try:
            CONFIG.parent.mkdir(parents=True, exist_ok=True)
            CONFIG.write_text(json.dumps({key: value.get() for key, value in [('file', self.file), ('backend', self.backend), ('factory', self.factory), ('python', self.python)]}), encoding='utf-8')
        except OSError:
            pass
        def logs():
            for line in process.stdout:
                self.events.put((generation, 'log', line.rstrip()))
        def status():
            while process.poll() is None and generation == self.generation:
                try:
                    with urlopen(f'http://127.0.0.1:{port}/api/status', timeout=1) as response:
                        self.events.put((generation, 'status', json.load(response)))
                except (OSError, ValueError):
                    pass
                time.sleep(.8)
        threading.Thread(target=logs, daemon=True).start()
        threading.Thread(target=status, daemon=True).start()

    def open_browser(self):
        if self.url:
            webbrowser.open(self.url)

    def poll(self):
        while not self.events.empty():
            generation, kind, value = self.events.get()
            if generation != self.generation:
                continue
            if kind == 'log':
                self.log.configure(state='normal')
                self.log.insert('end', value + '\n')
                if int(self.log.index('end-1c').split('.')[0]) > 160:
                    self.log.delete('1.0', '40.0')
                self.log.see('end')
                self.log.configure(state='disabled')
            elif value.get('error'):
                self.status.set('请查看运行日志并修改模型，保存后会自动重试。')
                self.set_visual_state('error')
            elif value.get('loading') and not value.get('ready'):
                self.set_visual_state('loading')
                self.status.set(f'正在首次捕获 · 已等待 {int(time.monotonic()-self.started_at)} 秒 · 完成后自动打开工作台')
            elif value.get('ready'):
                self.set_visual_state('loading' if value.get('loading') else 'ready')
                self.status.set(f"已连接 · 模型版本 {value['revision']} · 保存代码即可更新" if not value.get('loading') else '正在更新，工作台保留上一份模型…')
                self.open_button.configure(state='normal')
                if not self.auto_opened:
                    self.auto_opened = True
                    self.open_browser()
        if self.process and self.process.poll() is not None:
            self.status.set('服务已退出。请检查 Python 环境和下方错误信息，再重新启动。')
            self.process = None
            self.set_visual_state('error')
            self.lock_settings(False)
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            self.open_button.configure(state='disabled')
        if not self.closed:
            self.poll_job = self.window.after(200, self.poll)

    def stop(self):
        self.generation += 1
        if self.process and self.process.poll() is None:
            # Windows process tree: also stops the capture worker owned by this launcher.
            result = subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'], capture_output=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.status.set('服务未停止，请重试。' + result.stderr.decode(errors='replace')[-120:])
                return
        self.process = None
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.open_button.configure(state='disabled')
        self.status.set('可以选择另一个模型，或重新启动当前模型。')
        self.set_visual_state('stopped')
        self.lock_settings(False)

    def close(self):
        self.closed = True
        self.keynote_stage.pause()
        self.window.after_cancel(self.poll_job)
        if self.intro_job is not None:self.window.after_cancel(self.intro_job)
        self.progress.stop()
        self.stop()
        self.window.destroy()


if __name__ == '__main__':
    enable_high_dpi()
    window = tk.Tk()
    app = Application(window)
    window.mainloop()
