"""Local desktop launcher. The browser is the 3D workspace; this window owns its server."""
import json
import hashlib
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
from urllib.request import urlopen, Request
import webbrowser
from launcher_ui import build_ui, enable_high_dpi
from desktop_runtime import application_root, default_python, external_environment, external_libraries, startup_issue, install_command

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
        self.issue = None
        self.recent_log = []
        self.control_token = None
        try:
            self.saved = json.loads(CONFIG.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            self.saved = {}
        self.file = tk.StringVar(value=self.saved.get('file', str(ROOT / 'examples/live_model.py')))
        self.backend = tk.StringVar(value=self.saved.get('backend', 'auto'))
        self.factory = tk.StringVar(value=self.saved.get('factory', 'auto'))
        self.entry = tk.StringVar(value=self.saved.get('entry', ''))
        self.model_config = tk.StringVar(value=self.saved.get('model_config', ''))
        self.status = tk.StringVar(value='导入模型目录或多个文件，自动发现模型；保存代码即可更新。')
        self.python = tk.StringVar(value=self.saved.get('python') or default_python())
        self.timeout = tk.StringVar(value=str(self.saved.get('timeout',90)))
        self.issue_command = tk.StringVar()
        self.root_path = ROOT
        build_ui(self)
        window.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_job = window.after(150, self.poll)

    def choose(self):
        selected = filedialog.askopenfilenames(title='选择一个或多个模型文件', filetypes=[('Python 模型', '*.py')])
        if selected:
            self.file.set(selected[0] if len(selected) == 1 else json.dumps(list(selected), ensure_ascii=False))
            self.factory.set('auto')
            self.backend.set('auto')
            self.entry.set('')
            self.model_config.set('')

    def choose_directory(self):
        selected = filedialog.askdirectory(title='选择模型项目目录（保留原有包结构）')
        if selected:
            self.file.set(selected)
            self.factory.set('auto')
            self.backend.set('auto')
            self.entry.set('')
            self.model_config.set('')

    def choose_config(self):
        selected = filedialog.askopenfilename(title='选择模型导入配置', filetypes=[('JSON 配置', '*.json')])
        if selected:
            self.model_config.set(selected)

    def choose_python(self):
        selected = filedialog.askopenfilename(title='选择包含 PyTorch 的 Python 环境', filetypes=[('Python', '*.exe')])
        if selected:
            self.python.set(selected)

    def recover_environment(self):
        selected=filedialog.askopenfilename(title='选择该项目所用的 Python 环境',filetypes=[('Python','*.exe')])
        if selected:
            self.stop()
            self.python.set(selected)
            self.start()

    def show_issue(self, issue):
        self.issue=issue
        self.set_visual_state('error')
        self.state_title.configure(text=issue['title'])
        self.status.set(issue['message'])
        self.issue_panel.grid(row=0,column=0,sticky='ew',pady=(0,16))
        self.issue_hint.set('在本地工作台中选择入口、填写参数并重试。' if issue.get('schema') else '当前 Python：'+self.python.get())
        if issue.get('schema') and self.url:self.assist_button.grid()
        else:self.assist_button.grid_remove()
        command=install_command(issue,self.python.get());self.issue_command.set(command)
        if command:self.install_row.grid()
        else:self.install_row.grid_remove()

    def clear_issue(self):
        self.issue=None
        self.issue_panel.grid_remove()

    def copy_install(self):
        self.window.clipboard_clear();self.window.clipboard_append(self.issue_command.get())
        self.status.set('安装命令已复制。请在对应环境的终端中执行，完成后重试。')

    def retry(self):
        if self.process and self.process.poll() is None:
            if not self.control_token:return self.open_browser()
            generation=self.generation
            def submit():
                try:
                    request=Request(self.url.split('?')[0]+'api/retry',data=b'{}',headers={'Content-Type':'application/json','X-SiliconDevine-Token':self.control_token})
                    with urlopen(request,timeout=3):pass
                    self.events.put((generation,'retry_done',None))
                except OSError as exc:self.events.put((generation,'retry_error',str(exc)))
            threading.Thread(target=submit,daemon=True).start()
        else:self.start()

    def validation_error(self,title,message):
        self.show_issue({'code':'launcher_validation','title':title,'message':message})

    def start(self):
        if self.process and self.process.poll() is None:
            return
        try:
            raw = self.file.get().strip()
            values = json.loads(raw) if raw.startswith('[') else [raw]
            if not isinstance(values, list) or not values or not all(isinstance(p, str) for p in values):
                raise ValueError('文件列表格式无效')
            sources = [Path(p).expanduser().resolve() for p in values]
            if any(not (p.is_dir() or p.is_file() and p.suffix.lower() == '.py') for p in sources):
                raise ValueError('请选择存在的模型目录或 Python 文件。')
        except (ValueError, OSError) as exc:
            return self.validation_error('请选择模型代码',str(exc))
        if not self.factory.get().isidentifier():
            return self.validation_error('入口函数名称无效','自动导入请使用 auto；已有工厂函数请填写有效的 Python 函数名。')
        interpreter = Path(self.python.get()).expanduser()
        if not interpreter.is_file() or (getattr(sys, 'frozen', False) and interpreter.resolve() == Path(sys.executable).resolve()):
            return self.validation_error('请选择 Python 环境','请选择项目所用、已安装 PyTorch 的 python.exe。')
        if not (ROOT / 'demo-dist/index.html').exists():
            return self.validation_error('工作台文件不完整','应用目录缺少网页文件。请恢复完整的应用发布目录，再重新打开。')
        try:
            timeout=float(self.timeout.get())
            if not 10<=timeout<=3600:raise ValueError()
        except ValueError:return self.validation_error('捕获时限无效','请在高级设置中填写 10 至 3600 秒。')
        profile_key=hashlib.sha256('|'.join(map(str,sources)).encode()).hexdigest()[:20]
        profile=CONFIG.parent/'import-profiles'/(profile_key+'.json')
        self.clear_issue();self.recent_log=[];self.control_token=None
        # Use an available local port so another workspace is never disturbed.
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = external_environment()
        env['PYTHONPATH'] = str(ROOT / 'python') + os.pathsep + env.get('PYTHONPATH', '')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        command = [str(interpreter.resolve()), '-u', '-m', 'silicondevine.watch', *map(str, sources), '--factory', self.factory.get(), '--backend', self.backend.get(), '--port', str(port), '--static-dir', str(ROOT / 'demo-dist')]
        command += ['--settings',str(profile),'--timeout',str(timeout)]
        if self.entry.get().strip():command += ['--entry', self.entry.get().strip()]
        if self.model_config.get().strip():command += ['--config', self.model_config.get().strip()]
        try:
            with external_libraries():
                self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as exc:
            return self.validation_error('无法启动 Python',str(exc))
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
            CONFIG.write_text(json.dumps({key: value.get() for key, value in [('file', self.file), ('backend', self.backend), ('factory', self.factory), ('python', self.python), ('entry', self.entry), ('model_config', self.model_config),('timeout',self.timeout)]}), encoding='utf-8')
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
                self.recent_log=(self.recent_log+[value])[-100:]
                self.log.configure(state='normal')
                self.log.insert('end', value + '\n')
                if int(self.log.index('end-1c').split('.')[0]) > 160:
                    self.log.delete('1.0', '40.0')
                self.log.see('end')
                self.log.configure(state='disabled')
            elif kind == 'retry_done':
                self.clear_issue();self.set_visual_state('loading');self.status.set('正在重新捕获当前模型…')
            elif kind == 'retry_error':
                self.status.set('重试未提交。请打开工作台重试，或停止服务后重新启动。')
            elif value.get('error'):
                self.control_token=value.get('controlToken')
                self.show_issue(value.get('issue') or {'code':'capture_failed','title':'模型需要补充信息','message':value['error']})
                self.open_button.configure(state='normal')
                if (value.get('issue') or {}).get('schema') and not self.auto_opened:
                    self.auto_opened=True;self.open_browser()
            elif value.get('loading') and not value.get('ready'):
                self.control_token=value.get('controlToken')
                self.clear_issue()
                self.set_visual_state('loading')
                self.status.set(f'正在首次捕获 · 已等待 {int(time.monotonic()-self.started_at)} 秒 · 完成后自动打开工作台')
            elif value.get('ready'):
                self.control_token=value.get('controlToken')
                self.clear_issue()
                self.set_visual_state('loading' if value.get('loading') else 'ready')
                self.status.set(f"已连接 · 模型版本 {value['revision']} · 保存代码即可更新" if not value.get('loading') else '正在更新，工作台保留上一份模型…')
                self.open_button.configure(state='normal')
                if not self.auto_opened:
                    self.auto_opened = True
                    self.open_browser()
        if self.process and self.process.poll() is not None:
            self.show_issue(startup_issue('\n'.join(self.recent_log),self.python.get()))
            self.process = None
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
            result = None
            if os.name == 'nt':
                result = subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'], capture_output=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.status.set('服务未停止，请重试。' + (result.stderr.decode(errors='replace')[-120:] if result else ''))
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
