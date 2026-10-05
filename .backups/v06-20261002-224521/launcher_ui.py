"""DPI-aware native launcher UI; uses only Python's bundled Tk runtime."""
import ctypes
import os
import math
import time
import tkinter as tk
from tkinter import ttk, font as tkfont

COLORS = dict(bg='#06080d', surface='#10151e', field='#161e2b', line='#293346',
              text='#edf2fa', muted='#a7b4c9', accent='#dce9ff', ink='#142642',
              error='#ffb6af', disabled='#6f8395')


class KeynoteStage(tk.Canvas):
    """Finite, resolution-independent brand geometry; never model or tensor data."""
    def __init__(self, parent, scale, ui, display, heading, root_path):
        super().__init__(parent, bg=COLORS['bg'], highlightthickness=0, width=round(480*scale))
        self.scale,self.ui,self.display=scale,ui,display
        self.heading=heading
        image_path=root_path/'assets'/f'orbit-{400 if scale<1.3 else 600 if scale<1.8 else 800}.png'
        self.orbit_image=tk.PhotoImage(file=str(image_path)) if image_path.exists() else None
        self.job=None
        self.phase=0.
        self.motion=True
        if os.name=='nt':
            enabled=ctypes.c_int(1)
            try:ctypes.windll.user32.SystemParametersInfoW(0x1042,0,ctypes.byref(enabled),0)
            except (AttributeError,OSError):pass
            self.motion=bool(enabled.value)
        self.bind('<Configure>',lambda event:self.draw())
        self.bind('<Map>',lambda event:self.resume())
        self.bind('<Unmap>',lambda event:self.pause())

    def pause(self):
        if self.job is not None:self.after_cancel(self.job)
        self.job=None

    def resume(self):
        if self.job is not None:return
        def frame():
            self.job=None
            if not self.winfo_ismapped():return
            self.phase+=.012
            self.draw()
            if self.motion:self.job=self.after(65,frame)
        frame()

    def draw(self):
        self.delete('all')
        w,h=self.winfo_width(),self.winfo_height()
        if w<10 or h<10:return
        s=self.scale
        self.create_text(40*s,38*s,text='SiliconDevine',anchor='nw',fill='#ecf3ff',font=(self.display,26,'bold'))
        self.create_text(40*s,103*s,text='看见\n计算的形状。',anchor='nw',fill='#e9f0fc',font=(self.heading,40))
        self.create_text(42*s,h-80*s,text='从 PyTorch 代码，到可探索的三维模型。\n保存、连接，走进计算内部。',anchor='nw',fill='#aebfd8',font=(self.ui,10))
        cx,cy=w*.5,max(h*.59,310*s)
        r=min(w*.34,h*.23)
        if self.orbit_image is not None:
            self.create_image(cx,cy,image=self.orbit_image)
            r=self.orbit_image.width()*.4
        # Three orbital arcs share a crystal core; only the highlight moves.
        for ring,(angle,flatten) in enumerate(((.12,.34),(.95,.46),(-.9,.46))):
            points=[]
            for i in range(97):
                a=i*math.tau/96
                x,y=r*math.cos(a),r*flatten*math.sin(a)
                points.extend((cx+x*math.cos(angle)-y*math.sin(angle),cy+x*math.sin(angle)+y*math.cos(angle)))
            if self.orbit_image is None:self.create_line(*points,fill=('#344e6e','#52729a','#849dbd')[ring],width=max(1,s))
            start=self.phase+ring*2
            arc=[]
            for i in range(16):
                a=start+i*.023
                x,y=r*math.cos(a),r*flatten*math.sin(a)
                arc.extend((cx+x*math.cos(angle)-y*math.sin(angle),cy+x*math.sin(angle)+y*math.cos(angle)))
            # One small travelling light per orbit; the smooth body is pre-rendered.
            x,y=arc[len(arc)//2:len(arc)//2+2]
            self.create_oval(x-2*s,y-2*s,x+2*s,y+2*s,fill='#ecf6ff',outline='')
        q=r*.28
        for coords,fill in [((cx,cy-q,cx+q,cy-q*.4,cx,cy+q*.2,cx-q,cy-q*.4),'#293d55'),((cx-q,cy-q*.4,cx,cy+q*.2,cx,cy+q*1.2,cx-q,cy+q*.6),'#101e30'),((cx,cy+q*.2,cx+q,cy-q*.4,cx+q,cy+q*.6,cx,cy+q*1.2),'#1c3048')]:
            if self.orbit_image is None:self.create_polygon(*coords,fill=fill,outline='#a4c2e6',width=max(1,s))
        self.create_line(40*s,h-108*s,w-40*s,h-108*s,fill='#253348')


class LoadingLine(tk.Canvas):
    """A small activity indicator, not a fictitious percentage of model capture."""
    def __init__(self, parent, scale):
        super().__init__(parent, height=max(2,round(2*scale)), bg=COLORS['line'], highlightthickness=0)
        self.job=None
        self.phase=0
        self.stroke=self.create_rectangle(0,0,0,0,fill=COLORS['accent'],outline='',state='hidden')

    def start(self, interval=18):
        if self.job is not None:return
        def frame():
            width=self.winfo_width()
            band=max(30,width*.18)
            self.phase=(self.phase+.008)%1
            x=(width+band)*self.phase-band
            self.itemconfigure(self.stroke,state='normal')
            self.coords(self.stroke,x,0,x+band,self.winfo_height())
            self.job=self.after(30,frame)
        frame()

    def stop(self):
        if self.job is not None:self.after_cancel(self.job)
        self.job=None
        self.itemconfigure(self.stroke,state='hidden')


def enable_high_dpi():
    """Call before Tk creates any HWND, avoiding Windows bitmap enlargement."""
    if os.name != 'nt':
        return
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def build_ui(app):
    w = app.window
    c = COLORS
    scale = max(1., w.winfo_fpixels('1i') / 96.)
    app.ui_scale = scale
    px = lambda n: round(n * scale)
    families = set(tkfont.families(w))
    ui = 'Microsoft YaHei UI' if 'Microsoft YaHei UI' in families else 'Segoe UI'
    font_path=app.root_path/'assets/Tektur.ttf'
    if os.name=='nt' and font_path.exists():
        ctypes.windll.gdi32.AddFontResourceExW(str(font_path),0x10,0)
    display = 'Tektur' if font_path.exists() else 'Segoe UI'
    chinese_path=app.root_path/'assets/ZCOOLQingKeHuangYou-Regular.ttf'
    if os.name=='nt' and chinese_path.exists():ctypes.windll.gdi32.AddFontResourceExW(str(chinese_path),0x10,0)
    heading='ZCOOL QingKe HuangYou' if chinese_path.exists() else ui
    w.title('SiliconDevine · 本地模型工作台')
    if (app.root_path / 'launcher.ico').exists():
        w.iconbitmap(str(app.root_path / 'launcher.ico'))
    width = min(px(1180), w.winfo_screenwidth() - px(50))
    height = min(px(790), w.winfo_screenheight() - px(80))
    w.geometry(f'{width}x{height}')
    w.minsize(px(700), px(540))
    w.configure(bg=c['bg'])
    style = ttk.Style(w)
    style.theme_use('clam')
    style.configure('.', font=(ui, 10), background=c['bg'], foreground=c['text'])
    style.configure('TFrame', background=c['bg'])
    style.configure('TLabel', background=c['bg'])
    style.configure('Muted.TLabel', foreground=c['muted'])
    style.configure('TButton', padding=(px(16), px(10)), borderwidth=0,
                    relief='flat', background=c['field'], foreground=c['text'], focusthickness=1, focuscolor=c['accent'])
    style.map('TButton', background=[('disabled', c['surface']), ('pressed', '#344f62'), ('active', '#253a4c')],
              foreground=[('disabled', c['disabled'])])
    style.configure('Primary.TButton', background=c['accent'], foreground=c['ink'], font=(ui, 11, 'bold'))
    style.map('Primary.TButton', background=[('disabled', c['surface']), ('pressed', '#a8c8f5'), ('active', '#ffffff')],
              foreground=[('disabled', c['disabled'])])
    style.configure('Quiet.TButton', background=c['bg'], foreground=c['muted'], padding=(px(8), px(6)))
    # Native nine-slice surfaces keep vector text crisp at every Windows scale.
    app.button_surfaces=[]
    for prefix in ('', 'Primary.', 'Quiet.'):
        stem=prefix.rstrip('.').lower() or 'normal'
        paths=[app.root_path/'assets'/f'button-{stem}-{state}.png' for state in ('idle','active','pressed','disabled','focus')]
        if all(path.exists() for path in paths):
            photos=[tk.PhotoImage(file=str(path)) for path in paths]
            app.button_surfaces.extend(photos)
            element=f'{stem}.surface'
            style.element_create(element,'image',photos[0],('disabled',photos[3]),('pressed',photos[2]),('focus',photos[4]),('active',photos[1]),border=12,sticky='nsew')
            style.layout(prefix+'TButton',[(element,{'sticky':'nsew','children':[('Button.padding',{'sticky':'nsew','children':[('Button.label',{'sticky':'nsew'})]})]})])
    style.configure('TEntry', fieldbackground=c['field'], foreground=c['text'], insertcolor=c['text'],
                    bordercolor=c['line'], lightcolor=c['field'], darkcolor=c['field'], padding=px(10))
    style.map('TEntry', bordercolor=[('focus', c['accent'])], fieldbackground=[('disabled', c['surface'])],
              foreground=[('disabled', c['disabled'])])
    style.configure('TCombobox', fieldbackground=c['field'], background=c['field'], foreground=c['text'],
                    arrowcolor=c['muted'], bordercolor=c['line'], lightcolor=c['field'], darkcolor=c['field'],
                    padding=px(9), arrowsize=px(15))
    style.map('TCombobox', fieldbackground=[('disabled', c['surface']), ('readonly', c['field'])],
              selectbackground=[('readonly', c['field'])], selectforeground=[('readonly', c['text'])],
              bordercolor=[('focus', c['accent'])], foreground=[('disabled', c['disabled'])])
    w.option_add('*TCombobox*Listbox.background', c['field'])
    w.option_add('*TCombobox*Listbox.foreground', c['text'])
    w.option_add('*TCombobox*Listbox.selectBackground', '#2c514f')
    w.option_add('*TCombobox*Listbox.selectForeground', c['text'])
    w.option_add('*TCombobox*Listbox.font', (ui, 11))
    style.configure('TSeparator',background=c['line'])
    style.configure('Vertical.TScrollbar', background=c['line'], troughcolor=c['bg'],
                    borderwidth=0, arrowsize=px(10))

    shell=ttk.Frame(w)
    shell.pack(fill='both',expand=True)
    shell.columnconfigure(0,weight=1)
    shell.columnconfigure(1,weight=1)
    shell.rowconfigure(0,weight=1)
    app.keynote_stage=KeynoteStage(shell,scale,ui,display,heading,app.root_path)
    app.keynote_stage.grid(row=0,column=0,sticky='nsew')
    outer = ttk.Frame(shell, padding=(px(30), px(36), px(32), px(22)))
    outer.grid(row=0,column=1,sticky='nsew')
    def responsive(event):
        if event.widget!=w:return
        if event.width<px(1000):
            app.keynote_stage.grid_remove()
            shell.columnconfigure(0,weight=0)
        else:
            app.keynote_stage.grid()
            shell.columnconfigure(0,weight=1)
    w.bind('<Configure>',responsive,add='+')
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(2, weight=1)
    header = ttk.Frame(outer)
    header.grid(row=0, column=0, sticky='ew', pady=(0, px(23)))
    header.columnconfigure(0, weight=1)
    ttk.Label(header, text='打开你的模型', font=(heading,31)).grid(row=0,column=0,sticky='w')
    ttk.Label(header, text='选择代码，进入三维计算空间。', style='Muted.TLabel').grid(row=1,column=0,sticky='w',pady=(px(12),0))
    ttk.Button(header, text='使用说明', style='Quiet.TButton', command=lambda: os.startfile(str(app.root_path / 'docs/QUICKSTART.html'))).grid(row=0,column=1,sticky='e')

    # Scroll only the configuration area; status and primary action stay accessible.
    viewport = ttk.Frame(outer)
    viewport.grid(row=2,column=0,sticky='nsew',pady=(px(20),px(16)))
    viewport.columnconfigure(0,weight=1)
    viewport.rowconfigure(0,weight=1)
    canvas = tk.Canvas(viewport,bg=c['bg'],highlightthickness=0,bd=0)
    canvas.grid(row=0,column=0,sticky='nsew')
    scroll = ttk.Scrollbar(viewport,orient='vertical',command=canvas.yview)
    canvas.configure(yscrollcommand=scroll.set)
    content = ttk.Frame(canvas,padding=(0,0,px(3),px(8)))
    content.columnconfigure(0,weight=1)
    item = canvas.create_window(0,0,window=content,anchor='nw')
    def resize(event=None):
        canvas.itemconfigure(item,width=canvas.winfo_width())
        canvas.configure(scrollregion=canvas.bbox('all'))
        if content.winfo_reqheight()>canvas.winfo_height()+2:
            scroll.grid(row=0,column=1,sticky='ns',padx=(px(10),0))
        else:
            scroll.grid_remove()
            canvas.yview_moveto(0)
    content.bind('<Configure>',resize)
    canvas.bind('<Configure>',resize)
    def wheel(event):
        if content.winfo_reqheight()>canvas.winfo_height():
            canvas.yview_scroll(-int(event.delta/120),'units')
    w.bind('<MouseWheel>',wheel,add='+')
    def reveal(event):
        widget=event.widget
        if str(widget).startswith(str(content)):
            y=widget.winfo_rooty()-content.winfo_rooty()
            top=canvas.canvasy(0)
            bottom=top+canvas.winfo_height()
            if y<top or y+widget.winfo_height()>bottom:
                canvas.yview_moveto(max(0,y-px(12))/max(1,content.winfo_height()))
    w.bind('<FocusIn>',reveal,add='+')

    title=ttk.Frame(content)
    title.grid(row=0,column=0,sticky='ew',pady=(0,px(18)))
    title.columnconfigure(0,weight=1)
    ttk.Label(title,text='从示例开始',font=(ui,11,'bold')).grid(row=0,column=0,sticky='w')
    preset=ttk.Combobox(title,values=('选择示例','MLP','TinyGPT','LLaMA · Transformers','SAB · 作者实现','ISAB · 作者实现','VAE · PyTorch 官方','卷积 VAE','条件 VAE'),state='readonly',width=24)
    preset.current(0)
    preset.grid(row=0,column=1,sticky='e',padx=(px(15),0))
    app.preset=preset
    def select_preset(event=None):
        choice=preset.get().split(' · ')[0]
        if choice=='选择示例':return
        if choice in ('VAE','卷积 VAE','条件 VAE'):
            app.file.set(str(app.root_path/'examples/vae_models.py'))
            app.factory.set({'VAE':'build_official_vae','卷积 VAE':'build_conv_vae','条件 VAE':'build_conditional_vae'}[choice])
            app.backend.set('export')
            return
        app.file.set(str(app.root_path / ('examples/live_model.py' if choice=='MLP' else 'examples/official_models.py' if choice in ('SAB','ISAB','LLaMA') else 'examples/llm_models.py')))
        app.factory.set('build_model' if choice=='MLP' else 'build_tinygpt' if choice=='TinyGPT' else 'build_sab' if choice=='SAB' else 'build_isab' if choice=='ISAB' else 'build_llama')
        app.backend.set('fx' if choice=='MLP' else 'export')
    preset.bind('<<ComboboxSelected>>',select_preset)
    app.config_widgets=[(preset,'readonly')]
    def file_row(row,label,variable,action,command):
        ttk.Label(content,text=label).grid(row=row,column=0,sticky='w',pady=(0,px(7)))
        line=ttk.Frame(content)
        line.grid(row=row+1,column=0,sticky='ew',pady=(0,px(18)))
        line.columnconfigure(0,weight=1)
        entry=ttk.Entry(line,textvariable=variable,font=(ui,11))
        entry.grid(row=0,column=0,sticky='ew')
        button=ttk.Button(line,text=action,command=command)
        button.grid(row=0,column=1,padx=(px(10),0),sticky='ns')
        app.config_widgets.extend([(entry,'normal'),(button,'normal')])
    file_row(1,'模型文件',app.file,'选择文件',app.choose)
    file_row(3,'Python 环境 · 已安装 PyTorch 的解释器',app.python,'选择环境',app.choose_python)
    summary=tk.StringVar()
    def sync_summary(*args):
        summary.set(f'{app.backend.get()}  /  {app.factory.get()}()')
    app.backend.trace_add('write',sync_summary)
    app.factory.trace_add('write',sync_summary)
    sync_summary()
    advanced_heading=ttk.Frame(content)
    advanced_heading.grid(row=5,column=0,sticky='ew')
    advanced_heading.columnconfigure(1,weight=1)
    advanced=ttk.Frame(content,padding=(0,px(14),0,px(8)))
    advanced.columnconfigure(1,weight=1)
    ttk.Label(advanced,text='捕获接口').grid(row=0,column=0,sticky='w',padx=(0,px(24)))
    backend=ttk.Combobox(advanced,textvariable=app.backend,values=('fx','export'),state='readonly',width=10)
    backend.grid(row=1,column=0,sticky='w',padx=(0,px(24)),pady=(px(7),px(10)))
    ttk.Label(advanced,text='模型工厂函数').grid(row=0,column=1,sticky='w')
    factory=ttk.Entry(advanced,textvariable=app.factory)
    factory.grid(row=1,column=1,sticky='ew',pady=(px(7),px(10)))
    ttk.Label(advanced,text='普通模块使用 fx；原生 Transformer 建议使用 export。',style='Muted.TLabel').grid(row=2,column=0,columnspan=2,sticky='w')
    app.config_widgets.extend([(backend,'readonly'),(factory,'normal')])
    def toggle_advanced():
        visible=bool(advanced.winfo_manager())
        if visible:advanced.grid_remove()
        else:advanced.grid(row=6,column=0,sticky='ew')
        advanced_button.configure(text='高级设置' if visible else '收起设置')
    advanced_button=ttk.Button(advanced_heading,text='高级设置',style='Quiet.TButton',command=toggle_advanced)
    advanced_button.grid(row=0,column=0,sticky='w')
    ttk.Label(advanced_heading,textvariable=summary,style='Muted.TLabel').grid(row=0,column=1,sticky='e')

    footer=ttk.Frame(outer)
    footer.grid(row=3,column=0,sticky='ew')
    footer.columnconfigure(0,weight=1)
    state_frame=tk.Frame(footer,bg=c['surface'],padx=px(17),pady=px(14))
    state_frame.grid(row=0,column=0,sticky='ew')
    state_frame.columnconfigure(1,weight=1)
    app.state_dot=tk.Canvas(state_frame,width=px(10),height=px(10),bg=c['surface'],highlightthickness=0)
    app.state_dot.grid(row=0,column=0,padx=(0,px(10)))
    app.dot_id=app.state_dot.create_oval(px(1),px(1),px(9),px(9),fill=c['muted'],outline='')
    app.state_title=tk.Label(state_frame,text='准备就绪',font=(ui,11,'bold'),bg=c['surface'],fg=c['text'])
    app.state_title.grid(row=0,column=1,sticky='w')
    status=tk.Label(state_frame,textvariable=app.status,font=(ui,10),bg=c['surface'],fg=c['muted'],anchor='w',justify='left')
    status.grid(row=1,column=1,sticky='ew',pady=(px(6),0))
    state_frame.bind('<Configure>',lambda e: status.configure(wraplength=max(px(250),e.width-px(70))))
    app.progress=LoadingLine(footer,scale)
    app.progress.grid(row=1,column=0,sticky='ew',ipady=0)
    actions=ttk.Frame(footer)
    actions.grid(row=2,column=0,sticky='ew',pady=(px(18),px(12)))
    actions.columnconfigure(3,weight=1)
    app.start_button=ttk.Button(actions,text='进入模型空间',style='Primary.TButton',command=app.start)
    app.start_button.grid(row=0,column=0,columnspan=5,sticky='ew',pady=(0,px(12)))
    app.open_button=ttk.Button(actions,text='打开工作台',command=app.open_browser,state='disabled')
    app.open_button.grid(row=1,column=0,sticky='w')
    app.stop_button=ttk.Button(actions,text='停止服务',style='Quiet.TButton',command=app.stop,state='disabled')
    app.stop_button.grid(row=1,column=1,padx=(px(10),0))
    app.log_panel=ttk.Frame(footer)
    app.log_panel.columnconfigure(0,weight=1)
    app.log=tk.Text(app.log_panel,height=5,bg=c['surface'],fg='#b9cddc',relief='flat',borderwidth=0,
                    padx=px(12),pady=px(10),font=('Consolas',10),wrap='word',state='disabled',
                    insertbackground=c['accent'],selectbackground='#345768',highlightthickness=0)
    app.log.grid(row=0,column=0,sticky='ew')
    log_scroll=ttk.Scrollbar(app.log_panel,command=app.log.yview)
    log_scroll.grid(row=0,column=1,sticky='ns')
    app.log.configure(yscrollcommand=log_scroll.set)
    def toggle_log(show=None):
        visible=bool(app.log_panel.winfo_manager())
        desired=not visible if show is None else show
        if desired:app.log_panel.grid(row=3,column=0,sticky='ew',pady=(0,px(8)))
        else:app.log_panel.grid_remove()
        app.log_toggle.configure(text='收起日志' if desired else '运行日志')
    app.toggle_log=toggle_log
    app.log_toggle=ttk.Button(actions,text='运行日志',style='Quiet.TButton',command=toggle_log)
    app.log_toggle.grid(row=1,column=4,sticky='e')
    ttk.Label(footer,text='保存模型代码，网页自动同步。关闭此窗口会停止本次服务。',style='Muted.TLabel',font=(ui,9)).grid(row=4,column=0,sticky='w')
    app.visual_state=None
    def set_state(state):
        if app.visual_state==state:return
        app.visual_state=state
        title,color={'idle':('准备就绪',c['muted']),'loading':('正在捕获模型',c['accent']),
                     'ready':('工作台已连接',c['accent']),'error':('需要处理',c['error']),
                     'stopped':('服务已停止',c['muted'])}[state]
        app.state_title.configure(text=title,fg=color)
        app.state_dot.itemconfigure(app.dot_id,fill=color)
        if state=='loading':app.progress.start(18)
        else:app.progress.stop()
        if state=='error':toggle_log(True)
    app.set_visual_state=set_state
    app.set_visual_state('idle')
    app.intro_job=None
    introduced=False
    def entrance(event):
        nonlocal introduced
        if event.widget!=w or introduced:return
        introduced=True
        enabled=ctypes.c_int(1)
        if os.name=='nt':
            try:ctypes.windll.user32.SystemParametersInfoW(0x1042,0,ctypes.byref(enabled),0)
            except (AttributeError,OSError):pass
        if not enabled.value:return
        import time
        started=time.monotonic()
        def frame():
            t=min(1,(time.monotonic()-started)/.36)
            w.attributes('-alpha',.88+.12*(1-(1-t)**3))
            app.intro_job=w.after(24,frame) if t<1 else None
        frame()
    w.bind('<Map>',entrance,add='+')
    def lock_settings(locked):
        for widget,normal in app.config_widgets:widget.configure(state='disabled' if locked else normal)
    app.lock_settings=lock_settings
    w.bind('<Control-Return>',lambda e: app.start())
    if os.name=='nt':
        def dark_titlebar():
            try:
                hwnd=ctypes.windll.user32.GetParent(w.winfo_id())
                value=ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd),20,ctypes.byref(value),ctypes.sizeof(value))
                for attribute,rgb in ((35,0x0d0806),(36,0xfaf2ed)):
                    color=ctypes.c_uint(rgb)
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd),attribute,ctypes.byref(color),ctypes.sizeof(color))
            except (AttributeError,OSError):pass
        w.bind('<Map>',lambda event: dark_titlebar() if event.widget==w else None,add='+')
