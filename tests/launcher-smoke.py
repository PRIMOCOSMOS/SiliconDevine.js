"""Exercise the actual desktop start/status/stop path without opening user windows."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
import sys
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location('launcher', root / 'app.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix='sd-launcher-test-') as directory:
    module.CONFIG = Path(directory) / 'settings.json'
    opened = []
    module.webbrowser.open = opened.append
    module.enable_high_dpi()
    window = tk.Tk()
    window.withdraw()
    app = module.Application(window)
    try:
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        preset=next(w for w in descendants(window) if w.winfo_class()=='TCombobox' and 'LLaMA · Transformers' in w.cget('values'))
        for choice,factory in [('TinyGPT','build_tinygpt'),('SAB · 作者实现','build_sab'),('ISAB · 作者实现','build_isab'),('LLaMA · Transformers','build_llama')]:
            preset.set(choice)
            preset.event_generate('<<ComboboxSelected>>')
            window.update()
            assert app.factory.get()==factory
            assert app.backend.get()=='export'
            assert Path(app.file.get()).exists()
        app.start()
        process = app.process
        assert process is not None
        deadline = time.time() + 130
        while not app.auto_opened and time.time() < deadline:
            window.update()
            time.sleep(.1)
        assert app.auto_opened, app.status.get()+'\n'+app.log.get('1.0','end')
        assert opened == [app.url]
        with urlopen(app.url.split('?')[0] + 'api/model') as response:
            data = json.load(response)
        assert data['model']['provenance']['implementation'].endswith('LlamaForCausalLM')
        assert sum(u['kind']=='attention' for u in data['model']['functionalUnits'])==2
        assert module.CONFIG.exists()
        app.stop()
        assert process.poll() is not None
        assert app.process is None
        print('Desktop launcher: four presets, LLaMA capture, browser handoff, settings and process-tree stop passed.')
    finally:
        app.close()
