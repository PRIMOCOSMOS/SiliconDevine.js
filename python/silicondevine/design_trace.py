"""Record user Python call boundaries while PyTorch builds a concrete graph."""
import inspect
from torch.overrides import TorchFunctionMode
from torch.fx.experimental.proxy_tensor import get_proxy_mode


class DesignTrace(TorchFunctionMode):
    def __init__(self, model):
        super().__init__()
        self.root_code = getattr(getattr(model.forward, '__func__', model.forward), '__code__', None)
        self.frames = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        stack = []
        frame = inspect.currentframe().f_back
        found = False
        while frame:
            code = frame.f_code
            module = frame.f_globals.get('__name__', '')
            if not module.startswith(('torch.', 'silicondevine.')) and code.co_name != '__call__':
                key = id(frame)
                if key not in self.frames and len(self.frames) < 8192:
                    self.frames[key] = (frame, len(self.frames))
                call = self.frames.get(key, (None, -1))[1]
                stack.append({'file': code.co_filename, 'line': frame.f_lineno,
                              'firstLine': code.co_firstlineno, 'function': code.co_qualname,
                              'call': call})
            if code is self.root_code:
                found = True; break
            frame = frame.f_back
        del frame
        mode = get_proxy_mode()
        before = next(iter(reversed(mode.tracer.graph.nodes)), None) if mode else None
        result = func(*args, **(kwargs or {}))
        if found and mode:
            for node in reversed(mode.tracer.graph.nodes):
                if node is before: break
                if node.op == 'call_function':
                    node.meta['silicondevine_design_stack'] = list(reversed(stack))
        return result

    def release(self):
        self.frames.clear()
