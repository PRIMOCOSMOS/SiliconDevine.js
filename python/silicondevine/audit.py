"""IDE-side recognition audit. Render dependency support is reported by the viewer."""
from collections import Counter
from .exporter import export_model

def audit_model(model,args,**options):
    options.pop('include_values',None)
    graph=export_model(model,args,include_values=False,**options)
    unresolved=[{'name':n['name'],'source':n['source'],'reason':n['attrs'].get('boundary_reason','没有专用识别规则；保留捕获的输入输出。')}
                for n in graph['nodes'] if n['op']=='opaque' or n['attrs'].get('boundary_reason')]
    return {'producer':graph['producer'],'operators':dict(Counter(n['op'] for n in graph['nodes'])),
            'unresolved':unresolved,'notes':graph['notes'],
            'renderSupport':'在 Web 解读面板的「算子支持检查」查看实际渲染插件支持；识别成功不等同于数学动效完整。'}
