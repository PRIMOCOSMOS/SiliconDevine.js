"""Bounded source evidence for automatic import; never executes repository scripts.

Resolve literal construction examples and tensor inputs before asking for help.
Conflicting model recipes are retained as ambiguity, not ranked by class names.
"""
import ast
import dataclasses
from dataclasses import dataclass
import inspect
import math
from pathlib import Path
import typing
import sys

UNKNOWN=object()


@dataclass
class Symbol:
    name:str


@dataclass
class SourceTensor:
    descriptor:dict


def _encode(value,inputs=False):
    if isinstance(value,SourceTensor):return value.descriptor if inputs else {'$tensor':value.descriptor}
    if isinstance(value,(list,tuple)):return [_encode(v,inputs) for v in value]
    if isinstance(value,dict):return {k:_encode(v,inputs and '$class' not in value and '$namespace' not in value) for k,v in value.items()}
    return value


@dataclass
class Recipe:
    entry:str
    args:list
    kwargs:dict
    source:str
    line:int


def _literal(value):
    if isinstance(value,SourceTensor):return True
    if value is UNKNOWN or isinstance(value,(Symbol,Recipe)):return False
    if isinstance(value,(list,tuple)):return all(_literal(v) for v in value)
    if isinstance(value,dict):return all(isinstance(k,str) and _literal(v) for k,v in value.items())
    return value is None or type(value) in (str,bool,int) or type(value) is float and math.isfinite(value)


def source_evidence(report, selected=None):
    """Collect source-backed model construction and example forward invocations."""
    pool=report['modules'] if selected else report['candidates']
    targets={c['symbol']:c for c in pool if not selected or c['id']==selected['id']}
    constructors=[];inputs=[];exports={}
    classes={c['symbol']:c for f in report['files'] for c in f.get('classes',[])}
    config_parameters={}
    for record in report['files']:
        exports.update({record['module']+'.'+k:v for k,v in record.get('aliases',{}).items()})
    def canonical(name):
        seen=set()
        while name not in seen:
            seen.add(name)
            key=next((k for k in sorted(exports,key=len,reverse=True) if name==k or name.startswith(k+'.')),None)
            if not key:break
            name=exports[key]+name[len(key):]
        return name
    records={Path(r['path']):r for r in report['files']}
    # Adjacent entry scripts are read only, regardless of their filename.
    for root in {Path(r['root']) for r in report['files']}:
        for path in list(sorted(root.glob('*.py')))[:64]:
            records.setdefault(path,{'path':str(path),'module':path.stem,'aliases':{}})
    for path,record in list(records.items())[:512]:
        try:tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        except (OSError,UnicodeError,SyntaxError):continue
        aliases=dict(record.get('aliases',{}));module=record['module']
        for node in tree.body:
            if isinstance(node,ast.Import):
                for item in node.names:aliases[item.asname or item.name.split('.')[0]]=item.name if item.asname else item.name.split('.')[0]
            elif isinstance(node,ast.ImportFrom) and node.module and not node.level:
                for item in node.names:aliases[item.asname or item.name]=node.module+'.'+item.name
        local_classes={node.name for node in tree.body if isinstance(node,ast.ClassDef)}
        local_functions={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
        def symbol(node):
            if isinstance(node,ast.Name):return canonical(aliases.get(node.id,module+'.'+node.id if node.id in local_classes or node.id in local_functions else node.id))
            if isinstance(node,ast.Attribute):return symbol(node.value)+'.'+node.attr
            return ''
        # Literal defaults of local configuration classes are descriptions, not calls.
        def config_recipe(name,args,kwargs):
            cls=classes.get(canonical(name))
            if cls is None:return UNKNOWN
            # nn.Module constructors are handled separately; only data-like classes.
            if any(c['symbol']==name for c in report['modules']):return UNKNOWN
            if not all(_literal(v) for v in [*args,*kwargs.values()]):return UNKNOWN
            if args:
                if name not in config_parameters:
                    try:
                        config_tree=ast.parse(Path(cls['file']).read_text(encoding='utf-8-sig'))
                        definition=next(n for n in config_tree.body if isinstance(n,ast.ClassDef) and n.name==cls['name'])
                        init_fn=next((n for n in definition.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None)
                        config_parameters[name]=[a.arg for a in [*init_fn.args.posonlyargs,*init_fn.args.args] if a.arg!='self'] if init_fn else [n.target.id for n in definition.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)]
                    except (OSError,SyntaxError,StopIteration):return UNKNOWN
                names=config_parameters[name]
                if len(args)>len(names):return UNKNOWN
                kwargs={**dict(zip(names,args)),**kwargs}
            return {'$class':cls['id'],'kwargs':kwargs}
        nodes=0
        def evaluate(node,env,depth=0):
            nonlocal nodes
            nodes+=1
            if node is None or depth>20 or nodes>30000:return UNKNOWN
            if isinstance(node,ast.Constant):return node.value if _literal(node.value) else UNKNOWN
            if isinstance(node,ast.Name):return env.get(node.id,Symbol(symbol(node)))
            if isinstance(node,(ast.List,ast.Tuple)):
                return [evaluate(v,env,depth+1) for v in node.elts]
            if isinstance(node,ast.Dict):
                out={}
                for key,value in zip(node.keys,node.values):
                    k=evaluate(key,env,depth+1)
                    if not isinstance(k,str):return UNKNOWN
                    out[k]=evaluate(value,env,depth+1)
                return out
            if isinstance(node,ast.Attribute):
                base=evaluate(node.value,env,depth+1)
                if isinstance(base,SourceTensor):return base.descriptor.get('shape',UNKNOWN) if node.attr=='shape' else UNKNOWN
                if isinstance(base,dict):return base.get('$namespace',base).get(node.attr,UNKNOWN)
                if isinstance(base,Symbol):return Symbol(canonical(base.name+'.'+node.attr))
                return UNKNOWN
            if isinstance(node,ast.UnaryOp):
                v=evaluate(node.operand,env,depth+1)
                if type(v) in (int,float):
                    if isinstance(node.op,ast.USub):return -v
                    if isinstance(node.op,ast.UAdd):return v
                if isinstance(node.op,ast.Not) and (v is None or type(v) in (bool,str,int,float)):return not v
                return UNKNOWN
            if isinstance(node,ast.BinOp):
                a,b=evaluate(node.left,env,depth+1),evaluate(node.right,env,depth+1)
                if type(a) not in (int,float) or type(b) not in (int,float) or max(abs(a),abs(b))>1_000_000:return UNKNOWN
                try:
                    if isinstance(node.op,ast.Add):v=a+b
                    elif isinstance(node.op,ast.Sub):v=a-b
                    elif isinstance(node.op,ast.Mult):v=a*b
                    elif isinstance(node.op,ast.FloorDiv):v=a//b
                    elif isinstance(node.op,ast.Div):v=a/b
                    else:return UNKNOWN
                    return v if math.isfinite(v) and abs(v)<=2**53 else UNKNOWN
                except (ArithmeticError,TypeError):return UNKNOWN
            if isinstance(node,ast.Subscript):
                base=evaluate(node.value,env,depth+1);key=evaluate(node.slice,env,depth+1)
                if not _literal(key):return UNKNOWN
                try:return base[key] if isinstance(base,(dict,list,tuple)) else UNKNOWN
                except (KeyError,IndexError,TypeError):return UNKNOWN
            if isinstance(node,ast.Compare) and len(node.ops)==1:
                a,b=evaluate(node.left,env,depth+1),evaluate(node.comparators[0],env,depth+1)
                if isinstance(a,SourceTensor) or isinstance(b,SourceTensor):return UNKNOWN
                if not _literal(a) or not _literal(b):return UNKNOWN
                if isinstance(node.ops[0],ast.Eq):return a==b
                if isinstance(node.ops[0],ast.NotEq):return a!=b
                return UNKNOWN
            if isinstance(node,ast.Call):
                args=[evaluate(v,env,depth+1) for v in node.args]
                kwargs={k.arg:evaluate(k.value,env,depth+1) for k in node.keywords if k.arg}
                for k in node.keywords:
                    if k.arg is None:
                        value=evaluate(k.value,env,depth+1)
                        if not isinstance(value,dict):return UNKNOWN
                        kwargs.update(value)
                fn=evaluate(node.func,env,depth+1)
                if isinstance(fn,Recipe):
                    if all(_literal(v) for v in args) and all(_literal(v) for v in kwargs.values()):
                        inputs.append({'recipe':fn,'inputs':args,'kwargs':kwargs,'source':str(path),'line':node.lineno})
                    return UNKNOWN
                # Preserve model identity through device/evaluation wrappers.
                if isinstance(node.func,ast.Attribute) and node.func.attr in ('eval','train','to','cpu','cuda','float','double'):
                    receiver=evaluate(node.func.value,env,depth+1)
                    if isinstance(receiver,Recipe):return receiver
                name=fn.name if isinstance(fn,Symbol) else symbol(node.func)
                candidate=targets.get(canonical(name))
                if candidate:
                    recipe=Recipe(candidate['id'],args,kwargs,str(path),node.lineno)
                    constructors.append(recipe);return recipe
                if name in ('types.SimpleNamespace','argparse.Namespace') and not args and all(_literal(v) for v in kwargs.values()):return {'$namespace':kwargs}
                if name in ('torch.randn','torch.rand','torch.zeros','torch.ones','torch.randint','torch.tensor'):
                    dtype=kwargs.get('dtype');dtype=dtype.name.rsplit('.',1)[-1] if isinstance(dtype,Symbol) and dtype.name.startswith('torch.') else 'int64' if name=='torch.randint' else 'float32'
                    fill='zeros' if name=='torch.zeros' else 'ones' if name=='torch.ones' else 'random'
                    high=None
                    if name=='torch.tensor':
                        if not args or not _literal(args[0]):return UNKNOWN
                        value=args[0];shape=[]
                        while isinstance(value,list):
                            shape.append(len(value));value=value[0] if value else None
                        if 'dtype' not in kwargs:
                            dtype='bool' if type(value) is bool else 'int64' if type(value) is int else 'float32'
                        desc={'shape':shape,'dtype':dtype,'values':args[0]}
                    else:
                        if name=='torch.randint':
                            if len(args)==2:high,shape=args
                            elif len(args)==3:
                                low,high,shape=args
                                if low!=0:return UNKNOWN
                            else:return UNKNOWN
                        else:shape=args[0] if len(args)==1 and isinstance(args[0],list) else args or kwargs.get('size',UNKNOWN)
                        desc={'shape':shape,'dtype':dtype,'fill':fill}
                        if high is not None:desc['high']=high
                    if not isinstance(desc['shape'],list) or any(type(d) is not int or d<1 for d in desc['shape']) or math.prod(desc['shape'])>2_000_000:return UNKNOWN
                    return SourceTensor(desc) if _literal(desc) else UNKNOWN
                if name in ('dict','builtins.dict') and not args:return kwargs
                return config_recipe(name,args,kwargs)
            return UNKNOWN
        def run(statements,env,depth=0):
            if depth>12:return
            for node in statements:
                if isinstance(node,(ast.Assign,ast.AnnAssign)):
                    value=evaluate(node.value,env)
                    for target in node.targets if isinstance(node,ast.Assign) else [node.target]:
                        if isinstance(target,ast.Name):env[target.id]=value
                elif isinstance(node,ast.Expr):evaluate(node.value,env)
                elif isinstance(node,ast.Return):
                    evaluate(node.value,env);break
                elif isinstance(node,ast.Raise):break
                elif isinstance(node,(ast.With,ast.AsyncWith)):
                    run(node.body,env,depth+1)
                elif isinstance(node,ast.If):
                    condition=evaluate(node.test,env)
                    if condition is None or type(condition) in (bool,str,int,float):run(node.body if condition else node.orelse,env,depth+1)
                    else:
                        run(node.body,dict(env),depth+1);run(node.orelse,dict(env),depth+1)
                # No loop unrolling, arbitrary calls, file loads, or script execution.
        env={}
        run(tree.body,env)
        for fn in local_functions.values():
            local=dict(env);ps=[*fn.args.posonlyargs,*fn.args.args];defaults=[UNKNOWN]*(len(ps)-len(fn.args.defaults))+fn.args.defaults
            for p,v in zip(ps,defaults):local[p.arg]=UNKNOWN if v is UNKNOWN else evaluate(v,env)
            for p,v in zip(fn.args.kwonlyargs,fn.args.kw_defaults):local[p.arg]=UNKNOWN if v is None else evaluate(v,env)
            run(fn.body,local)
        # Literal child construction inside __init__ is useful when the user
        # selects a building block directly. Unknown runtime fields stay unknown.
        for definition in tree.body:
            if not isinstance(definition, ast.ClassDef):continue
            fn=next((n for n in definition.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None)
            if fn is None:continue
            local=dict(env)
            for p in [*fn.args.posonlyargs,*fn.args.args,*fn.args.kwonlyargs]:local[p.arg]=UNKNOWN
            run(fn.body,local)
    return {'constructors':constructors,'inputs':inputs}


def resolve_entry(report,evidence):
    candidates={c['id']:c for c in report['candidates']}
    used={r.entry for r in evidence['constructors'] if r.entry in candidates}
    return candidates[next(iter(used))] if len(used)==1 else None


def bind_recipe(recipe,cls):
    """Keep known fields of a constructor call without inventing missing values."""
    try:signature=inspect.signature(cls)
    except (ValueError,TypeError):return {}
    positional=[p for p in signature.parameters.values() if p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD)]
    out={p.name:v for p,v in zip(positional,recipe.args) if _literal(v)}
    out.update({k:v for k,v in recipe.kwargs.items() if _literal(v)})
    return {k:_encode(v) for k,v in out.items()}


def constructor_values(selected,cls,evidence,provided):
    recipes=[r for r in evidence['constructors'] if r.entry==selected['id']]
    values=[bind_recipe(r,cls) for r in recipes];result={};origins=[]
    for name in set().union(*(v.keys() for v in values)) if values else ():
        options=[v[name] for v in values if name in v]
        # Unknown or conflicting call sites are not silently resolved.
        if len(options)==len(values) and all(v==options[0] for v in options):
            result[name]=options[0]
            origins.append({'parameter':name,'value':options[0],'source':recipes[0].source,'line':recipes[0].line,'basis':'literal constructor call'})
    signature=inspect.signature(cls);module=inspect.getmodule(cls);namespace=vars(module) if module else {}
    for p in signature.parameters.values():
        if p.name in result or p.name in provided or p.default is not p.empty:continue
        annotation=p.annotation
        if isinstance(annotation,str):annotation=namespace.get(annotation.strip("'\""))
        if not inspect.isclass(annotation) or annotation.__module__.split('.')[0] in sys.stdlib_module_names|{'builtins','torch','numpy'}:continue
        if any(base.__module__=='torch.nn.modules.module' and base.__name__=='Module' for base in inspect.getmro(annotation)):continue
        try:
            parameters=inspect.signature(annotation).parameters.values()
            if any(v.default is v.empty and v.kind not in (v.VAR_POSITIONAL,v.VAR_KEYWORD) for v in parameters):continue
            if not dataclasses.is_dataclass(annotation) and not any(v.default is not v.empty for v in parameters) and not any(getattr(c,'__annotations__',{}) for c in inspect.getmro(annotation)):continue
            descriptor={'$class':annotation.__module__+':'+annotation.__qualname__,'kwargs':{}}
            result[p.name]=descriptor;origins.append({'parameter':p.name,'value':descriptor,'basis':'annotated configuration class defaults'})
        except (TypeError,ValueError):pass
    return {**result,**provided},origins


def example_inputs(selected,cls,evidence,init):
    result=[];seen=set()
    for item in evidence['inputs']:
        recipe=item['recipe']
        if recipe.entry!=selected['id']:continue
        known=bind_recipe(recipe,cls)
        if any(k in init and init[k]!=v for k,v in known.items()):continue
        key=repr((item['inputs'],item['kwargs']))
        if key in seen:continue
        seen.add(key)
        result.append({k:_encode(item[k],inputs=True) for k in ('inputs','kwargs','source','line')})
    return result[:3]
