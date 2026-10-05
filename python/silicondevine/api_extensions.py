"""Normalize public nn / functional calls and their ATen equivalents."""
import torch

ACTIVATIONS = {
    'ReLU6':'relu6', 'Hardswish':'hardswish', 'Hardsigmoid':'hardsigmoid',
    'Hardtanh':'hardtanh', 'Softsign':'softsign', 'SELU':'selu', 'CELU':'celu',
    'Mish':'mish', 'LogSigmoid':'logsigmoid', 'LogSoftmax':'log_softmax', 'PReLU':'prelu',
    'Unfold':'unfold2d', 'Fold':'fold2d',
    'PixelShuffle':'pixel_shuffle', 'PixelUnshuffle':'pixel_unshuffle',
    'Dropout1d':'dropout','Dropout2d':'dropout','Dropout3d':'dropout',
    'AlphaDropout':'dropout','FeatureAlphaDropout':'dropout','Unflatten':'reshape',
    'Upsample':'interpolate','UpsamplingNearest2d':'interpolate','UpsamplingBilinear2d':'interpolate',
}

def module_op(module):
    name=type(module).__name__
    if name in ACTIVATIONS:return ACTIVATIONS[name]
    if any(name.startswith(p) for p in ('ZeroPad','ConstantPad','ReflectionPad','ReplicationPad','CircularPad')):
        return 'pad'
    return None

def function_aliases():
    result={v:v for v in ACTIVATIONS.values()}
    result.update({'unfold':'unfold2d','im2col':'unfold2d','fold':'fold2d','col2im':'fold2d',
                   '_log_softmax':'log_softmax','_prelu_kernel':'prelu','log_sigmoid':'logsigmoid','abs':'abs'})
    for mode in ('reflection','replication'):
        result.update({f'{mode}_pad{d}d':'pad' for d in (1,2,3)})
    result.update({'constant_pad_nd':'pad','_pad_circular':'pad'})
    result.update({name:'dropout' for name in ('dropout1d','dropout2d','dropout3d','feature_dropout','alpha_dropout','feature_alpha_dropout')})
    result.update({name:name for name in ('interpolate','gather','index_select','sum','amax','amin')})
    result.update({name:'interpolate' for name in ('upsample_nearest1d','upsample_nearest2d','upsample_nearest3d','_upsample_nearest_exact1d','_upsample_nearest_exact2d','_upsample_nearest_exact3d','upsample_linear1d','upsample_bilinear2d','upsample_trilinear3d')})
    result.update({'grid_sample':'grid_sample2d','grid_sampler_2d':'grid_sample2d','grid_sampler':'grid_sample2d'})
    return result

def normalize(capture,node,module,op,attrs,parameters,inputs):
    def arg(index,default=None):
        return node.args[index] if len(node.args)>index else default
    if module is None and op in ('gather','index_select','grid_sample2d'):
        values=[arg(0,node.kwargs.get('input')),
                arg(1,node.kwargs.get('grid')) if op=='grid_sample2d' else arg(2,node.kwargs.get('index'))]
        inputs[:]=[ref for value in values for ref in capture.deps(value)]
    if module is not None:
        for key in ('min_val','max_val','alpha','dim','value'):
            if hasattr(module,key):attrs[key]=getattr(module,key)
    if op=='dropout':
        name=str(node.target)
        training=module.training if module is not None else arg(2,node.kwargs.get('training','alpha_dropout' not in name))
        probability=getattr(module,'p',arg(1,node.kwargs.get('p',.5)))
        attrs.update(training=bool(training),p=probability)
        if training and probability:
            attrs['boundary_reason']='训练态 Dropout 保留本次随机输出；不使用 eval 恒等映射。'
            return 'opaque'
    if op in ('gather','index_select'):attrs['dim']=arg(1,node.kwargs.get('dim',0))
    if op in ('sum','amax','amin'):
        dims=arg(1,node.kwargs.get('dim',None))
        attrs['dims']=None if dims is None else [dims] if isinstance(dims,int) else list(dims)
        attrs['keepdim']=bool(arg(2,node.kwargs.get('keepdim',False)))
    if op=='interpolate':
        target=str(node.target)
        if module is not None:
            attrs.update(mode=module.mode,align_corners=module.align_corners,
                         scale_factor=module.scale_factor,recompute_scale_factor=module.recompute_scale_factor)
        elif target.startswith('aten.'):
            mode='nearest-exact' if 'nearest_exact' in target else next((s for s in ('nearest','trilinear','bilinear','linear') if s in target),'unknown')
            linear='linear' in mode
            scales=arg(3 if linear else 2,None)
            if not target.endswith('.vec'):
                scales=list(node.args[3 if linear else 2:])
            attrs.update(mode=mode,align_corners=arg(2,False) if linear else False,scale_factor=scales)
        else:
            attrs.update(mode=arg(3,node.kwargs.get('mode','nearest')),
                         align_corners=arg(4,node.kwargs.get('align_corners',False)),
                         scale_factor=arg(2,node.kwargs.get('scale_factor')),
                         recompute_scale_factor=arg(5,node.kwargs.get('recompute_scale_factor',False)),
                         antialias=arg(6,node.kwargs.get('antialias',False)))
        if attrs.get('antialias') or attrs['mode'] not in ('nearest','nearest-exact','linear','bilinear','trilinear'):
            attrs['boundary_reason']='此插值模式保留真实结果；当前逐元素插件覆盖最近邻和非抗锯齿线性插值。'
            return 'opaque'
    if op=='grid_sample2d':
        aten=str(node.target).startswith('aten.')
        attrs.update(mode=arg(2,node.kwargs.get('mode','bilinear')),
                     padding_mode=arg(3,node.kwargs.get('padding_mode','zeros')),
                     align_corners=arg(4,node.kwargs.get('align_corners',False)))
        if aten:
            attrs['mode']={0:'bilinear',1:'nearest',2:'bicubic'}[attrs['mode']]
            attrs['padding_mode']={0:'zeros',1:'border',2:'reflection'}[attrs['padding_mode']]
        source=next(t for t in capture.tensors if t['id']==inputs[0])
        if len(source['shape'])!=4 or attrs['mode'] not in ('bilinear','nearest'):
            attrs['boundary_reason']='Grid Sample 当前逐元素插件覆盖二维 nearest / bilinear；其余保留真实结果。'
            return 'opaque'
    if op=='prelu' and module is None:
        refs=capture.deps(arg(1))
        if refs:parameters['weight']=refs[0];inputs[:]=[i for i in inputs if i not in refs]
    if op in ('pixel_shuffle','pixel_unshuffle'):
        attr='upscale_factor' if op=='pixel_shuffle' else 'downscale_factor'
        attrs['factor']=getattr(module,attr,arg(1,node.kwargs.get(attr,1)))
    if op=='log_softmax':attrs['dim']=getattr(module,'dim',arg(1,node.kwargs.get('dim',-1)))
    if op=='hardtanh' and module is None:
        attrs.update(min_val=arg(1,node.kwargs.get('min_val',-1)),max_val=arg(2,node.kwargs.get('max_val',1)))
    if op=='celu' and module is None:attrs['alpha']=arg(1,node.kwargs.get('alpha',1.))
    if op=='pad':
        name=type(module).__name__ if module is not None else str(node.target)
        mode='reflect' if 'Reflection' in name or 'reflection' in name else 'replicate' if 'Replication' in name or 'replication' in name else 'circular' if 'Circular' in name or 'circular' in name else 'constant'
        if module is not None:
            padding=module.padding;value=getattr(module,'value',0.)
        else:
            padding=arg(1,node.kwargs.get('pad',[]))
            if 'constant_pad_nd' in name:value=arg(2,0.)
            else:
                mode=arg(2,node.kwargs.get('mode',mode));value=arg(3,node.kwargs.get('value',0.))
        attrs.update(pad=list(padding),mode=mode,value=0. if value is None else value)
    if op in ('unfold2d','fold2d') and module is None:
        aten=str(node.target).startswith('aten.')
        keys=(['output_size'] if op=='fold2d' else [])+['kernel_size','dilation','padding','stride']
        for index,key in enumerate(keys,1):
            default=0 if key=='padding' else 1
            attrs[key]=arg(index,node.kwargs.get(key,default))
    # Tensor.unfold is a general sliding view, not the image im2col operation.
    if op=='unfold2d' and (node.op=='call_method' or str(node.target).startswith('aten.unfold.')):
        return 'opaque'
    return op
