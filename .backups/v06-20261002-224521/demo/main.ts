import { SiliconDevineViewer, LiveConnection, defineModel, compileDescription, validateModel, operatorVisual, type Model, type Operation } from '../src';
import '@fontsource-variable/tektur';
import './style.css';
import { installAppearance, modelEntrance } from './appearance';
import katex from 'katex';
import 'katex/dist/katex.min.css';
document.querySelector('#app')!.innerHTML = `
<header><div class="brand"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M20 3 36 12v16L20 37 4 28V12Z M4 12l16 9 16-9M20 21v16M12 8l16 9v15"/></svg><div><h1>SiliconDevine<span>.js</span></h1><p>从代码，进入模型。</p></div></div><div class="version">FRAMEWORK / 0.4.0</div></header>
<main><aside><section><h2>模型来源</h2><label for="example">载入示例</label><select id="example"><option value="live" disabled hidden>本地 IDE · 当前模型</option><option value="declarative">声明式 · 全连接网络</option><option value="mlp">PyTorch · MLP</option><option value="conv2d">PyTorch · 残差卷积</option><option value="conv3d">PyTorch · 3D 卷积</option><option value="attention">PyTorch · 注意力</option><option value="transformer">PyTorch · 原生 Transformer</option><optgroup label="VAE · 变分自编码"><option value="official_vae">VAE · PyTorch 官方原型</option><option value="conv_vae">卷积 VAE · 本地示例</option><option value="conditional_vae">条件 VAE · 本地示例</option></optgroup><optgroup label="上游原始实现"><option value="official_llama">LLaMA · Transformers 原生</option><option value="official_sab">SAB · 作者原代码</option><option value="official_isab">ISAB · 作者原代码</option></optgroup><optgroup label="缩小自建测试模型"><option value="tinygpt">TinyGPT · 因果注意力</option><option value="llama">LLaMA-style · RoPE / GQA</option></optgroup><optgroup label="旧版自建测试模型"><option value="sab">SAB · 集合自注意力</option><option value="isab">ISAB · 诱导点双向聚合</option></optgroup></select><div class="file-row"><label class="button" for="file">导入模型 JSON</label><input type="file" id="file" accept=".json,application/json" hidden><button id="download">导出</button></div><p class="muted" id="source">固定种子演示数值</p></section>
<details id="code-origin"><summary>代码来源与识别依据</summary><p id="provenance"></p></details><section class="live-panel"><h2>连接本地 IDE</h2><label for="live-url">PyTorch 服务地址</label><input id="live-url" value="http://127.0.0.1:5182" type="url"><div class="file-row"><button id="connect">连接</button><button id="disconnect" disabled>断开</button></div><p id="live-status" class="muted" role="status">运行 start-live.cmd，然后保存模型代码。</p><details><summary>原生 PyTorch 用法</summary><pre>from silicondevine import show
show(model, (example_input,))

# 保存后自动更新：
python -m silicondevine.watch model.py</pre></details></section><section id="params"><h2>张量维度</h2><div class="fields"><label>Batch<input id="batch" type="number" min="1" max="4" value="2"></label><label>输入通道<input id="channels" type="number" min="1" max="32" value="8"></label><label>隐藏通道<input id="hidden" type="number" min="1" max="32" value="12"></label><label>输出通道<input id="outputs" type="number" min="1" max="32" value="4"></label></div><button id="rebuild" class="wide">更新模型</button></section>
<section class="index"><h2>计算图索引 <span id="node-count"></span></h2><p class="muted">选择模块聚焦，悬停区域接管动效。</p><label for="modules">模块层级</label><select id="modules"><option value="">全部模块</option></select><nav id="outline" aria-label="计算模块"></nav></section>
<details><summary>张量坐标窗口</summary><label for="tensor-select">当前张量</label><select id="tensor-select"></select><label for="origin">各维起点（逗号分隔）</label><input id="origin" value="0,0"><button id="window-apply" class="wide">查看坐标窗口</button><p id="window-status" class="muted">连接后按需读取窗口外的实际数值。</p></details><details><summary>声明式接入</summary><pre>defineModel('My network')
  .input('x', [2, 8])
  .linear('encoder', 12)
  .activation('gate', 'gelu')
  .linear('output', 4)
  .build()</pre><p class="muted">任意分支可通过 operation() 或标准图 JSON 接入。浏览器不执行 Python。</p></details></aside>
<div class="workspace"><div class="toolbar"><div><strong id="model-name">全连接网络</strong><span id="active">等待模型</span></div><div class="controls"><button id="play">暂停</button><button id="fit">总览</button><button id="zoom-in" aria-label="放大">＋</button><button id="zoom-out" aria-label="缩小">−</button><label class="sr" for="labels">全息标牌显示</label><select id="labels"><option value="auto">按需标牌</option><option value="all">全部标牌</option><option value="none">隐藏标牌</option></select><label class="sr" for="speed">演示速度</label><select id="speed"><option value="0.5">0.5× 速度</option><option value="1" selected>1× 速度</option><option value="2">2× 速度</option></select><button id="fullscreen">全屏</button></div></div>
<div class="navigation"><button id="parent-module">上一级</button><span id="scope-label">完整模型</span><label for="representation">显示层级</label><select id="representation"><option value="architecture">功能架构</option><option value="operators">算子细节</option></select></div><div id="stage" tabindex="0"></div><div class="status"><span id="stats"></span><span id="computation"></span><div><button id="prev">上一图段</button><button id="next">下一图段</button></div></div>
<div class="timeline"><label for="progress">单步计算进度</label><input id="progress" type="range" min="0" max="100" value="0"><span>拖动暂停并定位</span></div><div class="legend" aria-label="数值颜色图例"><span><i style="--c:#ffc47e"></i>负值</span><span><i style="--c:#879bae"></i>零</span><span><i style="--c:#68e6d2"></i>正值</span><span><i style="--c:#effaff"></i>当前计算</span><span><i style="--c:#4b6574"></i>未知数值</span><span>× 屏蔽值 −∞</span><small>按张量归一化 · 对称 asinh</small></div>
<section class="inspector"><div><h2 id="detail-title">完整权重连接，逐次计算</h2><p id="detail">晶体代表具有明确坐标的张量元素。权重线保持可见，动效沿计算过程平滑流动。复杂图只激活当前区域；点击聚焦，拖动旋转。</p></div><div><code id="formula">y = Wx + b</code><p id="shape"></p><p id="support" class="muted"></p></div></section>
<div id="error" role="alert" hidden></div></div></main>`;
installAppearance();
const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
let model: Model, pageStart = 0, request = 0;
let live: LiveConnection | undefined;
let liveUpdating = false;
let navigationSignature = '';
const stage = el('stage');
function detail(n?: Operation) {
    if (!n) {
        el('detail-title').textContent = viewer.getNavigation().scope || '整体计算图';
        el('formula').textContent = '';
        el('shape').textContent = model.inputs.join(' / ') + ' → ' + model.outputs.join(' / ');
        el('support').textContent = model.notes?.[0] ?? '';
        el('detail').textContent = '水晶块保留真实坐标；参数与计算主干对齐。拖动旋转，滚轮缩放；悬停连续区域立即切换局部计算动效。';
        if(model.functionalUnits?.some(u=>u.kind==='gaussian_reparameterization')) {
            el('detail-title').textContent='高斯潜空间 · 从编码到重建';
            el('detail').textContent='编码特征分别进入均值头与对数方差头，定义逐维独立的高斯后验。采样模块将本次标准高斯噪声缩放、平移为潜变量 z，解码路径使用这个实际采样值。选中采样模块可检查原始算子；悬停区域显示局部动效。';
            el('formula').innerHTML=katex.renderToString(String.raw`q_\phi(z\mid x)=\mathcal N\!\left(\mu,\operatorname{diag}(\sigma^2)\right)`,{throwOnError:false});
            el('support').textContent='通常使用重建项与 KL 正则训练；这属于训练目标说明，不代表此处已捕获反向传播或损失计算。';
        }
        return;
    }
    const v = operatorVisual(n.op);
    const gaussian = n.attrs?.gaussian;
    if (gaussian) {
        el('detail-title').textContent = '高斯后验 · 重参数采样';
        el('formula').innerHTML = katex.renderToString(String.raw`z=\mu+\exp(\tfrac12\log\sigma^2)\odot\epsilon,\quad\epsilon\sim\mathcal N(0,I)`,{throwOnError:false});
        el('detail').textContent = '均值头和对数方差头描述 q(z|x)。标准噪声经缩放和平移得到潜变量，随后交给解码器重建输入。曲线在标准高斯与当前坐标的后验分布之间连续变换；晶体跟随本次捕获的 ε，不在浏览器重新抽样。';
        el('shape').textContent = n.outputs.map(id => '[' + model.tensors.find(t=>t.id===id)!.shape.join(' × ') + ']').join(' / ');
        el('support').textContent = '曲线按峰值归一化以便比较宽度，不是样本直方图。训练通常平衡重建项与 KL(q(z|x) || N(0,I))；当前视图只展示源码实际执行的前向过程。';
        return;
    }
    el('detail-title').textContent = `${v.label} · ${n.name}`;
    el('formula').textContent = v.formula;
    el('detail').textContent = `输入 ${n.inputs.join('、')} → 输出 ${n.outputs.join('、')}。来源：${n.source ?? '声明式定义'}。`;
    el('shape').textContent = n.outputs.map(id => { const t = model.tensors.find(t => t.id === id)!; return `${id} [${t.shape.join(' × ')}]`; }).join(' / ');
    el('support').textContent = n.op === 'module' ? `${n.attrs?.operatorCount} 个真实算子；已进入 ${n.attrs?.modulePath}。切换算子细节可展开全部计算。` : v.detail === 'exact' ? '按算子定义连接可见窗口中的实际元素。窗口之外的坐标仍保留在模型元数据中。' : '此算子保留实际输入输出与拓扑，内部数学动效尚未实现；可注册专用插件。';
    for (const b of el('outline').querySelectorAll('button'))
        b.setAttribute('aria-current', String(b.dataset.id === n.id));
}
const viewer = new SiliconDevineViewer(stage, { onSelect: detail, onStats: s => { syncNavigation(); pageStart = s.pageStart; el('computation').textContent = s.gaussian ? `潜变量 [${s.gaussian.coordinate.join(',')}] · μ ${s.gaussian.mean.toFixed(3)} · σ ${s.gaussian.std.toFixed(3)} · ε ${s.gaussian.epsilon.toFixed(3)} → z ${s.gaussian.sample.toFixed(3)}` : s.receptiveField ? `感受野 · 输出 [${s.receptiveField.output.join(',')}] · ${s.receptiveField.visibleSamples}/${s.receptiveField.totalSamples} 个输入坐标 · 核 ${s.receptiveField.kernel.join('×')}` : s.attention ? `Query ${s.attention.query} · ${s.attention.probabilities.length} 个 Key → 加权汇聚成输出 Token` : s.mechanism ? `${s.mechanism.kind} · 行 ${s.mechanism.row}${s.mechanism.sum !== undefined ? ' · Σp = ' + s.mechanism.sum.toFixed(3) : s.mechanism.rms !== undefined ? ' · RMS = ' + s.mechanism.rms.toFixed(3) : ''}` : s.weight ? `W[${s.weight.coordinates.join(',')}] = ${Number.isFinite(s.weight.value) ? s.weight.value.toFixed(4) : '未知'} · 矩阵 / 连线同步` : s.sample ? `x = ${s.sample.input.toFixed(4)} → y = ${s.sample.output.toFixed(4)}` : ''; el('active').textContent = s.active ? `计算区域 · ${s.active}` : ''; el('stats').textContent = `${s.cells} 个可见元素 · ${s.connections} 条联系 · ${s.drawCalls} 次绘制${s.partialTensors ? ` · ${s.partialTensors} 个坐标窗口` : ''}${s.unknownTensors ? ` · ${s.unknownTensors} 个未导出数值的张量` : ''}`; el<HTMLButtonElement>('prev').disabled = s.pageStart === 0; el<HTMLButtonElement>('next').disabled = s.pageStart + s.visibleNodes >= s.totalNodes; } });
// Debug information is read-only; a host application can use getStats() directly.
Object.assign(window, { siliconDevine: { viewer, get model() { return model; } } });
function error(e: unknown) { el('error').hidden = false; el('error').textContent = e instanceof Error ? e.message : String(e); }
function load(m: unknown) {
    const valid = (m as {
        format?: string;
    })?.format === 'silicondevine-recipe' ? compileDescription(m) : validateModel(m);
    if (liveUpdating && model)
        viewer.update(valid);
    else
        viewer.load(valid);
    model = valid;
    el('error').hidden = true;
    el('model-name').textContent = ({official_vae:'VAE · 高斯潜空间',conv_vae:'卷积 VAE',conditional_vae:'条件 VAE',official_llama:'LLaMA · 原生架构',official_sab:'SAB · 集合自注意力',official_isab:'ISAB · 诱导点注意力'} as Record<string,string>)[model.name] ?? model.name;
    if (!liveUpdating) modelEntrance(stage);
    el('node-count').textContent = String(model.nodes.length);
    el('source').textContent = `${model.producer?.backend ?? 'JSON'} · ${model.producer?.version ?? 'v1 图格式'}`;
    refreshSelectors();
    navigationSignature = '';
    syncNavigation();
    detail();
}
function generated() {
    const read = (id: string, max: number) => {
        const v = Number(el<HTMLInputElement>(id).value);
        if (!Number.isInteger(v) || v < 1 || v > max)
            throw Error(`请输入 1–${max} 之间的整数。`);
        return v;
    };
    return defineModel('全连接 · Crystal MLP').input('x', [read('batch', 4), read('channels', 32)]).linear('hidden', read('hidden', 32)).activation('gelu', 'gelu').linear('output', read('outputs', 32)).build();
}
el('rebuild').onclick = () => {
    disconnect();
    try {
        load(generated());
    }
    catch (e) {
        error(e);
    }
};
el<HTMLSelectElement>('example').onchange = async (e) => {
    disconnect();
    const id = ++request;
    try {
        const value = (e.target as HTMLSelectElement).value;
        el('params').hidden = value !== 'declarative';
        if (value === 'declarative') {
            load(generated());
            return;
        }
        const response = await fetch(`./models/${value}.sd.json`);
        if (!response.ok)
            throw Error('示例未生成，请运行 npm run examples。');
        const m = await response.json();
        if (id === request)
            load(m);
    }
    catch (e) {
        error(e);
    }
};
el<HTMLInputElement>('file').onchange = async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file)
        return;
    try {
        if (file.size > 32 * 1024 * 1024)
            throw Error('JSON 超过 32 MB，请减少数值导出窗口。');
        disconnect();
        ++request;
        load(JSON.parse(await file.text()));
        el('params').hidden = true;
    }
    catch (e) {
        error(e);
    }
    finally {
        el<HTMLInputElement>('file').value = '';
    }
};
el('download').onclick = () => { const url = URL.createObjectURL(new Blob([JSON.stringify(model, null, 2)], { type: 'application/json' })); const a = document.createElement('a'); a.href = url; a.download = 'model.sd.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); };
el('play').onclick = () => { viewer.setPlaying(!viewer.isPlaying); el('play').textContent = viewer.isPlaying ? '暂停' : '播放'; };
if (matchMedia('(prefers-reduced-motion: reduce)').matches) viewer.setPlaying(false);
el('play').textContent = viewer.isPlaying ? '暂停' : '播放';
el('fit').onclick = () => viewer.clearFocus();
el('zoom-in').onclick = () => viewer.zoom(.8);
el('zoom-out').onclick = () => viewer.zoom(1.25);
el('fullscreen').onclick = () => void viewer.fullscreen().catch(error);
el<HTMLSelectElement>('labels').onchange = e => viewer.setLabels((e.target as HTMLSelectElement).value as 'auto' | 'all' | 'none');
el('prev').onclick = () => viewer.setPage(Math.max(0, pageStart - 32));
el('next').onclick = () => viewer.setPage(pageStart + 32);
stage.addEventListener('keydown', e => {
    if (e.key === '+' || e.key === '=')
        viewer.zoom(.8);
    if (e.key === '-')
        viewer.zoom(1.25);
    if (e.key === 'Home')
        viewer.clearFocus();
    if (e.key === ' ') {
        e.preventDefault();
        el('play').click();
    }
});
window.addEventListener('beforeunload', () => { live?.stop(); viewer.dispose(); }, { once: true });
load(generated());
function refreshSelectors() {
    el('modules').replaceChildren();
    el<HTMLSelectElement>('modules').add(new Option('全部模块', ''));
    for (const path of viewer.getModules())
        el<HTMLSelectElement>('modules').add(new Option('　'.repeat(path.split('.').length - 1) + path, path));
    el('tensor-select').replaceChildren();
    for (const t of model.tensors)
        el<HTMLSelectElement>('tensor-select').add(new Option(`${t.id} [${t.shape.join('×')}]`, t.id));
    syncOrigin();
}
function syncOrigin() { const t = model.tensors.find(t => t.id === el<HTMLSelectElement>('tensor-select').value); el<HTMLInputElement>('origin').value = t?.shape.map(() => 0).join(',') ?? ''; }
el<HTMLSelectElement>('tensor-select').onchange = syncOrigin;
el<HTMLSelectElement>('modules').onchange = e => { viewer.showModule((e.target as HTMLSelectElement).value); syncNavigation(); detail(); };
el('window-apply').onclick = async () => {
    try {
        el('window-status').textContent = '读取坐标窗口…';
        await viewer.setTensorWindow(el<HTMLSelectElement>('tensor-select').value, el<HTMLInputElement>('origin').value.split(',').filter(v => v.trim() !== '').map(Number));
        el('window-status').textContent = live ? '已读取真实数值窗口。' : '窗口已移动；未导出的数值显示为未知。';
    }
    catch (e) {
        el('window-status').textContent = e instanceof Error ? e.message : String(e);
    }
};
function disconnect() { live?.stop(); live = undefined; viewer.setDataProvider(undefined); el<HTMLButtonElement>('connect').disabled = false; el<HTMLButtonElement>('disconnect').disabled = true; el('live-status').textContent = '连接已断开。'; }
el('disconnect').onclick = disconnect;
el('connect').onclick = () => {
    try {
        live?.stop();
        const connection = new LiveConnection(el<HTMLInputElement>('live-url').value, { onModel: m => {
                liveUpdating = !!model && model.name === m.name && model.producer?.backend === m.producer?.backend;
                try {
                    load(m);
                    el('params').hidden = true;
                    el<HTMLSelectElement>('example').value = 'live';
                }
                finally {
                    liveUpdating = false;
                }
            }, onStatus: s => { el('live-status').textContent = s.error ? '捕获失败，保留上一份模型：' + s.error.slice(-350) : s.loading ? '正在捕获模型…' : s.ready ? `已连接 · 版本 ${s.revision} · 保存代码自动更新` : '等待 IDE 提供模型…'; }, onError: e => { el('live-status').textContent = '连接未就绪，正在重试：' + e.message; } });
        live = connection;
        viewer.setDataProvider((id, indices, signal) => connection.tensorWindow(id, indices, signal));
        connection.start();
        el<HTMLButtonElement>('connect').disabled = true;
        el<HTMLButtonElement>('disconnect').disabled = false;
    }
    catch (e) {
        error(e);
    }
};
if (new URLSearchParams(location.search).has('live')) {
    el<HTMLInputElement>('live-url').value = location.origin;
    el('connect').click();
}
el<HTMLSelectElement>('speed').onchange = e => viewer.setSpeed(Number((e.target as HTMLSelectElement).value));
el<HTMLInputElement>('progress').oninput = e => { viewer.setProgress(Number((e.target as HTMLInputElement).value) / 100); el('play').textContent = '播放'; };
function syncNavigation() {
    if (model)
        el('provenance').textContent = model.provenance ? JSON.stringify(model.provenance, null, 2) : '原生 PyTorch 捕获；功能识别根据实际运算依赖，未匹配结构保留原始算子。';
    const nav = viewer.getNavigation(), signature = [nav.scope, nav.function ?? '', nav.representation, ...nav.nodes.map(n => n.id)].join('|');
    if (signature === navigationSignature)
        return;
    navigationSignature = signature;
    el('scope-label').textContent = (nav.function ?? nav.scope) || '完整模型';
    el<HTMLButtonElement>('parent-module').disabled = !nav.scope && !nav.function;
    el<HTMLSelectElement>('representation').value = nav.representation;
    el<HTMLSelectElement>('modules').value = nav.scope;
    el('outline').replaceChildren();
    for (const n of nav.nodes) {
        const b = document.createElement('button');
        b.dataset.id = n.id;
        b.textContent = `${n.op === 'module' ? n.attrs?.moduleType : operatorVisual(n.op).label} · ${n.name.split('.').slice(-2).join('.')}`;
        b.title = n.source ?? n.op;
        b.onclick = () => { viewer.focus(n.id); syncNavigation(); };
        el('outline').append(b);
    }
}
el('parent-module').onclick = () => { viewer.parentModule(); syncNavigation(); detail(); };
el<HTMLSelectElement>('representation').onchange = e => { viewer.setRepresentation((e.target as HTMLSelectElement).value as 'architecture' | 'operators'); syncNavigation(); };
