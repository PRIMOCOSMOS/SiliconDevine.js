import { SiliconDevineViewer, defineModel, compileDescription, validateModel, operatorVisual, type Model, type Operation } from '../src';
import '@fontsource-variable/tektur';
import './style.css';
document.querySelector('#app')!.innerHTML = `
<header><div class="brand"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M20 3 36 12v16L20 37 4 28V12Z M4 12l16 9 16-9M20 21v16M12 8l16 9v15"/></svg><div><h1>SILICONDEVINE<span>.js</span></h1><p>神经网络可视化框架</p></div></div><div class="version">FRAMEWORK / 0.1.0</div></header>
<main><aside><section><h2>模型来源</h2><label for="example">载入示例</label><select id="example"><option value="declarative">声明式 · 全连接网络</option><option value="mlp">PyTorch · MLP</option><option value="conv2d">PyTorch · 残差卷积</option><option value="conv3d">PyTorch · 3D 卷积</option><option value="attention">PyTorch · 注意力</option></select><div class="file-row"><label class="button" for="file">导入模型 JSON</label><input type="file" id="file" accept=".json,application/json" hidden><button id="download">导出</button></div><p class="muted" id="source">固定种子演示数值</p></section>
<section id="params"><h2>张量维度</h2><div class="fields"><label>Batch<input id="batch" type="number" min="1" max="4" value="2"></label><label>输入通道<input id="channels" type="number" min="1" max="32" value="8"></label><label>隐藏通道<input id="hidden" type="number" min="1" max="32" value="12"></label><label>输出通道<input id="outputs" type="number" min="1" max="32" value="4"></label></div><button id="rebuild" class="wide">更新模型</button></section>
<section class="index"><h2>计算图索引 <span id="node-count"></span></h2><p class="muted">选择模块聚焦，悬停区域接管动效。</p><nav id="outline" aria-label="计算模块"></nav></section>
<details><summary>声明式接入</summary><pre>defineModel('My network')
  .input('x', [2, 8])
  .linear('encoder', 12)
  .activation('gate', 'gelu')
  .linear('output', 4)
  .build()</pre><p class="muted">任意分支可通过 operation() 或标准图 JSON 接入。浏览器不执行 Python。</p></details></aside>
<div class="workspace"><div class="toolbar"><div><strong id="model-name">全连接网络</strong><span id="active">等待模型</span></div><div class="controls"><button id="play">暂停</button><button id="fit">总览</button><button id="zoom-in" aria-label="放大">＋</button><button id="zoom-out" aria-label="缩小">−</button><label class="sr" for="labels">全息标牌显示</label><select id="labels"><option value="auto">按需标牌</option><option value="all">全部标牌</option><option value="none">隐藏标牌</option></select><button id="fullscreen">全屏</button></div></div>
<div id="stage" tabindex="0"></div><div class="status"><span id="stats"></span><div><button id="prev">上一图段</button><button id="next">下一图段</button></div></div>
<div class="legend" aria-label="数值颜色图例"><span><i style="--c:#ffc47e"></i>负值</span><span><i style="--c:#879bae"></i>零</span><span><i style="--c:#68e6d2"></i>正值</span><span><i style="--c:#effaff"></i>当前计算</span><span><i style="--c:#4b6574"></i>未知数值</span><small>按张量归一化 · 对称 asinh</small></div>
<section class="inspector"><div><h2 id="detail-title">完整权重连接，逐次计算</h2><p id="detail">晶体代表具有明确坐标的张量元素。权重线保持可见，动效沿计算过程平滑流动。复杂图只激活当前区域；点击聚焦，拖动旋转。</p></div><div><code id="formula">y = Wx + b</code><p id="shape"></p><p id="support" class="muted"></p></div></section>
<div id="error" role="alert" hidden></div></div></main>`;
const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
let model: Model, pageStart = 0, request = 0;
const stage = el('stage');
function detail(n?: Operation) { if (!n) {
    el('detail-title').textContent = '整体计算图';
    el('formula').textContent = '';
    el('shape').textContent = model.inputs.join(' / ') + ' → ' + model.outputs.join(' / ');
    el('support').textContent = model.notes?.[0] ?? '';
    el('detail').textContent = '水晶块保留真实坐标；参数与计算主干对齐。拖动旋转，滚轮缩放；悬停连续区域立即切换局部计算动效。';
    return;
} const v = operatorVisual(n.op); el('detail-title').textContent = `${v.label} · ${n.name}`; el('formula').textContent = v.formula; el('detail').textContent = `输入 ${n.inputs.join('、')} → 输出 ${n.outputs.join('、')}。来源：${n.source ?? '声明式定义'}。`; el('shape').textContent = n.outputs.map(id => { const t = model.tensors.find(t => t.id === id)!; return `${id} [${t.shape.join(' × ')}]`; }).join(' / '); el('support').textContent = v.detail === 'exact' ? '按算子定义连接可见窗口中的实际元素。窗口之外的坐标仍保留在模型元数据中。' : '此算子保留实际输入输出与拓扑，内部数学动效尚未实现；可注册专用插件。'; for (const b of el('outline').querySelectorAll('button'))
    b.setAttribute('aria-current', String(b.dataset.id === n.id)); }
const viewer = new SiliconDevineViewer(stage, { onSelect: detail, onStats: s => { pageStart = s.pageStart; el('active').textContent = s.active ? `计算区域 · ${s.active}` : ''; el('stats').textContent = `${s.cells} 个可见元素 · ${s.connections} 条联系 · ${s.drawCalls} 次绘制${s.partialTensors ? ` · ${s.partialTensors} 个坐标窗口` : ''}${s.unknownTensors ? ` · ${s.unknownTensors} 个未导出数值的张量` : ''}`; el<HTMLButtonElement>('prev').disabled = s.pageStart === 0; el<HTMLButtonElement>('next').disabled = s.pageStart + s.visibleNodes >= s.totalNodes; } });
// Debug information is read-only; a host application can use getStats() directly.
Object.assign(window, { siliconDevine: { viewer, get model() { return model; } } });
function error(e: unknown) { el('error').hidden = false; el('error').textContent = e instanceof Error ? e.message : String(e); }
function load(m: unknown) { const valid = (m as {
    format?: string;
})?.format === 'silicondevine-recipe' ? compileDescription(m) : validateModel(m); viewer.load(valid); model = valid; el('error').hidden = true; el('model-name').textContent = model.name; el('node-count').textContent = String(model.nodes.length); el('source').textContent = `${model.producer?.backend ?? 'JSON'} · ${model.producer?.version ?? 'v1 图格式'}`; el('outline').replaceChildren(); for (const n of model.nodes) {
    const b = document.createElement('button');
    b.dataset.id = n.id;
    b.textContent = `${operatorVisual(n.op).label} · ${n.name}`;
    b.title = n.source ?? n.op;
    b.onclick = () => viewer.focus(n.id);
    el('outline').append(b);
} detail(); }
function generated() { const read = (id: string, max: number) => { const v = Number(el<HTMLInputElement>(id).value); if (!Number.isInteger(v) || v < 1 || v > max)
    throw Error(`请输入 1–${max} 之间的整数。`); return v; }; return defineModel('全连接 · Crystal MLP').input('x', [read('batch', 4), read('channels', 32)]).linear('hidden', read('hidden', 32)).activation('gelu', 'gelu').linear('output', read('outputs', 32)).build(); }
el('rebuild').onclick = () => { try {
    load(generated());
}
catch (e) {
    error(e);
} };
el<HTMLSelectElement>('example').onchange = async (e) => { const id = ++request; try {
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
} };
el<HTMLInputElement>('file').onchange = async (e) => { const file = (e.target as HTMLInputElement).files?.[0]; if (!file)
    return; try {
    if (file.size > 32 * 1024 * 1024)
        throw Error('JSON 超过 32 MB，请减少数值导出窗口。');
    ++request;
    load(JSON.parse(await file.text()));
    el('params').hidden = true;
}
catch (e) {
    error(e);
}
finally {
    el<HTMLInputElement>('file').value = '';
} };
el('download').onclick = () => { const url = URL.createObjectURL(new Blob([JSON.stringify(model, null, 2)], { type: 'application/json' })); const a = document.createElement('a'); a.href = url; a.download = 'model.sd.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); };
el('play').onclick = () => { viewer.setPlaying(!viewer.isPlaying); el('play').textContent = viewer.isPlaying ? '暂停' : '播放'; };
el('play').textContent = viewer.isPlaying ? '暂停' : '播放';
el('fit').onclick = () => viewer.clearFocus();
el('zoom-in').onclick = () => viewer.zoom(.8);
el('zoom-out').onclick = () => viewer.zoom(1.25);
el('fullscreen').onclick = () => void viewer.fullscreen().catch(error);
el<HTMLSelectElement>('labels').onchange = e => viewer.setLabels((e.target as HTMLSelectElement).value as 'auto' | 'all' | 'none');
el('prev').onclick = () => viewer.setPage(Math.max(0, pageStart - 32));
el('next').onclick = () => viewer.setPage(pageStart + 32);
stage.addEventListener('keydown', e => { if (e.key === '+' || e.key === '=')
    viewer.zoom(.8); if (e.key === '-')
    viewer.zoom(1.25); if (e.key === 'Home')
    viewer.clearFocus(); if (e.key === ' ') {
    e.preventDefault();
    el('play').click();
} });
window.addEventListener('beforeunload', () => viewer.dispose(), { once: true });
load(generated());
