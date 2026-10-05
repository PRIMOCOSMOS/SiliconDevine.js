# 图格式与 API

[使用说明](USER_GUIDE.md) · [文档目录](INDEX.md)


## IR v1

```ts
interface Model {
  format: 'silicondevine'; version: 1; name: string;
  tensors: Tensor[]; nodes: Operation[];
  inputs: string[]; outputs: string[];
}
interface Tensor {
  id: string;
  shape: (number | string)[];
  dtype: string;
  role: 'input' | 'activation' | 'parameter' | 'buffer' | 'constant';
  data?: {offset: number; values: (number | null)[]};
}
interface Operation {
  id: string; name: string; op: string;
  inputs: string[]; outputs: string[];
  parameters?: Record<string, string>;
  attrs?: Record<string, unknown>;
  group?: string; source?: string;
}
```

张量是逻辑行优先顺序；`stride` 保留来源信息，但 `data` 总是逻辑 flatten 后的数据。数值窗口 `[offset, offset + values.length)` 不等于整个张量。窗口之外未知；null 表示 NaN/Inf 等无法以 JSON 数字表达的值。框架不把未知值当成零，也不生成伪权重。`stats` 只统计已保存窗口，不声称整张量的统计量。

符号形状可用字符串传输，查看器保留结构标识；没有具体尺寸时不会生成声称准确的晶体网格。PyTorch 导出器当前取得的是示例执行的具体尺寸，export 后端通过 constraints 字段保留范围约束（描述性字符串）；dynamic_shapes 由 PyTorch 解释。

`inputs`、`outputs` 和 `parameters` 引用全局 Tensor ID。一个张量最多由一个算子产生；复用相同参数 ID 表示共享权重。节点不必预排序，但必须是 DAG。不允许缺失引用、重复 ID 或循环。

## 查看器

| 方法 | 用途 |
| --- | --- |
| `load(model)` | 校验并加载 IR；无效模型在替换旧模型之前报错 |
| `focus(nodeId)` | 定位到模块输入、输出与参数；跨图段会自动切换 |
| `clearFocus()` / `fit()` | 返回当前图段总览 / 重新适配镜头 |
| `setPage(start)` | 查看按拓扑排序的下一个图段，边界张量保留 |
| `setPlaying(boolean)` | 暂停或继续计算动效 |
| `setLabels('auto'|'all'|'none')` | 标牌按需、全部或隐藏 |
| `zoom(factor)` | 相对缩放；小于 1 放大 |
| `fullscreen()` | 全屏容器；需用户交互触发 |
| `getStats()` | 可见晶体/连线/绘制次数、窗口数量、激活算子等 |
| `dispose()` | 释放事件、观察器、GPU 资源和 canvas |

自动演示按算子轮转；悬停优先于固定选择，固定选择优先于自动演示。依赖层之间的区域在 Y 轴相接，同层并行算子用相邻 X 区域分隔。小型 MLP 保留所有可见坐标之间的权重线；复杂图只生成当前算子的连线。

## 数学插件

```ts
import {registerOperator} from 'silicondevine.js';

registerOperator('square', {
  label: '逐元素平方', color: '#b6a0e6', formula: 'y = x²',
  detail: 'exact',
  dependencies(node, output, index, tensors) {
    return [{tensor: node.inputs[0], index}];
  },
  curve: x => x*x,
});
```

`dependencies` 必须返回实际标量的 ID 与 flatten 坐标。`weight` 可给出实际乘数，配合 `parameter` / `parameterIndex` 让权重晶体与连线同步高亮。未提供数值时不应捏造乘数。调用方负责插件的数学正确性；内置插件由真实 PyTorch 输出验证。插件在 `viewer.load()` 之前注册。

当前插件范围是依赖、颜色、公式、标量曲线；不是任意 Three.js 场景插件。对特殊网络需要新增渲染原语时，应通过下一版显式接口扩展，避免业务方访问查看器内部状态。


## 更新与数据窗口

| 方法 | 功能 |
| --- | --- |
| update(model) | 更新图并保持镜头与旋转目标 |
| setDataProvider(provider) | 注入按需数值源；签名 (tensorId, indices, signal) => Promise<values> |
| setTensorWindow(id, origin) | 各维真实坐标偏移；异步读取数值；Linear 输入与 W 列窗口联动 |
| getModules() / showModule(path) | 查询/进入模块层级；空路径返回完整图 |
| setSpeed(0.1…4) | 动效速度 |
| setProgress(0…1) | 暂停并定位当前区域的计算进度 |

getStats() 新增 weight {tensor,index,coordinates,value} 与 sample {input,output,index}。同一 parameterIndex 使矩阵水晶与各 Batch 对应的权重线同步高亮。标牌为侧旁世界坐标网格，不使用屏幕浮窗或引线。

IR Tensor 新增可选 samples（坐标到数值的映射）和 spatialRank（三维体布局提示）。null/缺失值不变成零。Model 可选 constraints 与 live 元数据。LiveConnection 的完整示例见 LIVE.md。

## 层级与计算机制

- getNavigation() 返回 scope、representation 与当前图段 nodes。
- setRepresentation('architecture' | 'operators') 在功能架构和原始算子图间切换；parentModule() 返回上一级。
- moduleView(model, scope?, detail?) 可独立生成模块视图；功能架构中的 layout 单元保存原始 children，复合依赖保持精确坐标。
- Model.modules 记录原生模块 path/type；Tensor.axes 为 B/H/Q/K 等轴标签。
- Tensor.specialValues 记录 flatten index 对应的 '-inf'、'+inf'、'nan'。JSON data 中相应元素为 null，通过 valueAt 获取完整语义；缺失值依然未知，不能当零。
- getStats().mechanism 提供当前 kind、row、实际输出 values、visibleItems，以及可用的 mean/rms/sum。插值动画不改变这些实际输出数值。

架构分组和动画资源预算不改变模型快照。过大或训练 dropout 的注意力保留融合边界，并在 attrs.boundary_reason 中解释。

## 代码依据与局部动效

Model 可选 provenance、functionalUnits；modules 条目增加 qualified 和 code（file/line/sha256）。Operation.attrs 的 projectionRoles、attention、attentionRole、residualInput、sourceFunction、codeTrace 来自捕获图和保守模式识别。Tensor.semantic 标出Q、Kᵀ、V等用途。

getNavigation().function 表示当前捕获函数体；parentModule() 优先退出函数体。getStats() 增加 receptiveField（输出坐标/可见与总采样数/核尺寸）、attention（Query/头索引/真实概率/输出/贡献数）和 residuals。每个浮点值仍源于捕获数据；单位连接不代表未捕获的训练权重。
