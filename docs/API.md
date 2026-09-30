# 图格式与 API

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

符号形状可用字符串传输，查看器保留结构标识；没有具体尺寸时不会生成声称准确的晶体网格。PyTorch 导出器当前取得的是示例执行的具体尺寸，不导出完整符号约束。

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

0.1 的插件范围是依赖、颜色、公式、标量曲线；不是任意 Three.js 场景插件。对特殊网络需要新增渲染原语时，应通过下一版显式接口扩展，避免业务方访问查看器内部状态。
