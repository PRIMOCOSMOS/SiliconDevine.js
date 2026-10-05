# 用基础算子看复杂模型

在大模型中点击 Attention、FFN 或专家模块，选择 **展开原子组合**。默认显示完整计算骨架；上方可选择功能区域，或开启演示巡航。悬停会优先演示当前计算区域。

所有组合使用同一套基础算子：Linear 的权重连线、矩阵乘法的标量依赖、激活、Softmax、归一化、逐元素乘法与残差。新增算子通过 `registerOperator` 注册后，也会进入组合清单。

大模型总览自动轮流预览一个功能模块的原子组合。预览与展开视图复用张量坐标布局、水晶元素、权重色谱和计算动效。重复层保留数量和模板关系，每个实例的参数仍然独立。一次只创建一个预览，离开时释放其图形资源。

## 数据来源

- 本地 PyTorch 捕获：组合来自实际执行图，保留张量、参数、输入输出与源码信息。
- DeepSeek、GLM、MiniMax 的配置结构：总览保留真实配置尺寸；原子计算使用已核对公式的小规模 PyTorch 实算教学图。它们使用固定种子的未训练参数，并非完整模型权重或实际 Token 推理结果。
- 未覆盖的自定义结构：保留计算边界。框架不会擅自把它解释成全连接层。

## 接口

```js
import { atomicComposition, compositionCatalog } from 'silicondevine.js';
const computation = atomicComposition(capturedModel);
// computation.atoms: 基础算子、参数引用、数据依赖、源码与支持状态
// computation.boundaries: 尚无内部计算可视化的节点
const catalog = compositionCatalog(largeModel);
// catalog.units: 功能单元、子单元引用、重复次数及原子组合
```

`viewer.showMechanism()` 打开完整原子组合；`viewer.setComputationStage(i)` 聚焦第 i 个功能区域，传入 `-1` 恢复全部。`viewer.showStructure()` 返回总览。

预览采用有限坐标窗口和最多 8,192 条联系，完整图仍通过展开与分页访问。当前机制演示覆盖随框架提供的模板；其他模型需要真实捕获图，或显式提供自己的机制图与引用。
