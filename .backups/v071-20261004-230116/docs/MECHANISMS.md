# 看清模型怎样计算

## 在应用里打开

重新运行 `SiliconDevine.exe`。启动器和网页模型库都加入了「动态卷积核」「条件注意力」。这两个示例会实际运行 PyTorch，显示捕获到的输入、参数、中间张量和输出。

DeepSeek-V3、GLM-4.5、MiniMax-M1 保留完整配置结构。点击模块进入注意力、路由或专家后，按「查看计算原理」。这里直接使用工作站的数值张量、权重连线、激活曲线、感受野和计算高亮。

复杂计算按阶段展开：从上方选择「投影」「注意力分配」「输出」等阶段；「全部步骤」显示完整计算图。阶段切换不会再加一层目录。「返回结构」回到当前功能模板，「上一级」返回所属结构。点击末端结构算子也能直接打开对应的原理演示。

「演示巡航」每 12 秒切换一个计算阶段。鼠标悬停时停在当前阶段并演示该区域；拖动或滚轮缩放会停止巡航和镜头过渡。系统的减少动态效果设置会关闭镜头动画。

## 演示数据代表什么

大模型结构页显示官方配置尺寸、重复层数和逻辑参数量。计算页使用固定种子、缩小维度的未训练 PyTorch 张量；顶部持续显示这一状态。计算页中的水晶元素和依赖线对应实算坐标，颜色表示数值。它不包含已训练大模型的权重，也不代表它的推理结果。

- MLA：单头吸收式计算；低秩 Query、压缩 KV、位置与内容分数、因果 Softmax、潜变量聚合、Value 解压、输出投影。长上下文 mscale 和多头复制留在结构说明中。
- GQA：两个 Query 头共享一组 Key/Value；显示投影、部分位置旋转、保留不旋转分量、概率和加权结果。GLM 示例包含 Q/K RMSNorm 和配置中的投影偏置。
- MoE：四个独立专家、Top-2；显示条件路由、选择校正、分组筛选、独立 SwiGLU、加权汇聚及共享分支。为方便比较，演示页同时计算小型专家的结果；全尺寸模型仍按实际稀疏分派运行。
- Lightning：一个头、三个连续 Token；逐步显示外积写入、衰减状态、Query 读取和输出门。使用 decode 递推等价式，未运行 GPU 分块 kernel。
- RMSNorm、RoPE、门控、线性投影、残差和查表也有数值视图。

源码来源和固定版本在 `examples/upstream/large_models/SOURCE.json`。测试直接调用已保存的上游 MLA、路由、Lightning 函数，以及 Transformers 的 GLM 注意力实现，比对小型演示输出。

## 在 IDE 使用

```python
import torch
from silicondevine import DynamicConv2d, ConditionalAttention, show

model = DynamicConv2d(4, 8, kernel_size=3, experts=4).eval()
show(model, (torch.randn(2, 4, 12, 12),))
```

动态卷积以输入的全局平均池化特征生成 Softmax 系数，混合可学习核库，再对每个 Batch 项执行自己的卷积。支持 stride、padding、dilation、groups；当前接口为二维、每个样本一套核、不含偏置。用默认 FX 导出可保留「核混合」「逐样本卷积」专用动效。任意自定义动态卷积变体需要由其实际计算图或插件说明。

```python
model = ConditionalAttention(dim=8, condition_dim=12, heads=2).eval()
x = torch.randn(1, 4, 8)
condition = torch.randn(1, 6, 12)
mask = torch.zeros(4, 6)  # 加性 mask；不可见位置填 -inf
show(model, (x, condition, mask), backend="export")
```

Query 来自 x，Key/Value 来自 condition。输入数量和序列长度可以不同。此接口提供条件交叉注意力；其他门控、调制或条件位置编码变体按模型自己的 PyTorch 图导出。

给已有大模型配置图附加计算演示：

```python
from silicondevine import architecture_from_config, attach_mechanisms
graph = attach_mechanisms(architecture_from_config(config, "deepseek_v3"))
```

JavaScript 框架接口：`viewer.showMechanism()`、`viewer.showStructure()`、`viewer.setComputationStage(index)`、`viewer.setTour(true)`。自定义架构可通过 `architecture.mechanisms` 提供经过验证的数值模型，并用 `mechanismRef` 关联；`defineArchitecture` 的第四个参数接收这些演示。
