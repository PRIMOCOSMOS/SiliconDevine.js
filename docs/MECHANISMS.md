# 看清模型怎样计算

[使用说明](USER_GUIDE.md) · [文档目录](INDEX.md)


## 在应用里打开

重新运行 `SiliconDevine.exe`。启动器和网页模型库都加入了「动态卷积核」「条件注意力」。这两个示例会实际运行 PyTorch，显示捕获到的输入、参数、中间张量和输出。

内置 DeepSeek-V3、GLM-4.5、MiniMax-M1 默认进入源码执行组合，使用原始 forward 的缩小配置捕获图。操作与适配边界见 [源码组合](ATOMIC-COMPOSITION.md)。

本文件下面的 `attach_mechanisms` 是另一项显式接口：给配置结构附加固定维度的独立教学机制。它保留旧版按阶段查看与巡航功能。独立机制用于演示算法，不自动证明给定模型的自定义 forward 与之相同；实际代码验证使用 execution 捕获。

## 动效与结构视图

功能架构聚合坐标组织、RoPE、注意力分配和门控，并保留原始 children 与边界张量 ID。点击功能段查看组成；原始算子视图可逐项检查。重排以晶体坐标移动表示，广播展示共享源坐标向目标位置展开。它们与乘加权重线使用不同的动效。

复杂图只生成活动区域的数学连接。侧旁三维标注使用固定世界坐标，旋转和缩放不会触发位移、换侧或补偿缩放，只有文字平面转向镜头。按需模式每个模块最多显示一张标牌，并控制标牌间重叠；全部模式保留所有标牌。标注不再用模型投影包围盒判定遮挡，避免错误隐藏。默认主干标注不随计算轮播跳换；悬停标注或结构时按所属模块联动高亮。聚焦会一次性将模块主标牌纳入取景范围。

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
