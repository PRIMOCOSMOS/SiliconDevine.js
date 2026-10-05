# 从源码展开模型

[使用说明](USER_GUIDE.md) · [文档目录](INDEX.md)


打开 DeepSeek、GLM 或 MiniMax，点击重复层，进入一个代表性 Decoder 的执行图。继续点击 Attention、MoE、专家或线性层，逐层查看组成。右上角可返回；“算子”视图可查看当前模块的原始计算节点。

展开后的层来自同一次前向计算。父模块的输出与子模块输出使用同一个张量 ID；注意力输出直接连接残差，再进入归一化和前馈网络。基础层与大模型共用水晶张量、权重矩阵、连续连线高亮和数值色谱。以前附加在大模块上的小型演示图已移除。

## 来源和范围

- DeepSeek-V3、GLM-4.5：执行本机 Transformers 4.57.1 对应 Decoder 类的原始构造函数和 forward，采用 eager 注意力。DeepSeek 使用该实现的展开式 MLA；根级全尺寸概要仍来自项目原先核对的配置。
- MiniMax-M1：使用 examples/upstream/large_models 中固定版本的原始类代码。CPU 全注意力使用上游 eager 类；Lightning 执行原始 decode 递推。仅适配输入接口与 einops 的坐标重排。
- 为限制显示和计算开销，示例缩小隐藏维度、头数和专家数，使用固定种子的未训练权重。根级概要保留原配置层数；执行图展示一个代表单元，并非完整检查点推理。
- MoE 路由随输入变化。执行图记录本次输入经过的路径；未执行的专家分支不会被编造。修改输入后重新捕获即可查看新的路径。

每个算子记录 ATen 调用、模块路径和代码调用栈；来源面板保留实现版本及文件哈希。归一化可作为一个基础模块显示，并保留其捕获到的内部运算记录。

## 接入自己的代码

```python
from silicondevine import export_model

graph = export_model(
    model, (example_input,),
    "my-model.sd.json",
    backend="execution",
    include_values=True,
)
```

将 JSON 导入工作台。IDE 热更新也可选择 execution 捕获接口；启动器的高级设置已加入此选项。fx 和 export 接口仍保留。

execution 使用 PyTorch make_fx 记录真实执行路径，适合带实际路由判断的小规模模型或选定子模块。请先用小输入验证；传入巨型实例会真正运行该实例，框架不会自动缩小任意模型。

## 重新生成内置示例

在项目目录设置 PYTHONPATH=python 后，先运行 examples/source_models.py，再运行 examples/large_models.py。需要本地 PyTorch 与 Transformers；MiniMax 源文件随项目提供。

JavaScript 中，atomicComposition(model) 返回基础算子清单，compositionCatalog(model) 返回子模块关系与重复次数。未知运算保留明确的边界；支持自定义 registerOperator 插件。

## 功能视图的组织

默认注意力视图将 Q/K/V 投影后的分头、共享头对齐、RoPE、缩放掩码与 Softmax、合并多头按用途组织。原始算子保存在 attrs.children 中，边界输出仍使用捕获张量 ID。点击功能段可查看内部步骤；切换算子视图可核查完整执行顺序。

坐标操作用晶体重排和展开展示，计算操作保留数值依赖线。Q/K/V 采用稳定并行位置，注意力分数、概率、Value 汇聚和输出投影回到竖直主干。
