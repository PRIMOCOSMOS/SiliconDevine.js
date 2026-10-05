# 大模型怎么查看

双击 `D:\SiliconDevine.js\SiliconDevine.exe`，在「从示例开始」选择 **DeepSeek-V3、GLM-4.5 或 MiniMax-M1**，点击「进入模型空间」。网页的「模型库 → 大模型结构」也能直接切换。

这些示例只读取本地代码对应的配置，不下载权重、不占用显存，也不执行大模型推理。

## 看图

- **堆叠框**：重复层。名称中的 ×58 表示 58 个独立层；点击查看一个结构模板。
- **圆环与汇聚线**：注意力。展开可查看 QKV、位置编码、缓存与输出投影。
- **扇形分支**：专家路由或专家组。专家总数、每个 Token 激活数写在模块名称中；图形数量只表示分组。
- **回环**：Lightning Attention 的状态更新。
- **薄的刻面矩阵**：模型参数；较厚的水晶：数据张量；缓存单独标识。

玻璃分区概括整个逻辑张量，旁边保留完整形状。B 是 Batch，T 是本次序列长度，S 是含历史缓存的长度。结构模式没有数值颜色；动画表示通路，不表示模型实际选中了哪个专家。

鼠标悬停会立即切到对应模块的通路；移开后恢复自动轮播。点击模块展开，点击「上一级」返回。原来的缩放、全屏、暂停和标牌显示设置都可继续使用。

## 已核对的范围

| 示例 | 主干 | 重点 |
|---|---|---|
| DeepSeek-V3 | 61 层；前 3 层稠密，其余 58 层 MoE | MLA 压缩 KV、独立位置分量、256 专家中选 8、共享专家 |
| GLM-4.5 | 92 层；前 3 层稠密，其余 89 层 MoE | 96 个 Query 头 / 8 个 KV 头、QK Norm、160 专家中选 8 |
| MiniMax-M1-80k | 80 层；7 层 Lightning + 1 层全注意力，重复 10 次 | 递归 KV 状态、输出门、32 专家中选 2、源码中的缩放残差 |

显示的参数量是推理主干的逻辑参数统计，不包含 MTP、量化尺度、缓存和优化器状态。它不等于模型文件大小，也不表示单个 Token 激活的参数量。

来源、提交号、配置版本和文件校验值保存在 `examples/upstream/large_models/SOURCE.json`。此版覆盖表中的具体实现；其他同名系列版本需要另行核对适配。

## 在 IDE 中接入自己的配置

新建 Python 文件，写一个返回结构图的工厂函数：

```python
import json
from silicondevine import architecture_from_config

def build_model():
    with open(r"D:\MyModels\config.json", encoding="utf-8") as file:
        config = json.load(file)
    return architecture_from_config(config, "glm45")
```

在启动器选择这个文件，工厂函数填 `build_model`。支持的适配器名称是 `deepseek_v3`、`glm45`、`minimax_m1`。配置必须对应适配器所支持的源码结构；任意修改 forward 后，需要导出实际执行图或编写新适配器。

启动器会监视邻近 Python 文件。修改配置 JSON 后，请重新保存这个 Python 文件以刷新页面。

已在 Python 中构造模型时，也可使用 `export_architecture(model)`。GLM-4.5 的原生类支持在 PyTorch `meta` 设备上创建，无需分配实际参数存储：

```python
import torch
from transformers import Glm4MoeConfig, Glm4MoeForCausalLM
from silicondevine import export_architecture

def build_model():
    config = Glm4MoeConfig.from_json_file(r"D:\MyModels\config.json")
    with torch.device("meta"):
        model = Glm4MoeForCausalLM(config)
    return export_architecture(model)
```

这里读取模型配置并统计实际模块参数，不运行 forward。想看真实激活数值时，继续使用原来的 `return model, inputs`，先选能在本机运行的子模块。

## 给新功能模块编写模板

Web API 新增 `defineArchitecture(name, scopes, entry)`。每个 scope 使用原有的 `nodes / tensors / inputs / outputs` 图格式；张量加 `representation: 'aggregate'`，保留真实逻辑形状，不填写模拟数值。节点使用 `op: 'structure'`。

节点的 `attrs.scopeRef` 指向子模板；`attrs.repeat` 表示独立实例数量。`attrs.kind` 可选 `repeat`、`attention`、`router`、`experts`、`recurrent`、`ffn`、`gate`、`norm`、`rope`、`merge`、`projection`、`embedding`。它决定功能图形及结构色，不修改数值模式的颜色。

框架校验引用、循环、重复次数和总规模。参数量按模板权重形状及重复次数计算；共享参数必须只声明一次，不能用 repeat 表示。每次只构建当前 scope 的几何。`aggregatePartition(tensor, tile)` 可读取每个分区所覆盖的逻辑范围，数字范围采用左闭右开区间。

可参考 `python/silicondevine/architecture.py` 中已实现的模板组合。注册模板属于显式适配；框架不会仅凭类名就认定自定义代码实现了某种算法。
