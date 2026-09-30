# SiliconDevine.js

从 PRIMOCOSMOS 工作站提取的独立 Web 神经网络可视化框架，当前版本 **0.1.0**。

保留水晶张量、数值着色、完整的 MLP 权重线、连续计算高亮、卷积感受野、三维体数据、主干层级和嵌入模型空间的全息标牌。框架不依赖 React，也不依赖原个人主页。

## 立即查看

交付目录已经包含构建好的 `demo-dist`。无需安装 Node 依赖：

```powershell
cd D:\SiliconDevine.js
python serve.py
```

打开 **http://127.0.0.1:5181**。也可以双击 `start-demo.cmd`。默认仅监听本机。

提供声明式 MLP、PyTorch MLP、残差卷积、3D 卷积、注意力五种入口。JSON 导入在浏览器内完成，不上传模型。支持拖动旋转、滚轮缩放、悬停计算区域、点击/索引聚焦、暂停和全屏。

## 开发与构建

需要 Node.js 20.19+ 或 22.12+；导出工具需要 Python 3.10+ 与 PyTorch 2.6+。本次实际验证环境为 Node 24.12、PyTorch 2.9 CPU。

```powershell
npm install
npm run dev                 # http://127.0.0.1:5180
npm run build               # dist/ 库与 demo-dist/ 独立演示
python -m pip install -e ./python
npm run examples            # 重新运行 PyTorch 模型，产生示例 JSON
npm test
```

## 接入 PyTorch 模型

```python
import torch
from torch import nn
from silicondevine import export_model

model = nn.Sequential(nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 4))
example = torch.randn(2, 16)

export_model(
    model, (example,), "encoder.sd.json",
    backend="fx",           # 保留 nn.Module 边界
    include_values=True,    # 默认 False；默认只导出图和形状
    value_limit=4096,        # 每个张量最多保存多少连续元素
    total_value_limit=500000,
)
```

`backend="export"` 使用 `torch.export` 取得函数化的 ATen 图，适合把模块进一步展开为算子。两种后端都实际执行示例输入以取得形状与可选数值。导出器对模型与输入取副本，使用 eval 模式，并恢复 RNG 状态；原模型的训练状态和参数不改动。

**不会假装能无损识别任意 Python。** 数据依赖分支、无法跟踪的自定义算子和特殊张量都有边界。导出失败会报错；可导出但没有数学插件的算子保留真实拓扑与输入输出，并标明仅支持边界。详见 [支持范围](docs/SUPPORT.md)。

## 用描述性代码定义模型

```ts
import {defineModel, SiliconDevineViewer} from 'silicondevine.js';

const graph = defineModel('Encoder', 42)
  .input('x', [2, 16])
  .linear('encoder', 8)
  .activation('gate', 'gelu')
  .linear('latent', 4)
  .build();

const viewer = new SiliconDevineViewer(document.querySelector('#stage')!, {
  maxCells: 8192,
  cellsPerTensor: 128,
  maxConnections: 12000,
  maxNodes: 32,
  pixelRatio: 1.5,
  onSelect: node => console.log(node),
});
viewer.load(graph);
// 卸载页面组件时：viewer.dispose();
```

容器需有明确高度，例如 `#stage { height: 650px; }`。Three.js 是 peer dependency，与宿主共用一份运行时。尚未发布到 npm，可通过 `npm install ./SiliconDevine.js` 或 `npm pack` 本地接入。

还可直接导入 [examples/recipe.json](examples/recipe.json)，用纯数据描述顺序层；没有 `eval()`，不执行代码字符串。分支、共享张量、多输出和自定义算子使用完整 IR，或 `ModelBuilder.tensor().operation()`。

## 框架结构

| 部分 | 责任 |
| --- | --- |
| `python/silicondevine` | FX / torch.export → 真实执行图与张量窗口 |
| `src/core/model.ts` | 版本化 IR、校验、拓扑、坐标与数值访问 |
| `src/core/builder.ts` | 类型化描述式构建器与可复现示例数值 |
| `src/core/operators.ts` | 可扩展的标量依赖、公式、激活曲线 |
| `src/core/layout.ts` | 同质平面、依赖层级、参数对齐、真实体数据 |
| `src/render` | Three.js 实例化晶体、GPU 连线动效、标牌与交互 |
| `demo` | 独立模型浏览器；可以换成 React/Vue 等宿主 |

进一步阅读：[图格式与 API](docs/API.md) · [架构与性能](docs/ARCHITECTURE.md) · [支持范围](docs/SUPPORT.md) · [验证记录](docs/VALIDATION.md)

本次没有更改个人主页既有渲染器。框架和网站可以后续通过统一 IR 适配器接入，避免把论文专用逻辑重新写入通用内核。
