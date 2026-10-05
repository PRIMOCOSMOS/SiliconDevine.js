# SiliconDevine.js

将 PyTorch 模型代码转换为可交互的三维计算图。保留真实张量坐标、数值配色、权重连线与分层计算动效。

**开始使用：[统一使用说明](docs/USER_GUIDE.md)** · [浏览器阅读版](docs/QUICKSTART.html) · [文档目录](docs/INDEX.md)

在本项目文件夹中双击 `SiliconDevine.exe`，选择模型文件和 Python 环境，点击「进入模型空间」。保存代码后网页自动更新。EXE 需要同目录资源；日常使用无需 Node。

模型文件最小示例：

```python
import torch
from torch import nn

def build_model():
    model = nn.Sequential(nn.Linear(16, 8), nn.GELU())
    return model, (torch.randn(2, 16),)
```

首次在模型环境安装接口：

```powershell
python -m pip install -e D:\SiliconDevine.js\python
```

普通模块先用 `fx`；函数化 ATen 图用 `export`；实际分支执行用 `execution`。配置、示例、导出、IDE 连接、大模型边界及故障处理均见使用说明。

## 开发

```powershell
npm install
npm run dev
npm run build
```

前端构建输出为 `dist` 与 `demo-dist`。Python 接口在 `python/silicondevine`，渲染器在 `src/render`，图格式与算子在 `src/core`。版本 0.7.1；本地已验证环境：Node 24、PyTorch 2.9 CPU。
