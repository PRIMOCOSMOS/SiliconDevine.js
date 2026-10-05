# SiliconDevine.js 使用说明

在 IDE 中写普通 PyTorch 模型，给出一组示例输入，即可查看它的结构、张量和计算过程。推荐先用启动器连接代码；后续保存文件，网页会自动更新。

## 1. 准备环境

- 保留 `D:\SiliconDevine.js` 整个文件夹，EXE 需要其中的网页资源与 Python 工具。
- 模型运行环境需要 Python 3.10+、PyTorch 2.6+ 和模型自身依赖。本机已验证环境为 `D:\Anaconda3\python.exe`，PyTorch 2.9 CPU。
- 使用已打包应用无需 Node.js。首次在**模型所用的 Python 环境**安装本地接口：

```powershell
D:\Anaconda3\python.exe -m pip install -e D:\SiliconDevine.js\python
```

若使用虚拟环境，将命令开头换成该环境的 `python.exe`。Transformers 示例额外需要 `transformers==4.57.1`；普通 MLP/CNN 无需此依赖。

## 2. 推荐：写模型，保存即更新

新建 `D:\MyModels\my_model.py`，复制以下完整示例：

```python
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 24), nn.GELU(), nn.Linear(24, 8)
        )

    def forward(self, x):
        return self.layers(x)

def build_model():
    torch.manual_seed(42)
    model = Encoder().eval()
    x = torch.randn(2, 16)  # Batch=2，每项16个特征
    return model, (x,)     # 一个输入也要写成元组，注意逗号
```

1. 双击 `D:\SiliconDevine.js\SiliconDevine.exe`。
2. 选择刚保存的文件，以及安装了模型依赖的 `python.exe`。
3. 高级设置：工厂函数填 `build_model`；这个示例选择 `fx`。
4. 点击「进入模型空间」。浏览器自动打开后，修改代码并保存即可刷新。

框架运行一次前向计算，记录本次输入对应的形状与数值。已有权重可在 `build_model()` 返回前通过 `load_state_dict()` 加载。工厂函数中不要启动训练、启动服务或调用 `show()`。

更换文件或环境前，先停止服务。捕获失败时保留上一份有效图，错误会显示在界面上；修正代码并再次保存即可。

### 多输入、关键字参数与捕获设置

`args` 对应 `forward` 的位置参数，`kwargs` 对应关键字参数。下面是工厂函数的另一种返回格式；`model`、`x`、`context`、`mask` 使用你实际创建的对象：

```python
return {
    "model": model,
    "args": (x, context),
    "kwargs": {"attention_mask": mask},
    "options": {
        "backend": "execution",
        "include_values": True,
        "value_limit": 4096,
        "snapshot_budget": 64 * 1024 * 1024,
    },
}
```

文件中的 `options.backend` 优先于启动器选项。输入尺寸、类型和设备必须与模型匹配；例如嵌入层通常接收整型 Token ID，卷积通常接收 `[B,C,H,W]`。

## 3. 选择哪种捕获接口

| 接口 | 适合的情况 | 要点 |
| --- | --- | --- |
| `fx` | 普通 MLP、CNN、自定义 `nn.Module` | 先从这里开始，保留模块边界；依赖输入数值的 Python 分支可能无法跟踪。 |
| `export` | 需要函数化 ATen 图、声明动态形状约束 | 使用 `torch.export`；不支持的控制流会报错。`dynamic_shapes` 只用于此接口。 |
| `execution` | MoE 等依赖实际路由的代码、选定 LLM 子模块 | 使用 `make_fx` 记录本次执行路径；输入改变后需重新捕获，不包含未执行分支。 |

三种接口都会实际执行模型。导出器复制模型与输入，以 eval 模式捕获，并恢复随机数状态；它展示前向计算，不展示训练反向传播。自定义 C++/CUDA 算子或特殊容器可能仍需适配，导出失败时先缩小到相关子模块排查。

## 4. 直接从正在写的 Python 程序打开

无需工厂函数时，在模型与示例输入准备好后调用：

```python
from silicondevine import show

show(model, (x,), backend="fx")
```

终端会打印本地网页地址，打开该地址即可。此调用默认阻塞，结束时按 Ctrl+C。`show()` 不监视文件；保存自动更新使用启动器或下一节的 `watch`。

需要自己控制生命周期：

```python
server = show(model, (x,), backend="execution", block=False)
try:
    server.update(model, (new_x,), backend="execution")
    server.wait()
finally:
    server.close()
```

不使用启动器也能监视文件：

```powershell
D:\Anaconda3\python.exe -m silicondevine.watch D:\MyModels\my_model.py --backend execution --static-dir D:\SiliconDevine.js\demo-dist
```

监视器同时检查模型目录中的 Python 文件。建议给模型建立独立目录；配置 JSON 改动后，重新保存入口 Python 文件触发更新。默认端口为 5182，被占用时添加 `--port 5183`。启动器会自动寻找空闲端口。

## 5. 导出 JSON，离线查看或分享

```python
from silicondevine import export_model

export_model(
    model, (x,), "my-model.sd.json",
    backend="execution",
    include_values=True,
    value_limit=4096,          # 单个张量保存的连续元素上限
    total_value_limit=500000,  # 本次JSON保存的数值总量上限
)
```

网页打开「模型库 → 导入模型 JSON」，选择文件。也可运行项目中的 `start-demo.cmd` 打开预构建网页；无需启动 IDE 服务。

`export_model()` 默认不包含数值，想看数值配色与计算结果应设置 `include_values=True`。导出的窗口之外只保留形状；离线 JSON 无法读取未保存的值。IDE 服务可按需读取快照中保留的坐标，默认快照预算 64 MiB。

## 6. 如何阅读和操作模型

| 操作 | 结果 |
| --- | --- |
| 拖动、滚轮或缩放按钮 | 旋转、缩放当前视图；适配按钮恢复当前图段全貌。 |
| 悬停模块区域 | 优先演示该区域的计算；移开后恢复自动轮播。 |
| 点击模块／层级索引 | 聚焦或进入子模块；「上一级」返回。 |
| 功能架构／算子视图 | 前者组织有意义的功能段，后者查看捕获到的原始操作。 |
| 暂停、进度、速度 | 停住或调整计算演示；全屏用于仔细观察。 |
| 标注：按需／全部／隐藏 | 控制侧旁的三维全息文字；按需模式优先显示当前计算与主干。 |

- **数据晶体**表示可见坐标处的实际数值；权重矩阵使用不同材质。全连接的权重线与同一 W 元素联动。
- **数值颜色**按当前张量已知数值的范围归一化；跨张量的相同颜色不保证数值相等。
- **注意力主干**保留投影、位置编码、相似度、概率分配、Value 汇聚与输出投影。缩放、掩码、Softmax 可聚合为一个功能段，点击后查看原始步骤。
- **按头组织**将通道分成 Head × 每头通道；**Key 对齐**交换 Token/通道轴以计算 QKᵀ；**合并多头**恢复每个 Token 的完整通道。晶体重排与共享展开用于说明坐标变化，元素数值保持不变。
- **全息标注固定在对应张量近侧**，随模型一起旋转与缩放，不随镜头重新排位。遮挡时按需隐藏；悬停文字会高亮它所属的结构。
- **函数图象**中的输入点位于 x 轴，输出点位于对应曲线上；刻度、曲线和投影线共用坐标。曲线颜色表示函数值。
- 界面播放的动效用于解释捕获的数值与依赖，播放速度不等于实际计算耗时。

张量尺寸来自代码。要改变自己模型的 Batch、通道或 Token 数，请修改输入和模型代码，再保存；网页里的参数控件只用于支持重建的内置示例。

## 7. 大模型：先看结构，再看真实计算单元

模型库提供 DeepSeek-V3、GLM-4.5、MiniMax-M1。总览按原配置表达重复层和专家组；点击后查看代表性 Decoder 的执行图，再进入 Attention、MoE、专家或基础算子。

内置执行图来自参考实现的原始 forward，使用缩小维度、固定种子、未训练权重。不同层级共享捕获张量 ID。它们不是完整检查点推理：DeepSeek 使用 Transformers 的展开式 MLA；MiniMax Lightning 使用原始 decode 分支并作 CPU 接口适配。[实现来源与边界](ATOMIC-COMPOSITION.md) 中有具体说明。

对自己的大模型，可选两种方式：

- 只查看支持家族的完整配置：使用 `architecture_from_config(config, family)`，适配器为 `deepseek_v3`、`glm45`、`minimax_m1`。见 [配置结构接口](LARGE_MODELS.md)。
- 查看实际计算：选择能在本机运行的子模块和小输入，交给 `export_model` 或工厂函数；若要观察已有权重，自己加载对应权重。框架不会自动缩小任意模型。

显示预算只限制绘图开销，不限制 PyTorch 执行内存。大模型的复制、加载与前向计算仍可能占用大量内存；无须为查看内置示例下载完整权重。

## 8. 常见问题

| 现象 | 处理 |
| --- | --- |
| 找不到 `silicondevine` / `torch` | 确认启动器选中了正确 Python；在该环境安装依赖与本地接口。 |
| 网页打开但没有模型 | 查看服务错误；确认 `build_model()` 返回 `(model, args)`，并且服务仍在运行。 |
| FX 报控制流错误 | 改用 `export`；若依赖实际输入选择分支，用 `execution` 或先导出子模块。 |
| 缺少数值或窗口无法读取 | 导出时启用 `include_values`；确认 IDE 连接、快照预算及窗口是否被保留。 |
| 模块只显示输入输出 | 查看「解读与参数 → 算子支持检查」；该操作可能需要数学插件。 |
| 图太大、文字太多或卡顿 | 返回功能架构、进入子模块、将标注设为按需，减小输入或可见坐标窗口。 |
| 保存后仍是旧图 | 捕获失败会保留旧图；查看错误。改 JSON 后重新保存入口 `.py` 文件。 |
| 更新后仍看到旧界面 | 停止旧服务，关闭旧启动器，重新打开项目目录内的 EXE，再刷新网页。 |

## 9. 前端集成与继续阅读

把框架嵌入自己的网页：

```ts
import { SiliconDevineViewer } from 'silicondevine.js';
const viewer = new SiliconDevineViewer(document.querySelector('#stage')!, {
  maxCells: 8192, cellsPerTensor: 128, maxConnections: 12000,
  labels: 'auto', pixelRatio: 1.5,
});
viewer.load(graph); // graph 为导入的 .sd.json 对象
// 页面卸载时：viewer.dispose();
```

容器需有高度，例如 `#stage { height: 650px; }`。库通过本地目录安装，尚未发布到 npm；宿主需使用项目声明兼容的 Three.js 版本。运行项目开发环境用 `npm install`、`npm run dev`，构建用 `npm run build`。

需要更细的资料时，从 [文档目录](INDEX.md) 进入；接口、模型来源、支持范围和验证记录各自独立维护。本页是日常使用的主入口。
