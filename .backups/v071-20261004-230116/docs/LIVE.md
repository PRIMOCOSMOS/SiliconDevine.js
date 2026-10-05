# IDE 与 Web 工作台

## 应用入口

双击 start-app.cmd，选择模型文件、工厂函数（默认 build_model）、捕获接口及已安装 PyTorch 的 Python 环境，点击启动。自动选择空闲本机端口并打开网页。停止或关闭只影响此启动器拥有的服务。设置保存在 %LOCALAPPDATA%/SiliconDevine/launcher.json。

启动器使用 Python 自带 Tkinter。工作台已经构建在 demo-dist，不需运行 Node。启动器 Python 和模型 Python 可以不同，不会自动安装或改变环境。更换模型或环境时先停止，再启动。

## 保存即更新

```python
import torch
from torch import nn

def build_model():
    return nn.Sequential(nn.Linear(16, 8), nn.GELU()), (torch.randn(2, 16),)
```

也支持字典：{'model': model, 'args': (x,), 'kwargs': {}, 'options': {'backend': 'export', 'snapshot_budget': 67108864}}。

命令行：python -m silicondevine.watch model.py --factory build_model --backend fx

监视模型所在目录的 Python 文件（最多 5000 个），每次更新在独立子进程重新导入捕获，默认超时 90 秒。建议使用专门模型目录。导入代码会真实执行，只运行自己信任的本地代码。捕获失败保留旧图，保存后重试。

## 正在运行的 Python 程序

```python
from silicondevine import show
server = show(model, (example,), block=False)
server.update(model, (another_example,), backend='export')
server.wait()  # 保持服务；完成后 server.close()
```

默认 block=True。show 输出网页地址但不监视文件；自动监视用启动器/watch。先在目标环境 editable 安装本地 python 目录。非 editable 安装可传 static_dir='D:/SiliconDevine.js/demo-dist'。

## 数据接口与预算

服务仅绑定 127.0.0.1，没有 HTTP 上传执行接口。

- /api/status：revision、ready、loading、error。
- /api/model：当前版本与完整图 JSON。
- /api/tensor?id=...&indices=0,1&revision=...：指定真实 flatten 坐标数值，每次最多 2048 项。

过时窗口请求返回 409，未保留快照返回 404。默认快照最多 64 MiB，使用临时 float64 二进制文件。这不是整个模型内存上限：捕获仍需要模型复制和前向计算。超预算张量保留 shape 与初始 JSON 数值窗口，不伪造缺失值。

```ts
const live = new LiveConnection('http://127.0.0.1:5182', {
  onModel: model => viewer.update(model),
  onError: console.error,
});
viewer.setDataProvider((id, indices, signal) => live.tensorWindow(id, indices, signal));
live.start();
// 卸载：live.stop(); viewer.dispose();
```

前端自动轮询版本、按窗口读取，保留用户镜头。使用本机 HTTP 或自带工作台；公网 HTTPS 网站访问本地 HTTP 可能受混合内容/本地网络权限限制。
