# 架构与性能

[使用说明](USER_GUIDE.md) · [文档目录](INDEX.md)


## 数据流

```text
PyTorch 模型 + 示例输入             描述式 TypeScript / JSON
      ↓ FX / torch.export                  ↓ Builder
      └────────────── Model IR v1 ──────────┘
                            ↓ 校验
                   拓扑排序 + 坐标窗口
                            ↓
             张量布局 + 算子标量依赖插件
                            ↓
       实例化水晶 / GPU 权重线 / 内嵌标牌 / 交互调度
```

模型定义、数学依赖、布局和 WebGL 渲染各自独立。没有把某篇论文的名称、固定层数或网站 React 状态写入框架。宿主可以自行加载 IR，不需要执行 Python；Python 仅用于在本地生成 IR。

## 继承的视觉约束

- 统一晶体尺寸。数值只改变颜色，不改变元素的形状或大小。
- 有符号、对称 asinh 色阶；未知值用独立暗蓝色，当前计算使用淡白高亮。
- 同一依赖层的并行节点在同一平面；不同计算步骤沿主干分层。
- 权重/偏置在输入和输出之间对齐主干；不把偏置另算一次输出层。
- 5D NCDHW 张量保留真实 D 方向。二维数据不伪装成立方体体数据。
- 标牌是世界坐标里的固定网格，贴在张量平台前缘；没有引出线，也不随相机自动转向。
- MLP 保留权重网，激活函数有真实曲线；复杂图的当前算子展示逐项依赖，其他算子保留晶体结构。
- 卷积感受野用覆盖实际参与坐标的包围体与高亮晶体展示。

## 性能机制

1. 张量晶体合并到共享 InstancedMesh；基础晶体/边缘和高亮覆盖层分开。基础晶体只在加载时上传，不在每帧重建。
2. 权重关系用 LineSegments + shader。流动高亮只更新 uniform，稠密场景仍保留较暗连线。
3. 小型 MLP 保留全部可见权重；复杂图保留一份静态拓扑主干，仅当前算子显示动态数学连线，旧算子的 GPU 对象立即释放。缓存不随浏览无限增长。
4. 默认单图段 32 算子、8,192 个晶体预算、单张量 128 个可见坐标、当前算子 12,000 条连线预算。晶体预算可设为 512–16,384；一个图段最多 512 张量，超出时明确要求降低 maxNodes。张量窗口使用真实多维坐标；图段边界仍带输入张量。
5. DPR 默认封顶 1.5；窗口不可见或页面隐藏时停止渲染。尊重 reduced-motion，启动时暂停自动演示。
6. 手动相机操作不会触发 fit。只有显式载入/换图段/聚焦/总览改变适配镜头；ResizeObserver 只更新尺寸与 aspect。
7. `dispose()` 释放观察器、监听器、材质、纹理、几何体、实例化缓冲和 WebGL renderer。

**预算是明确的取舍，不是宣称“同时显示任意规模完整网络”。** 当前 0.1 使用连续图段控制上限，没有完整的语义聚类 LOD、LOD 跨级插值、远程流式张量分页或 GPU 推理。图格式保留源图，窗口和分段在 UI 中说明。

默认 WebGL2 / Three.js。WebGPU 不是当前版本的前提；先复用已有稳定渲染机制。源码包不内置第二份 Three.js，示例应用则打包为可离线部署的静态文件。

## 官方参考

- [PyTorch FX](https://docs.pytorch.org/docs/stable/fx.html)：图捕获与 Interpreter。
- [torch.export API](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html)：导出程序与函数化计算图。
- [torch.export 教程](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html)：输入约束与图捕获边界。
- [Three.js InstancedMesh](https://threejs.org/docs/pages/InstancedMesh.html)：实例化绘制、实例颜色、资源生命周期。

版本兼容需通过导出器测试核对；在线文档可能比当前本机 PyTorch 新。
