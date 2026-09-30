# 验证记录

本机验证：2026-09-29，Windows，Node 24.12.0，PyTorch 2.9.0+cpu，Edge / WebGL。

- TypeScript 严格类型检查通过。
- 库构建、类型声明构建、可离线部署的演示构建通过。
- 764 个标量结果与真实 PyTorch 导出值一致：Linear、2D/3D 卷积、矩阵乘法、转置。
- 校验缺失引用、重复 ID、循环、未知值保留，以及实际多维坐标窗口、三维深度、参数主干对齐。
- Python 测试包括 FX/export 两个后端、原模型状态与 RNG 保持、数值预算、字典输出、自定义算子、动态分支明确失败、共享权重保留。
- 浏览器覆盖 MLP、2D/3D 卷积和注意力的桌面视图，以及手机宽度布局；测试悬停即时接管、缩放后镜头稳定、Batch/通道修改和错误 JSON 不破坏旧图。
- 浏览器脚本未报告页面错误。首轮四个示例约 46–71 次 draw call；这是当前小型示例的绘制统计，不是大模型性能或帧率保证。

复现：

```powershell
npm run examples
npm run typecheck
npm test
npm run build
# 另开终端运行 npm run dev
# 测试浏览器需要 Playwright（可自行安装为开发依赖）
node tests/browser.mjs
```

浏览器测试可用 `PLAYWRIGHT_MODULE` 指定已安装的 Playwright 模块，`BROWSER_CHANNEL` 默认 msedge，`DEMO_URL` 默认 http://127.0.0.1:5180。截图与机器可读报告写到 `.qa/`，不属于库运行时。

还没有 GPU/CUDA、Safari、WebXR 或十万节点级别的基准验证；也没有声称所有 ATen 算子的数学插件已经完备。
