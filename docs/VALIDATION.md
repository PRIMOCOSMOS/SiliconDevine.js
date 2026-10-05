# 0.4 验证记录

[使用说明](USER_GUIDE.md) · [文档目录](INDEX.md)


2026-10-01，Windows / PyTorch 2.9.0+cpu / Transformers 4.57.1 / Edge WebGL。

- 16项Python测试；基础764、扩展4342个标量；13份 LLM / 注意力图的63013个标量及全部依赖坐标通过。
- 作者原始SAB/ISAB文件哈希验证，Transformers原生LLaMA全模型输出对照，三个上游实例没有opaque基础算子。
- 官方Q/K/V投影识别、概率到Value到Token的局部计算，ISAB两阶段矩阵，RoPE函数体下钻/返回均通过浏览器测试。
- 2D/3D卷积感受野、残差路由、390px移动端无横向溢出；控制台与页面无错误。
- 首次捕获原生 LLaMA 的等待可能超过旧验收脚本的50秒上限；启动器显示已等待时长，验收等待130秒并保留详细失败日志。
- TypeScript、库、声明及独立应用构建通过。世界空间文字靠近所在层，消除矩形边框并进行投影避让。
- 功能识别是明确图模式和可验证代码调用位置，不是任意Python语义理解；性能与模型规模边界见UPSTREAM.md。

复现：npm test；npm run build；python serve.py --port 5186 后执行 npm run test:v04:browser。

---

# 0.3 验证记录

验证日期：2026-10-01；Windows，Node 24.12，PyTorch 2.9.0+cpu，Edge WebGL。

- 13 项 Python 测试；原有 764 + 4342 个标量校验；新增10份 LLM / SDPA / SAB / ISAB 图的44617个标量与全部依赖坐标校验通过。
- TinyGPT、LLaMA-style 的原生模型与导出输出对照；因果性、GQA、非方阵注意力、布尔/加性 mask、全屏蔽行、共享权重、RoPE 配对与融合预算边界均覆盖。
- SAB/ISAB 额外对照作者式 split/bmm 参考公式，验证1/sqrt(dim_V)缩放、置换等变性、诱导点参数和双向矩阵形状。
- 浏览器验证 Block/Attention/FFN 下钻、悬停立即接管、单一活动连线区域、掩码非有限值、SAB/ISAB 概率行、Norm/残差/门控动效、移动端无横向溢出；无页面脚本错误。
- 修复聚焦后立即拖动进度条切回旧模块的状态冲突。
- 同一视口的完整架构视图：TinyGPT 55次绘制 / 686可见元素，LLaMA 47次 / 520元素；相比原始算子第一页的128次 / 2605元素与109次 / 1670元素减少负担。这不是整图 FPS 或超大模型性能保证。
- TypeScript、库/类型声明/独立应用构建通过。启动器支持 MLP、TinyGPT、LLaMA-style、SAB、ISAB 预设。

复现：npm run examples；npm test；npm run build。启动 python serve.py --port 5184 后，执行 npm run test:llm:browser 与 npm run test:mechanism:browser。浏览器测试需要 Playwright，可指定 PLAYWRIGHT_MODULE。

原始模型与浏览器脚本随框架交付；本机截图和临时 .qa 报告不作为运行时文件部署。

---

# 0.2 验证记录

验证日期：2026-09-30。Windows，Node 24.12，PyTorch 2.9.0+cpu，Edge WebGL。

- TypeScript 检查、库/声明/独立工作台构建通过。
- 基础 764 个标量结果和新增 4342 个标量结果与 PyTorch 一致。新增 67 份参考图覆盖卷积/转置卷积/池化/归一化/Embedding/Concat/向量与矩阵运算，包含无 Batch 维卷积与池化。
- 9 项 Python 测试通过：双后端、输入与 RNG 保持、共享权重、kwargs/嵌套输入、动态约束、窗口读取、版本冲突、捕获失败恢复等。
- 原生 TransformerEncoderLayer 展开 40 个节点；展开后输出与原模型在 1e-6 误差范围内一致。
- 浏览器完整流程：模型保存热更新、语法错误保留旧图、修正恢复、镜头保留、W 水晶悬停与连线同步、真实激活样本、坐标窗口读取；无页面脚本错误。
- 桌面/390px 移动端检查；移动端无横向溢出。侧卡使用真实投影角点做按需避让。
- 桌面启动器实际启动捕获、浏览器地址交接、设置保存和自有进程树停止通过。该测试需允许 Windows 管理其创建的子进程，沙箱限制下停止测试可能失败。

复现：npm run examples；npm test；npm run build。浏览器：npm run test:browser（需 Playwright，可用 PLAYWRIGHT_MODULE 指定已安装模块）。启动器：python tests/launcher-smoke.py。

构建保留 Three.js 独立 chunk，Vite 会提示其大小超过 500 kB；不是构建失败。当前策略包括实例化晶体、GPU 流式线条、局部动效、坐标预算、图段分页及暂停时跳过不变场景的渲染。没有把小型示例结果当作超大模型帧率保证。

尚未验证 CUDA、Safari、超大模型或全部 ATen 算子。支持边界见 SUPPORT.md。

## v0.6 大模型扩展（2026-10-02）

- 25 项 Python 测试通过；包含官方 GLM 原生 meta 模块，以及从固定 DeepSeek / MiniMax 源文件提取的构造器参数校验。未运行这两个模型的 CUDA 内核。
- 原有 764 项 PyTorch 标量校验通过；新增模板循环、缺失引用、伪数值拒绝和逻辑分区无重叠覆盖检查。
- 浏览器逐个打开三种模型的所有 scope；真实鼠标悬停、点击进入、返回、相机稳定、数学排版与移动布局通过。
- 当前示例单视图为 132–336 个分区，不随重复层/专家数展开；动态通路只为当前功能区创建。
- VAE、MLP 参数控制和减少动态效果的回归检查通过。结构模式不提供数值推理、实际专家激活统计或运行显存预测。


## v0.7.1 接口与标注核查（2026-10-03）

- PyTorch 2.9.0+cpu：36 项 Python 测试通过，包含训练态 Dropout 边界、一般 Tensor.unfold 误识别防护和不支持的重采样模式。
- 新增接口：76 组 FX / export 捕获示例；8,872 项激活、填充、分块、重组、像素重排、eval Dropout 的前端标量校验。
- 空间与索引扩展：86 组捕获示例；10,394 项插值、二维网格采样、归约和索引的前端标量校验，包括非整数与不等比例缩放、边界反射、align_corners、多轴归约、关键字输入。
- 既有大模型小型计算演示的 1,425 项标量校验通过。
- Edge：DeepSeek / GLM 总览和子模块侧旁标注、桌面/移动排版、支持报告、空间采样邻域与加权贡献、VAE 和 MLP 参数回归通过。
- 视觉验证分为一次批量检查、一次集中修复和一次确认；保留用户指定的黑色科幻舞台与数值色彩。

复现：先构建，再运行 `npm run test:api`。浏览器专项为 `tests/browser-v071.mjs` 和 `tests/browser-spatial.mjs`，通过 DEMO_URL 选择已启动的站点。特殊张量、CUDA 和未覆盖的数学插件范围见 SUPPORT.md。

## 2026-10 · 功能聚合与使用文档

- `tests/semantic-views.mjs`：检查注意力聚合边界、无环拓扑与坐标映射，保留实际张量 ID。
- `tests/browser-atomic.mjs`：检查三个大模型的原始执行组合、概率功能段、重排动效、侧旁标注碰撞、移动布局及 Lightning 分支。
- `tests/test_user_guide.py`：运行使用说明中的完整模型示例，验证 fx / export / execution；检查生成文档链接。
- `tests/browser-guide.mjs`：检查阅读版章节跳转、手机宽度及参考文档导航。

本轮 Python 回归共 39 项通过。显示默认聚合有意义的功能段，原始计算仍可进入核查。浏览器检查使用 Edge / WebGL；大型真实检查点的显存与吞吐不在此测试范围内。
