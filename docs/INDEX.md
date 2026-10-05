# 文档目录

日常使用只需阅读 [使用说明](USER_GUIDE.md)。启动器的「使用说明」按钮打开其 [浏览器阅读版](QUICKSTART.html)，两份内容从同一源文件生成。

## 接口参考

- [API](API.md)：图格式、查看器方法、算子插件。
- [IDE 与本地服务](LIVE.md)：watch、服务生命周期、张量窗口与协议。
- [支持范围](SUPPORT.md)：已覆盖运算及识别边界。
- [框架架构](ARCHITECTURE.md)：渲染、资源管理与性能预算。

## 模型与代码依据

- [源码组合](ATOMIC-COMPOSITION.md)：真实执行图、逐层展开、内置 LLM 的来源和限制。
- [大模型配置结构](LARGE_MODELS.md)：配置适配器、重复层与逻辑张量。
- [LLM 示例](LLM.md)：TinyGPT 与 LLaMA-style 测试架构。
- [上游原始实现](UPSTREAM.md)：SAB、ISAB 与原生 LLaMA 的来源。
- [VAE](VAE.md)：高斯参数、重参数采样及示例。
- [特殊算子与教学机制](MECHANISMS.md)：动态卷积、条件注意力及独立教学机制。

## 维护记录

- [验证记录](VALIDATION.md)：测试方式与适用范围。
- [启动器说明](LAUNCHER.md)：应用入口；日常流程以使用说明为准。

README 和「开始使用」仅作入口，不再分别维护另一套教程。修改使用流程后，更新 USER_GUIDE.md 并运行 `python tests/build-guide.py` 同步浏览器阅读版。
