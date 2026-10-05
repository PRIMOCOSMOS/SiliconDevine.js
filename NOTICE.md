# 来源与依赖

水晶几何、数值色阶和权重动效从用户拥有的 `D:\Personpage` 工作站实现提取，并在此适配成独立框架。没有复制 TensorSpace.js 的代码；其分层、忠于张量的数据表达是此前设计讨论中的参考。

本项目尚未代用户选择开源许可证或发布 npm 包。

运行时依赖 Three.js（MIT）。示例字体 Tektur 来自 Fontsource 分发（SIL Open Font License）。正式再分发构建产物时应携带相应许可；构建包的 `THIRD_PARTY_LICENSES` 保存本次依赖许可原文。

Set Transformer 原文件位于 examples/upstream/set_transformer，保留原作者 MIT 许可证和固定提交元数据。LLaMA 通过独立安装的 Transformers 4.57.1 原生类运行，该库遵循 Apache-2.0；本项目未复制其实现或分发模型权重。

VAE 官方原型来自 pytorch/examples，固定提交与许可证保存在 examples/upstream/pytorch_vae。KaTeX 用于 VAE 数学讲解，许可证见 THIRD_PARTY_LICENSES/KaTeX.txt。
Chinese display lettering uses ZCOOL QingKe HuangYou, distributed under the SIL Open Font License. Source: https://github.com/google/fonts/tree/main/ofl/zcoolqingkehuangyou ; license: THIRD_PARTY_LICENSES/ZCOOL-OFL.txt. Numerical data colors and orbital brand geometry are independent systems.

大模型适配来源：DeepSeek-V3 的官方 inference/model.py（MIT）、MiniMax-M1 官方实现（Apache-2.0）、Transformers 4.57.1 的 GLM4-MoE 实现（Apache-2.0）与官方 checkpoint 配置。文件、原许可、提交号与 SHA-256 位于 examples/upstream/large_models；没有分发模型权重。GLM 实现文件保持原样，适配图代码位于 python/silicondevine/architecture.py。
