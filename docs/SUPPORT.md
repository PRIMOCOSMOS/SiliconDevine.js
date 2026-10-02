# 0.3 支持范围

| 能力 | 实际范围 |
| --- | --- |
| 接入 | FX / torch.export、位置/关键字与嵌套输入、合法 DAG、共享参数、多输入输出 |
| 本地应用 | 桌面启动器、环境选择、模型监视、失败恢复、自动更新、按需张量窗口 |
| 数学依赖 | Linear、分组/膨胀 Conv1/2/3D、转置卷积1/2/3D、矩阵/向量/批量乘法、广播加法/乘法 |
| 更多依赖 | 最大/平均/自适应池化1/2/3D、BatchNorm、GroupNorm、InstanceNorm、Embedding、Concat、Softmax、LayerNorm、索引/切片、置换/转置/reshape、RMSNorm、chunk/split、stack、GQA 重复、掩码、差值 |
| 激活 | ReLU、GELU、Sigmoid、Tanh、SiLU、LeakyReLU、ELU、Softplus；真实输入输出同步着色 |
| 注意力 | export 的标准注意力、因果掩码、布尔/加性掩码与 GQA 展开为真实 repeat、transpose、matmul、scale、mask、softmax、matmul |
| 动态形状 | 接收 dynamic_shapes，记录 range_constraints；晶体表示本次输入的具体形状 |
| 导航 | 模块路径索引、图段分页、坐标窗口、保留边界输入输出 |

Conv 支持 same/valid、整数/元组 padding、reflect/replicate/circular。权重线与 W 晶体通过相同 Tensor ID / flatten index 联动。

## 明确边界

- 这是给定输入的 eval 前向图，不是反向梯度、优化器或全部 Python 分支。FX 和 export 各有可跟踪性限制。
- 完整依赖规则不意味着整个模型同时渲染。图段、晶体与连线有预算；复杂图按区域懒惰显示，未知值不当作零。
- 注意力分数矩阵超过 262144 项时保留融合边界。非零 attention dropout 保留融合边界。已验证 TinyGPT、LLaMA-style 与原生 Transformer 小型实例，不代表所有模型变体。
- 未注册算子（包括独立 padding 等）保留真实输入输出，提示内部数学动效未实现。插件不是所有 ATen 算子的集合，可通过 registerOperator 扩展。
- 稀疏、量化、复数 Tensor 明确拒绝；超出 JavaScript 安全整数范围的整数不能保证精确传输。主要目标是密集浮点网络。
- 静态 JSON 只包含已有数值；坐标偏移不产生缺失值。连接服务后才可读取预算内快照的其他坐标。
- 动态约束是元数据，不在浏览器实现符号求解或自动覆盖所有输入。修改输入需要重新执行 PyTorch。
- 激活曲线展示数学函数，输出数值由 PyTorch 执行捕获；声明式 GELU 使用 erf 近似，不宣称位级复现。
- 本次验证 PyTorch 2.9 CPU / Windows Edge；CUDA、Safari、超大图、WebXR 和所有 PyTorch 版本尚未验证。

模型文件由所选本机 Python 执行。浏览器不接受 Python 上传执行，不提供远程任意代码沙箱。

0.3 的 LLM 示例为随机初始化、小尺寸 prefill 前向模型；不含 tokenizer、预训练权重、KV-cache 增量解码或生成服务。RoPE 使用与相邻复数对等价的实数运算，框架仍不接受一般复数 Tensor。

SAB / ISAB 使用带归一化的 MAB 独立示例；纯坐标变换可在功能架构中合并，数学映射保持。Norm、最后一轴 Softmax、二元逐元素合并提供专用局部机制动效；原始图始终可访问。

0.4 新增上游源码直接接入和保守功能识别，详见 UPSTREAM.md。类型转换保留实际输出dtype和捕获数值，前端不模拟任意dtype位级转换。
