# 开发与验证

建议使用 Node 22.12+、Python 3.10+。当前验证版本和平台范围见 [兼容性说明](docs/COMPATIBILITY.md)。

```sh
npm ci
python -m pip install -e './python[test]'
npm run build
npm test
npx playwright install chromium
npm run test:browser
```

`npm test` 串行生成并验证 Python 导出结果，避免多个测试同时覆盖 `.qa` 文件。单独检查可运行 `npm run test:core` 或 `npm run test:python`。`PYTHON` 可指定解释器；`BROWSER_CHANNEL=msedge` 可使用已安装的 Edge。浏览器主套件自动启动和关闭预览服务。

附加算子回归：`npm run test:api`。验证可信本地仓库：`node tests/repository-browser.mjs --source PATH --entry module:Class`；该命令使用独立测试配置，自动管理服务并把结果写入 `.qa`。

修改捕获或算子时，应使用小型真实 PyTorch 模型对照输出数值；新增未识别算子应保留边界，不能伪造内部计算。涉及自动导入时，同时验证构造参数、输入契约、失败恢复及上一份模型保留。不得为了演示自动执行训练入口、下载权重或安装模型依赖。

提交需说明问题、行为变化、验证命令和未验证范围。CI 覆盖 Windows/Linux 的构建、数值回归与浏览器恢复流程；远端结果以实际运行记录为准。

本仓库尚未选定项目自身的开源许可证。第三方代码和资产必须保留 `NOTICE.md`、`THIRD_PARTY_LICENSES` 与上游来源记录；正式公开发布前由项目所有者决定许可证。
