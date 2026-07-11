# 闽台宫庙建筑识别 Agent

这是闽台宫庙建筑识别 Agent 的源代码仓库。当前源码版本为 **1.0.3**，对应 GitHub Release `v1.0.3`。

## 下载可直接使用的 Windows 版本

普通用户请前往 [Releases](https://github.com/ssxxbaba133133-dev/Minnan-Taiwan-Recognition-Agent/releases) 下载：

`TempleRecognitionAgent-Windows-CPU-1.0.3.zip`

Release 压缩包内已经包含 Windows Python 运行环境、程序依赖和 7 个视觉识别模型。解压后双击 `启动Agent.bat` 即可使用，不需要另外安装 Python，也不会在首次识别时下载视觉模型。

本仓库默认分支只保存源码、构建脚本和清单，不重复提交约 1.8 GiB 的运行环境、模型权重、用户上传图片或识别结果。因此 GitHub 自动生成的 `Source code (zip)` 不是可直接运行的完整成品。

## 主要功能

- 塌寿三分类
- 屋顶四分类
- 开间分类
- 瓦片分类
- 屋脊装饰识别
- 建筑主体区域识别
- 建筑屋顶区域识别
- OpenAI-compatible 远程大模型对话
- 可选联网搜索、批量图片和压缩包处理

视觉识别在本机运行；大模型对话通过网络调用配置的 OpenAI-compatible API。

## 项目结构

```text
backend/                    Web API、任务路由和模型调用
frontend/                   浏览器聊天界面
desktop_app/                视觉识别桌面程序源码
desktop_app/models/         模型目录说明（权重仅随 Release 提供）
scripts/                    启动、验证、构建和辅助脚本
config/runtime.conf.example 远程模型接口配置示例
models-manifest.json        7 个视觉权重的文件名、大小和 SHA256
runtime-manifest.json       便携运行环境版本清单
```

## 远程大模型配置

开发或自行构建时，将 `config/runtime.conf.example` 复制为 `config/runtime.conf`，再填写自己的接口信息：

```text
MODEL_API_BASE_URL=https://your-server.example/v1
MODEL_API_KEY=replace-with-your-api-token
MODEL_NAME=your-model-id
ENABLE_LOCATE_ANYTHING=0
```

接口需要兼容：

- `POST /v1/chat/completions`
- 推荐提供 `GET /v1/models`
- 如需鉴权，使用 `Authorization: Bearer <token>`

`config/runtime.conf` 和 `.env` 已被忽略，请勿把真实 Token 提交到仓库。

## 从源码构建便携包

源码仓库不包含 `runtime/` 和模型权重。维护者需要先把 `models-manifest.json` 中列出的权重放入 `desktop_app/models/`，然后构建 CPU 便携环境：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_portable_runtime.ps1 -BuilderPython "C:\path\to\python.exe"
```

完成后运行验证和打包：

```powershell
runtime\python.exe scripts\verify_package.py --full --imports
powershell -ExecutionPolicy Bypass -File scripts\build_release_zip.ps1
```

## 本地数据

- 上传文件写入 `data/`
- 识别结果写入 `outputs/`
- 这些目录中的运行数据不会提交到 GitHub

## 模型与许可证

视觉权重的再分发要求见 [MODEL_LICENSE_NOTICE.md](MODEL_LICENSE_NOTICE.md)。本仓库目前没有声明覆盖全部代码与模型的统一开源许可证；公开仓库不等同于自动授予再使用许可。

`LocateAnything-3B` 属于未随标准包提供的实验性外部模型，默认关闭，详见 `docs/LocateAnything接入说明.md`。
