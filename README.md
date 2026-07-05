# Minnan Taiwan Recognition Agent

闽台宫庙建筑图像识别 Agent。项目提供一个本地网页聊天入口，将远程 OpenAI-compatible 大语言模型 API 与本地宫庙建筑视觉识别模型连接起来，支持单图、多图、ZIP 批量上传、结果整理和导出。

## 功能

- 远程大语言模型调用：通过 `MODEL_API_BASE_URL` 连接公网模型 API，下载者本地不需要部署语言大模型。
- 宫庙建筑图像识别：支持屋顶样式、开间、瓦片、屋脊装饰、建筑主体区域、建筑屋顶区域等任务。
- 批量处理：支持多图和 ZIP 上传，输出识别结果、标注图、CSV 和 JSON。
- 正立面筛选：可对大量宫庙图片进行正立面/主体建筑筛选与整理。
- LocateAnything 接入：保留通用视觉定位相关代码和独立示例。
- 网页聊天界面：启动后访问 `http://127.0.0.1:7860` 使用。

## 重要说明

本仓库包含视觉模型权重、示例数据和历史输出文件，体积较大，并使用 Git LFS 管理大文件。克隆前请先安装 Git LFS，否则 `.pt`、`.pth`、图片、压缩包等文件可能只会下载为 LFS 指针文件。

```powershell
git lfs install
git clone https://github.com/ssxxbaba133133-dev/Minnan-Taiwan-Recognition-Agent.git
cd Minnan-Taiwan-Recognition-Agent
git lfs pull
```

语言大模型走远程 API；视觉识别模型权重已放在 `desktop_app/models` 中。

## 环境安装

推荐 Python 3.10 或 3.11。进入项目根目录后安装依赖：

```powershell
pip install -r requirements.txt
```

如果要使用 NVIDIA GPU 运行视觉模型，建议先按自己的 CUDA 版本安装对应的 PyTorch，再安装本项目依赖。

## 远程模型 API 配置

项目通过 `.env` 读取远程模型 API：

```env
MODEL_API_BASE_URL=https://your-public-api.example/v1
MODEL_API_KEY=your-api-key
MODEL_NAME=qwen3.5-27b@q3_k_s
```

接口需要兼容 OpenAI Chat Completions，至少应支持：

- `GET /v1/models`
- `POST /v1/chat/completions`

可以使用 LM Studio、vLLM、Ollama OpenAI-compatible server、llama.cpp server 等在模型机器上启动服务，再通过 cpolar、Cloudflare Tunnel、frp 或反向代理暴露公网地址。

## 启动

Windows 下可直接双击：

```text
run_agent.bat
```

或使用控制脚本：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\agent_control.ps1 start
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\agent_control.ps1 status
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\agent_control.ps1 stop
```

启动成功后打开：

```text
http://127.0.0.1:7860
```

## 目录结构

```text
backend/                 FastAPI 后端与模型调用逻辑
frontend/                网页聊天界面
desktop_app/             原桌面识别程序和视觉模型权重
desktop_app/models/      YOLO/分类模型权重
scripts/                 批处理、筛选、标注和控制脚本
locateanything_local/    LocateAnything 本地示例
data/                    上传数据和示例数据
outputs/                 历史输出、标注图、CSV、JSON
tools/                   cpolar、cloudflared 等辅助工具
docs/                    补充说明文档
```

## 常见问题

### 克隆后模型文件很小

说明 Git LFS 文件没有拉取下来。运行：

```powershell
git lfs install
git lfs pull
```

### 后端能启动，但聊天报模型 API 错误

检查 `.env` 中的 `MODEL_API_BASE_URL` 是否以 `/v1` 结尾，确认公网隧道正在运行，并确认远程服务能访问 `/v1/models`。

### 图片识别报模型文件缺失

确认 `desktop_app/models` 下的 `.pt`、`.pth` 权重文件已通过 Git LFS 拉取完成。

### 端口被占用

默认端口是 `7860`。可以先运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\agent_control.ps1 stop
```

再重新启动。
