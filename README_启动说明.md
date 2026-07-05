# 宫庙建筑识别 Agent

这个项目把本地 LM Studio 大模型和已有的宫庙建筑识别程序连接起来，提供一个带聊天框的网页入口。

## 启动前

1. 打开 LM Studio。
2. 加载 `qwen3.5-9b`。
3. 启动 Local Server，地址保持：

```text
http://127.0.0.1:1234/v1
```

## 安装环境

建议使用之前的 `yolo` 环境，或者新建环境后安装：

```powershell
pip install -r requirements.txt
```

如果要用 RTX 5080 跑识别，请优先安装 CUDA 版 PyTorch。

## 启动

双击：

```text
run_agent.bat
```

然后浏览器打开：

```text
http://127.0.0.1:7860
```

## 当前功能

- 连接 LM Studio 的 OpenAI-compatible API。
- 网页聊天框。
- 单图、多图、ZIP 上传识别。
- 调用 `desktop_app` 中的现有模型权重。
- 输出结果图、CSV、JSON。
- 初版清晰宫庙正立面筛选。

