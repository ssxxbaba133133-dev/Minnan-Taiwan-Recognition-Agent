# LocateAnything-3B 接入说明

`nvidia/LocateAnything-3B` 是图像加文本的视觉定位模型。它不是 LM Studio 里常见的 GGUF 聊天模型，建议通过 Hugging Face Transformers 加载，然后让本项目后端调用。

## 适用任务

- 找出图片中符合描述的目标，例如“屋顶上的龙”“牌匾”“入口门洞”。
- 返回目标框坐标和带框预览图。
- 作为现有 YOLO/分类模型的补充，用来处理开放词表目标。

## 标准包状态

标准 Windows Release 不包含 `LocateAnything-3B` 模型或其额外依赖，且通过 `ENABLE_LOCATE_ANYTHING=0` 默认关闭。下面内容仅用于维护者将外部服务接入源码，不影响随包提供的 7 个宫庙视觉任务。

## 安装

建议在 WSL2 Ubuntu 或 Linux + NVIDIA GPU 环境里运行。

请在独立环境中安装与目标 GPU、CUDA 和模型版本匹配的 PyTorch、Transformers 及该模型要求的可选依赖。本仓库不固定或随包分发这组实验性依赖。

如果你在 Windows 原生环境运行，模型可能能加载，但更容易遇到 CUDA、flash attention、视频/视觉预处理依赖问题。

## 环境变量

可选配置：

```powershell
$env:LOCATE_ANYTHING_MODEL_ID="nvidia/LocateAnything-3B"
$env:LOCATE_ANYTHING_DEVICE="cuda"
$env:LOCATE_ANYTHING_DTYPE="bfloat16"
$env:LOCATE_ANYTHING_MAX_NEW_TOKENS="8192"
```

## 直接调用 API

启动后端后调用：

```powershell
curl.exe -X POST http://127.0.0.1:7860/api/locate_anything `
  -F "query=roof ridge dragon" `
  -F "mode=all" `
  -F "files=@C:\path\to\example.jpg"
```

返回字段里重点看：

- `results[].boxes`: 像素坐标框，格式为 `x1,y1,x2,y2`。
- `results[].points`: 点坐标。
- `results[].answer`: 模型原始输出。
- `images[].url`: 后端生成的可视化结果图。

## 在聊天里触发

上传图片后，消息里明确包含下面任意关键词即可走 LocateAnything：

- `LocateAnything`
- `通用定位`
- `通用视觉定位`
- `目标定位`
- `视觉定位`
- `用这个模型`
- `用定位模型`

示例：

```text
用 LocateAnything 定位屋顶上的龙
通用视觉定位：牌匾
目标定位：入口处的门神
```

普通的“屋顶区域识别”“建筑主体区域识别”等仍走项目原来的 YOLO 模型。

## 代码位置

- 推理封装：`backend/locate_anything_client.py`
- 独立接口：`backend/app.py` 的 `/api/locate_anything`
- 聊天显式触发：`backend/app.py` 的 `/api/agent_message`
