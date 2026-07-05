# LocateAnything-3B 本地单独运行版

这个目录是一个不依赖 TempleRecognitionAgent 后端的独立测试版。它只做一件事：输入图片和目标描述，调用 `nvidia/LocateAnything-3B`，输出坐标 JSON 和带框预览图。

## 1. 环境建议

推荐：

- WSL2 Ubuntu 或 Linux
- NVIDIA GPU
- CUDA 版 PyTorch
- 显存建议 12GB 起步，16GB 以上更稳

Windows 原生也可以尝试，但这类 Hugging Face 自定义视觉模型在 Linux/WSL2 下通常更省事。

## 2. 安装依赖

先安装与你 CUDA 匹配的 PyTorch。示例：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

然后安装本目录依赖：

```powershell
pip install -r locateanything_local/requirements.txt
```

第一次运行会从 Hugging Face 下载模型权重。模型较大，请确保网络可访问 Hugging Face。

## 3. 命令行运行

在项目根目录运行：

```powershell
python locateanything_local/locate.py `
  --input "E:\TempleRecognitionAgent\data\uploads\example.jpg" `
  --query "roof ridge dragon" `
  --output-dir "outputs\locateanything_local"
```

也可以对整个文件夹运行：

```powershell
python locateanything_local/locate.py `
  --input "E:\TempleRecognitionAgent\data\test_images" `
  --query "temple plaque" `
  --continue-on-error
```

## 4. PyQt5 可视化界面

运行：

```powershell
powershell -ExecutionPolicy Bypass -File locateanything_local\run_gui.ps1
```

或者双击：

```text
locateanything_local\run_gui.bat
```

界面支持：

- 图片拖拽输入
- 输入图和输出图并排对比
- 输入图片/文件夹配置
- 输出目录配置
- 定位指令输入框
- 可选完整 prompt
- 处理当前图片或处理整个输入目录

GUI 默认读取 `locate.py` 前面的配置：

```python
DEFAULT_INPUT_PATH = r"E:\TempleRecognitionAgent\data\uploads"
DEFAULT_OUTPUT_DIR = r"E:\TempleRecognitionAgent\outputs\locateanything_local"
```

## 5. 中文目标描述

可以直接用中文：

```powershell
python locateanything_local/locate.py `
  --input "E:\TempleRecognitionAgent\data\uploads\example.jpg" `
  --query "屋顶上的龙"
```

脚本会自动包装成：

```text
Locate all the instances that match the following description: 屋顶上的龙.
```

如果你想完全控制 prompt，可以用 `--prompt`：

```powershell
python locateanything_local/locate.py `
  --input "E:\TempleRecognitionAgent\data\uploads\example.jpg" `
  --query "unused" `
  --prompt "Locate a single instance that matches the following description: the main entrance plaque."
```

## 6. 输出

默认输出到：

```text
outputs\locateanything_local
```

里面会有：

- `locate_results.json`: 所有结果、原始模型输出、像素坐标
- `*_locate_*.png`: 带框/点的可视化结果图

JSON 中常用字段：

```json
{
  "boxes": [{"x1": 100, "y1": 80, "x2": 300, "y2": 260}],
  "points": [{"x": 210, "y": 160}],
  "answer": "<box><50><80><150><260></box>",
  "result_image": "outputs/locateanything_local/example_locate_..."
}
```

`boxes` 和 `points` 已经从模型的 0-1000 归一化坐标转换成原图像素坐标。

## 7. 常用参数

```powershell
--mode all       # 默认，找所有匹配目标
--mode single    # 只找一个目标
--mode point     # 输出点
--device cuda    # 或 cuda:0 / cpu
--dtype bfloat16 # 或 float16 / float32
--model-id nvidia/LocateAnything-3B
```

如果你已经手动下载模型，也可以把 `--model-id` 指向本地模型目录。
