# Temple Recognition Agent - 远程大模型 API 版

这个目录是从原项目复制出来的独立版本，原项目不会被修改。语言大模型改为调用远程 OpenAI-compatible API，因此下载者本地不需要安装 LM Studio，也不需要下载语言大模型权重。

## 下载者怎么运行

1. 安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

2. 复制配置文件：

```powershell
copy .env.example .env
```

3. 编辑 `.env`：

```env
MODEL_API_BASE_URL=https://你的公网模型API地址/v1
MODEL_API_KEY=你分配给下载者的密钥
MODEL_NAME=你的模型ID
```

4. 双击运行：

```text
run_agent.bat
```

浏览器打开：

```text
http://127.0.0.1:7860
```

## 你这边需要提供什么

你的机器上继续运行本地大模型服务，例如 LM Studio、vLLM、Ollama OpenAI-compatible server 或 llama.cpp server。公网入口需要最终转发到类似这个地址：

```text
http://127.0.0.1:1234/v1
```

对外给下载者填写的是公网地址，例如：

```text
https://xxxx.cpolar.top/v1
```

建议公网入口必须加：

- HTTPS
- API Key 鉴权
- 访问频率限制
- 日志和异常监控

## 重要说明

这个版本远程化的是“语言大模型 API”。项目里的宫庙图片识别功能仍然依赖 `desktop_app/models` 下的 YOLO/分类权重；我没有把这些权重复制进来。没有这些视觉权重时，普通聊天和远程大模型调用可以工作，图片识别任务会提示模型文件缺失。

如果你希望别人连视觉识别权重也完全不用下载，需要下一步把“图片识别”也做成你这边的远程服务。
