# 语迹 VoiceTrace

本地优先的中文语音转文字与录音审阅工作台。

语迹把长录音整理成可回听、可检索、可复核的文字，适合会议、访谈、课程和个人录音。录音和转写结果默认保存在本机；云端服务是可选项，工作台默认只上传经过语音活动检测的语音区间。

## 功能

- 可恢复的长录音流水线：静音切分、语音活动检测（VAD）、重叠去重和 Markdown 渲染；
- 默认使用云端 ASR、本地 CPU VAD 和 ERes2NetV2 说话人关联；
- 可选安装并启用本地 Qwen3-ASR，不作为默认路径的硬件要求；
- 说话人识别、显示名管理和跨片段说话人记忆；
- 沿时间轴回听录音、阅读转写、查看话题、收藏句子和导出 Markdown；
- 本地 FastAPI 服务、浏览器审阅界面、命令行入口和 Windows 启动脚本；
- API Key 从环境变量读取，不写入代码和运行产物。

## 当前状态

版本 `0.1.0` 仍处于开发阶段，当前重点是 Windows 本机个人使用。项目暂不提供多人账号、远程音频存储或可直接暴露到公网的部署方案。

## 快速开始

需要 Python 3.12、[uv](https://docs.astral.sh/uv/) 和 Node.js 22.13 或更高版本。

```powershell
uv sync
cd audio-review-ui
npm install
cd ..
uv run voice-trace --help
```

启动录音审阅工作台：

```powershell
启动录音文本工作台.bat
```

也可以手动启动：

```powershell
uv run voice-trace audio-review --data-dir data
```

默认链路会使用 CPU 版 FunASR 做 VAD 和说话人特征提取，不要求 CUDA。
如果需要本地 Qwen3-ASR，再额外安装本地识别依赖：

```powershell
uv sync --extra local
```

具体的模型准备、API Key 配置和审阅流程见[录音审阅工作台使用手册](docs/reference/audio-review-user-guide.md)。

## 运行模式

- 云端增强：需要设置 `DASHSCOPE_API_KEY`；工作台默认先在本地用 CPU VAD 筛选语音区间，再上传这些区间，并用 CPU ERes2NetV2 关联说话人。
- 本地 ASR：默认关闭；安装 `local` 可选依赖、准备 Qwen3-ASR 模型，并显式传入 `--local` 才会启用。该路径需要 CUDA。
- 文本整理与总结：需要时设置 `DEEPSEEK_API_KEY`；原始转写不会被总结流程覆盖。

模型默认放在 `data/models/audio`，也可以通过 `VOICETRACE_AUDIO_MODEL_DIR` 或 `--model-dir` 指定其他目录。

## 项目结构

```text
research_kb/                 Python 运行时、API 和命令行入口
audio-review-ui/             浏览器审阅界面
docs/                        使用手册和模型说明
tests/                       单元测试和项目约束测试
data/models/audio/           本地模型目录占位，不提交模型权重
```

## 开发与检查

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
cd audio-review-ui
npm test
```

开发 benchmark、模型、缓存、录音和转写结果只保留在本地，不进入 Git。不要提交 `.env`、API Key、私钥、录音、模型权重或个人转写内容。

## 文档

- [录音审阅工作台使用手册](docs/reference/audio-review-user-guide.md)
- [模型打包与安装约定](docs/model-packaging.md)

## 许可证

当前仓库尚未附带许可证文件。项目维护者确定许可证并提交 `LICENSE` 之前，不应将本项目视为已经授予公众自由使用、修改或分发的权利。
