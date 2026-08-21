# 语音转文字助手开源

一个面向桌面端的开源语音转文字助手项目。

## 当前状态

长录音转写模块已经迁入本仓库，当前仓库可以独立导入、运行本地 API、启动审阅界面并执行音频单元测试。

当前迁移范围包括：

- 可恢复的长录音流水线：静音切分、VAD、云端/本地双轨转写、重叠去重、说话人记忆和 Markdown 渲染；
- 音频预处理、云端提供方、本地 Qwen3-ASR 提供方、说话人识别和文本复核；
- 本地 FastAPI 审阅服务和 `audio-review-ui` 前端；
- 命令行入口、Windows 启动脚本、测试和使用手册。

模型权重、录音、转写产物和前端 `node_modules` 不进入仓库。模型默认放在 `data/models/audio`，也可以通过 `VOICE_ASSISTANT_AUDIO_MODEL_DIR` 或 `--model-dir` 指定用户自己的模型目录。

## 快速开始

开发版可以使用 `requirements.txt` 安装基础依赖，也可以使用 `pyproject.toml` 和 uv：

```powershell
python -m pip install -r requirements.txt
uv sync
uv run file-assistant --help
```

本地审阅工作台：

```powershell
启动录音文本工作台.bat
```

只使用本地模型时需要额外安装本地识别依赖：

```powershell
python -m pip install -r requirements-local.txt
uv sync --extra local
```

工作台默认使用低成本云端增强：先用 VAD 切出真正的人声区间，过滤空白和噪音，再使用 ERes2NetV2 保持说话人身份，只上传语音区间。云端完整转写不作为普通入口；本地 ASR 才需要额外安装 Qwen3-ASR。

具体模型准备、云端 API 配置和审阅工作流见 [录音文本工作台使用手册](docs/reference/audio-review-user-guide.md)。

## 开发原则

- 本地优先，明确区分本地处理与可选云端服务。
- 优先保证中文识别、专有名词和跨应用输入的稳定性。
- 每项功能都应尽量可测试、可解释、可回退。
- 发布前检查依赖、模型和第三方组件的许可证。
