# 本地音频模型目录

这是本地模型的默认目录。模型文件不会提交到 Git；也可以通过环境变量
`VOICE_ASSISTANT_AUDIO_MODEL_DIR` 或命令行参数 `--model-dir` 指定其他位置。

默认目录结构：

```text
audio/
├─ Qwen3-ASR-1.7B/   # 可选，本地 ASR，约 4.38 GiB
├─ fsmn-vad/         # 低成本语音区间检测，约 4 MB
└─ ERes2NetV2/       # 跨片段说话人特征，约 71 MB
```

云端完整转写模式可以不安装这些模型。启用“只上传语音区间”时需要准备
`fsmn-vad` 和 `ERes2NetV2`；启用本地 ASR 时再准备 `Qwen3-ASR-1.7B`。
程序会在第一次运行时下载缺失模型，或者使用用户指定目录中的已有模型。
