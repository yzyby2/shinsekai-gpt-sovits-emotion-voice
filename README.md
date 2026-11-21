# GPT-SoVITS 多参考语音插件

按**角色**配置多条带情绪标注的参考语音，对话时由 AI 根据台词情绪自动切换；支持每条单独采样步数、
无参考文本模式，以及音频检测/转换、Whisper 转写两个工具分页。

Shinsekai 插件。安装：把 `gpt_sovits_emotion_voice/` 复制到 `<Shinsekai根目录>/plugins/`，
在 `data/config/plugins.yaml` 末尾加：

```yaml
- entry: plugins.gpt_sovits_emotion_voice.plugin:GptSovitsEmotionVoicePlugin
  enabled: true
```

然后完全重启。详细用法见 [`gpt_sovits_emotion_voice/README.md`](gpt_sovits_emotion_voice/README.md)。

## 功能

- **情绪参考语音**：每个角色可配多条（情绪标注 + 参考音频 + 参考文本 + 采样步数 + 无参考文本）。
- **音频检测 / 转换**：校验音频能否被 GPT-SoVITS 使用，可用就规范成 3~9.5 秒 wav。
- **Whisper 语音转文本**：把音频转写成文本，方便填参考文本。
- 数据按角色存在 `data/plugins/com.local.gpt_sovits_emotion_voice/data/<角色>/`。

## 依赖

- 无额外 Python 依赖；
- 音频转换需要 `ffmpeg`（优先用 GPT-SoVITS 自带的 `runtime/ffmpeg`）；
- Whisper 分页可选，需要 `faster-whisper`。

## 许可

个人自用分享。
