# GPT-SoVITS 多参考语音插件

按**角色**配置多条带情绪标注的参考语音，对话时由 AI 自动切换；每条还能单独设采样步数。

## 功能

- **情绪参考语音**分页：先选一个角色，再为它配置多条参考语音（情绪标注 + 参考音频 + 参考文本 + 采样步数）。
- **音频检测 / 转换**分页：检测音频能不能被 GPT-SoVITS 使用；可用就转成 3~9.5 秒的 wav，不可用会写明原因。
- **Whisper 语音转文本**分页：把音频转写成文本，方便填参考文本。
- 对话时 AI 在台词 JSON 里输出 `emotion`，插件按情绪挑该角色对应的参考音，在合成那一刻临时换
  `refer_audio_path` / `prompt_text` / `sample_steps`，**不改任何文件**。

## 安装

1. 把 `gpt_sovits_emotion_voice/` 整个目录复制到你的 Shinsekai 根目录下的
   `<Shinsekai根目录>/plugins/gpt_sovits_emotion_voice/`
   （最终应存在 `plugins/gpt_sovits_emotion_voice/plugin.py`）。
2. 在 `<Shinsekai根目录>/data/config/plugins.yaml` 末尾加：
   ```yaml
   - entry: plugins.gpt_sovits_emotion_voice.plugin:GptSovitsEmotionVoicePlugin
     enabled: true
   ```
3. **完全重启** Shinsekai（`plugins.yaml` 改动不支持热加载）。

### 依赖

- **必需**：无额外 Python 依赖（只用到宿主 SDK 和自带的 `yaml`）。
- **音频转换**：需要 `ffmpeg`；会自动优先用 GPT-SoVITS 自带的
  `<gpt_sovits_api_path>/runtime/ffmpeg(.exe)`，其次系统 PATH 里的 `ffmpeg`。
- **Whisper 分页（可选）**：需要 `faster-whisper`，通常启用 `whisper_asr` 插件时会一并装好；
  没装也不影响情绪切换，只是那个分页会提示未安装。
- 插件自身不会往 `plugins/` 里写数据；所有配置都写到 `<Shinsekai根目录>/data/plugins/com.local.gpt_sovits_emotion_voice/`。

## 数据位置

全部在插件根目录（`data/plugins/com.local.gpt_sovits_emotion_voice/`）下，按角色分类：

```
<plugin_root>/
  config.json                 # 全局开关（enabled / add_emotion_field / active_character / whisper）
  data/
    kukuri/
      config.json             # 该角色的参考语音（标注/文本/采样步数/音频相对路径）
      audio/
        3f2a....wav           # 已导入并规范化的参考音（wav、3~9.5 秒）
    _all/                     # 「所有角色」的配置放这里
    _tools/                   # 音频工具页转换出来的文件
```

> 从旧版升级时，插件会自动把旧 `config.json` 里的 `references`（绝对路径）迁移到
> `data/<角色>/`、把音频导入进来，旧文件备份为 `config.json.legacy.bak`。

## 使用

### 情绪参考语音（settings）

1. **基本设置**：启用、让 AI 输出 `emotion` 字段。
2. **角色**：选「当前编辑角色」→ 点 **切换角色** 载入它的参考语音；「该角色的兜底情绪」填某条的标注名。
3. **参考语音 1..8**：每条填
   - **情绪标注**：`开心/高兴/笑`（第一个词是主标注，后面是同义词，用 `/` 或 `，` 分隔）
   - **参考语音**：选本地音频（保存时会校验，不可用则整页不保存）
   - **参考文本**：与音频逐字一致的文本
   - **采样步数**：0 = 用全局/服务端默认；否则这条单独用这个值
   - **无参考文本**：勾选后这条不发参考文本（`prompt_text` 留空），只用参考音频驱动

### 无参考文本模式

GPT-SoVITS 在 `TTS_infer_pack/TTS.py` 里：`prompt_text` 为空时会走「无参考文本」路径，但如果模型走 vocoder 就直接报
`prompt_text cannot be empty when using SoVITS_V3`。

| 模型版本 | 支持无参考文本 |
|---|---|
| v1 / v2 / v2Pro / v2ProPlus | ✅ |
| v3 / v4 / v5 / v5dev / v5turbo | ❌（必须给参考文本） |

插件会从角色 SoVITS 权重文件的头 2 字节自动识别版本（GPT-SoVITS 自己的约定：`05`=v2Pro、`06`=v2ProPlus、
`08`=v5turbo …）。如果当前角色是 v3/v4/v5*，**保存时勾了「无参考文本」会直接拒绝保存并提示版本不支持**。

> 改完点 **保存**；「保存」只保存当前已载入的角色。切换角色用 **切换角色** 按钮。

### 音频检测 / 转换（tools）

选一个音频 → 点 **检测并转换**：

- ✅ 可用：显示原始时长，并转成 `data/_tools/<hash>.wav`（3~9.5 秒）
- ❌ 不可用：写明原因（文件不存在 / 无法解码 / 太短 / 太长）

### Whisper 语音转文本（tools）

选音频 → 选模型（默认复用 GPT-SoVITS 自带的 `large-v3`）→ 点 **开始转写**，结果可直接复制去填参考文本。

## 参考语音的硬性要求

GPT-SoVITS v2Pro 在 `TTS_SoVITS/TTS_infer_pack/TTS.py` 里**硬性要求参考音 3~10 秒**，超了直接报
「参考音频在3~10秒范围外」。所以：

- **保存时校验**：文件存在、能被 ffmpeg 解码、原始时长 1~300 秒；任何一条不满足，**整页不保存**并报错。
- **导入时规范化**：统一转成 `wav`，超过 10 秒的**裁到 9.5 秒**、不足 3 秒的**补静音到 3 秒**。
- 裁剪后参考文本要和「实际用到的片段」对得上，所以最好自己准备 3~10 秒的片段。

## 每条参考语音的采样步数

插件注册了自己的 `gpt-sovits` 适配器（覆盖内置实现），行为兼容 `gpt_sovits_sampling`：

- 适配器构造参数 `sample_steps` / `super_sampling` 来自 `data/config/api.yaml` 的
  `tts_extra_configs['gpt-sovits']`，作为**全局默认**；
- 某条参考语音命中且它的「采样步数」> 0 时，这次请求用它的值；
- 0 或没命中时用全局值。

参考（v5turbo 建议 4~16，v3/v4 建议 16~32；0 = 服务端默认）。

## 情绪怎么匹配

情绪信号来自（按优先级）：`emotion` 字段 → `vibe` 字段 → 文本里的 `(emotion: XXX)`。

匹配规则：标注/同义词与情绪信号**双向包含**（「失望」能命中 `被伤透了心感到彻底失望`；
`有点调侃的语气` 能命中 `调侃`）。都没命中用该角色的「兜底情绪」，再没有就回退宿主默认参考音。

> AI 输出的情绪词是「失望 / 生气 / 怒气 / 不满 / 嘲讽 / 鄙视 / 嫌弃 / 冷淡 / 调侃 / 戏谑 / 坏笑…」
> 这类短词。**标注建议直接写成 AI 会说的词**（可用 `/` 写多个），否则永远不会命中。

## config 结构

`<plugin_root>/config.json`：

```json
{ "enabled": true, "add_emotion_field": true, "active_character": "kukuri", "whisper": {} }
```

`<plugin_root>/data/<角色>/config.json`：

```json
{
  "character": "kukuri",
  "references": [
    {
      "label": "被伤透了心感到彻底失望",
      "keywords": ["难过", "悲伤", "倔强"],
      "audio": "audio/e1d8d3dc9e27090d.wav",
      "prompt_text": "ついてこないでよ今は一人になりたい気分なの…",
      "prompt_lang": "",
      "sample_steps": 12,
      "no_prompt": false,
      "default": true
    }
  ]
}
```

## 注意事项

- 只对 `gpt-sovits` / `kaggle-gpt-sovits` 生效。
- 参考文本必须与参考音频**逐字一致**。
- 命中情绪参考音时会**覆盖**立绘自带的参考音。
- 匹配逻辑按文件 mtime 自动重载，**保存后无需重启**；只有 `emotion` 字段开关需要重启。
- 如果同时装了 `gpt_sovits_sampling`，本插件的适配器会覆盖它的（全局采样步数仍从同一份配置读取，并多支持每条覆盖）。
