"""给 GPT-SoVITS 在对话时按情绪切换多条「带标注」的参考语音。

数据全部在插件根目录下，按角色分类：

    <plugin_root>/config.json              # 全局开关
    <plugin_root>/data/<角色>/config.json  # 该角色的参考语音（标注/文本/采样步数）
    <plugin_root>/data/<角色>/audio/*.wav  # 已规范化的参考音（wav、3~9.5 秒）

对话时 AI 在台词 JSON 里输出 ``emotion``，插件按这句的情绪在该角色的参考语音里挑一条，
在合成那一刻临时换 ``refer_audio_path`` / ``prompt_text`` / ``sample_steps``，不改任何文件。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import yaml

from sdk.logging import get_logger
from sdk.plugin import PluginBase
from sdk.plugin_host_context import PluginHostContext
from sdk.register import PluginCapabilityRegistry
from sdk.types import (
    FrontendConfigAction,
    FrontendConfigContribution,
    OutputContractPatch,
    OutputFieldSpec,
    RequirementSpec,
)

from ai.tts.tts_adapter import GPTSoVitsAdapter
from application.chat.dialog_media import DefaultTtsGenerationStrategy
from application.chat.dialog_media.lookup.factory import create_asset_lookup_strategy
from application.chat.presentation.handlers.dialog_media import CharacterMediaHandler
from application.runtime.context import get_app_runtime

from . import config as cfg_mod

logger = get_logger(__name__, plugin_id="com.local.gpt_sovits_emotion_voice")

GPT_SOVITS_PROVIDERS = {"gpt-sovits", "kaggle-gpt-sovits"}
CONTRACT_ID = "default.dialog.v1"
MAX_SLOTS = 8
PROVIDER = "gpt-sovits"

# 每条参考语音自己的采样步数（线程内生效，0 = 用全局）
_STEPS = threading.local()


def _current_steps() -> int:
    return int(getattr(_STEPS, "value", 0) or 0)


def _set_current_steps(value: int) -> None:
    _STEPS.value = max(0, int(value or 0))


# ----------------------------------------------------------------------
# 带采样步数的 GPT-SoVITS 适配器
# ----------------------------------------------------------------------
class EmotionStepsGPTSoVitsAdapter(GPTSoVitsAdapter):
    """在请求里补 ``sample_steps``，支持「每条参考语音单独设置」。"""

    def __init__(
        self,
        tts_server_url: str = "http://127.0.0.1:9880/",
        gpt_sovits_work_path: str | None = None,
        sample_steps: int = 0,
        super_sampling: bool = False,
    ):
        super().__init__(
            tts_server_url=tts_server_url,
            gpt_sovits_work_path=gpt_sovits_work_path,
        )
        self.sample_steps = cfg_mod._coerce_steps(sample_steps)
        self.super_sampling = bool(super_sampling)
        logger.info(
            "GPT-SoVITS sampling config: sample_steps=%s super_sampling=%s",
            self.sample_steps or "server-default",
            self.super_sampling,
            extra={"event": "tts.sovits.sampling"},
        )

    @classmethod
    def get_config_schema(cls) -> dict[str, dict]:
        return {
            "sample_steps": {
                "label": "GPT-SoVITS 全局采样步数（0=用服务端默认；v5turbo 建议 4~16，v3/v4 建议 16~32）",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 64,
            },
            "super_sampling": {
                "label": "super_sampling（仅 v3 模型生效）",
                "type": "bool",
                "default": False,
            },
        }

    def generate_speech(self, text, file_path=None, **kwargs):
        steps = _current_steps() or self.sample_steps
        if steps > 0:
            kwargs["sample_steps"] = int(steps)
        if self.super_sampling:
            kwargs["super_sampling"] = True
        return super().generate_speech(text, file_path=file_path, **kwargs)


# ----------------------------------------------------------------------
# 情绪信号与 TTS 策略
# ----------------------------------------------------------------------
def _message_field(msg: Any, key: str) -> str:
    value = getattr(msg, key, "")
    if not value:
        extra = getattr(msg, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            value = extra.get(key, "")
    return str(value or "")


def _emotion_signal(msg: Any) -> str:
    parts: list[str] = []
    for attr in ("emotion", "vibe"):
        value = _message_field(msg, attr)
        if value:
            parts.append(value)
    text = " ".join(
        x for x in (getattr(msg, "text", "") or "", getattr(msg, "translate", "") or "") if x
    )
    for match in re.findall(r"\(emotion:\s*([^)]+)\)", text, flags=re.IGNORECASE):
        parts.append(match.strip())
    return " ".join(parts)


class EmotionTtsGenerationStrategy(DefaultTtsGenerationStrategy):
    """在合成前按情绪覆盖参考音、参考文本和采样步数。"""

    def __init__(self, select, resolve_audio=None) -> None:
        super().__init__()
        self._select = select
        self._resolve_audio = resolve_audio
        self._active = False

    def generate(self, request):
        entry = self._select(request.character_name, _emotion_signal(request.message))
        if entry is None:
            yield from super().generate(request)
            return
        character = request.character
        saved = (
            character.refer_audio_path,
            character.prompt_text,
            character.prompt_lang,
        )
        ref_audio = entry.ref_audio_path
        if self._resolve_audio is not None:
            try:
                ref_audio = self._resolve_audio(entry.ref_audio_path) or ref_audio
            except Exception:
                ref_audio = entry.ref_audio_path
        character.refer_audio_path = ref_audio
        character.prompt_text = "" if entry.no_prompt else entry.prompt_text
        if entry.prompt_lang:
            character.prompt_lang = entry.prompt_lang
        self._active = True
        _set_current_steps(entry.sample_steps)
        try:
            yield from super().generate(request)
        finally:
            _set_current_steps(0)
            self._active = False
            (
                character.refer_audio_path,
                character.prompt_text,
                character.prompt_lang,
            ) = saved

    def _can_use_sprite_reference(self, request) -> bool:
        if self._active:
            return False
        return DefaultTtsGenerationStrategy._can_use_sprite_reference(request)


class EmotionVoiceHandler(CharacterMediaHandler):
    """角色台词处理器：命中情绪参考语音时接管，否则交回宿主内置处理器。"""

    def __init__(self, plugin) -> None:
        self._plugin = plugin
        self._lookup_cache: dict[str, Any] = {}
        strategy = EmotionTtsGenerationStrategy(
            plugin.select_reference, plugin.ensure_playable_audio
        )
        super().__init__(tts_generation_strategy=strategy)

    def can_handle(self, msg) -> bool:
        try:
            cfg = self._plugin.current_config()
            if not cfg.enabled:
                return False
            rt = get_app_runtime()
            if self._tts_provider(rt) not in GPT_SOVITS_PROVIDERS:
                return False
            name = self._resolve_character_name(rt, msg)
            if name is None:
                return False
            return cfg_mod.has_applicable_reference(cfg, name)
        except Exception:
            return False

    def handle(self, msg) -> None:
        try:
            rt = get_app_runtime()
            mode = str(
                getattr(getattr(rt, "llm_manager", None), "media_selection_mode", "")
                or "indexed"
            ).strip().lower()
            key = "semantic" if mode == "semantic" else "indexed"
            self.asset_lookup_strategy = self._lookup_cache.setdefault(
                key, create_asset_lookup_strategy(key)
            )
        except Exception:
            logger.debug("failed to resolve asset lookup strategy", exc_info=True)
        super().handle(msg)

    @staticmethod
    def _tts_provider(rt) -> str:
        try:
            api = getattr(getattr(rt, "config", None), "config", None)
            api = getattr(api, "api_config", None)
            return str(getattr(api, "tts_provider", "") or "").strip().lower()
        except Exception:
            return ""

    @staticmethod
    def _resolve_character_name(rt, msg) -> str | None:
        raw = str(getattr(msg, "name", "") or "")
        if not raw:
            return None
        config = getattr(rt, "config", None)
        get_char = getattr(config, "get_character_by_name", None)
        if get_char is None:
            return None
        name = raw
        if get_char(raw) is None:
            opencc = getattr(rt, "opencc", None)
            if opencc is not None:
                try:
                    name = opencc.convert(raw)
                except Exception:
                    name = raw
        return name if get_char(name) is not None else None


# ----------------------------------------------------------------------
# 插件
# ----------------------------------------------------------------------
class GptSovitsEmotionVoicePlugin(PluginBase):
    WHISPER_MODELS = [
        ("tiny", "tiny（最快、最不准）"),
        ("base", "base"),
        ("small", "small"),
        ("medium", "medium"),
        ("large-v2", "large-v2"),
        ("large-v3", "large-v3（最准，GPT-SoVITS 自带）"),
        ("large-v3-turbo", "large-v3-turbo（快、省显存，GPT-SoVITS 自带）"),
    ]
    WHISPER_LANGS = [
        ("auto", "自动识别"),
        ("ja", "日语 ja"),
        ("zh", "中文 zh"),
        ("en", "英语 en"),
        ("ko", "韩语 ko"),
        ("es", "西班牙语 es"),
        ("fr", "法语 fr"),
        ("de", "德语 de"),
        ("ru", "俄语 ru"),
    ]

    @property
    def plugin_id(self) -> str:
        return "com.local.gpt_sovits_emotion_voice"

    @property
    def plugin_version(self) -> str:
        return "0.6.1"

    @property
    def plugin_name(self) -> str:
        return "GPT-SoVITS 多参考语音插件"

    @property
    def plugin_description(self) -> str:
        return "按角色配置多条带情绪标注的参考语音，对话时由 AI 自动切换；支持每条单独采样步数与无参考文本模式。"

    @property
    def plugin_author(self) -> str:
        return "local"

    @property
    def priority(self) -> int:
        return 20

    # ------------------------------------------------------------------
    def initialize(
        self,
        register: PluginCapabilityRegistry,
        plugin_root: Path,
        host: PluginHostContext,
    ) -> None:
        _ = host
        self._plugin_root = Path(plugin_root)
        self._characters_path = (
            self._plugin_root.resolve().parents[2] / "data" / "config" / "characters.yaml"
        )
        self._api_config_path = (
            self._plugin_root.resolve().parents[2] / "data" / "config" / "api.yaml"
        )
        self._config_sig = None
        self._audio_tool_path = ""
        self._audio_tool_result = ""
        self._last_transcription = ""
        self._last_model_used = ""
        self._whisper_model = None
        self._whisper_key = ""
        self._models_link_failed = False

        self._migrate_legacy_if_needed()
        self._ensure_global_config()
        self.config = self._build_config()
        self._config_sig = cfg_mod.signature(self._plugin_root)

        register.register_tts_adapter(PROVIDER, EmotionStepsGPTSoVitsAdapter)
        register.register_message_handler(dialog_media_handler=EmotionVoiceHandler(self))
        register.register_frontend_config_page(self._build_config_page())
        register.register_frontend_config_page(self._build_audio_tool_page())
        register.register_frontend_config_page(self._build_whisper_page())
        if self.config.add_emotion_field:
            register.register_output_contract_patch(self._build_contract_patch())

        logger.info(
            "gpt-sovits emotion voice initialized (%d references, %d characters)",
            len(self.config.references),
            len(cfg_mod.list_characters(self._plugin_root)),
            extra={"event": "gpt_sovits_emotion_voice.initialized"},
        )

    # ------------------------------------------------------------------
    # 配置存储
    # ------------------------------------------------------------------
    def _ensure_global_config(self) -> None:
        path = cfg_mod.global_config_path(self._plugin_root)
        if path.is_file():
            return
        cfg_mod.save_global(
            self._plugin_root,
            {"enabled": True, "add_emotion_field": True, "active_character": "", "whisper": {}},
        )

    def _migrate_legacy_if_needed(self) -> None:
        """老版本把 references 写在 config.json 里（绝对路径）；迁移到 data/<角色>/。"""
        root = self._plugin_root
        marker = root / cfg_mod.DATA_DIR / ".migrated"
        if marker.is_file():
            return
        data = cfg_mod.load_global(root)
        legacy = data.get("references")
        if not isinstance(legacy, list) or not legacy:
            return
        logger.info("migrating legacy references into %s/data", root)
        by_char: dict[str, list[dict[str, Any]]] = {}
        for item in legacy:
            if not isinstance(item, dict):
                continue
            by_char.setdefault(str(item.get("character") or "").strip(), []).append(item)
        for character, items in by_char.items():
            entries: list[cfg_mod.ReferenceEntry] = []
            for item in items:
                source = str(item.get("ref_audio_path") or "").strip()
                rel = ""
                if source:
                    try:
                        rel = self._import_audio(character, source)
                    except Exception:
                        logger.warning("migrate: import failed for %s", source, exc_info=True)
                entries.append(
                    cfg_mod.ReferenceEntry(
                        character=character,
                        label=str(item.get("label") or ""),
                        keywords=[str(k) for k in (item.get("keywords") or [])],
                        audio=rel,
                        prompt_text=str(item.get("prompt_text") or ""),
                        prompt_lang=str(item.get("prompt_lang") or ""),
                        sample_steps=cfg_mod._coerce_steps(item.get("sample_steps")),
                        default=bool(item.get("default")),
                    )
                )
            cfg_mod.save_character(root, character, entries)
        keep = {
            key: value
            for key, value in data.items()
            if key not in ("references", "characters")
        }
        try:
            shutil.copy2(cfg_mod.global_config_path(root), root / "config.json.legacy.bak")
        except Exception:
            pass
        cfg_mod.save_global(root, keep)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("1", encoding="utf-8")
        except OSError:
            pass
        logger.info("legacy migration done")

    def _build_config(self) -> cfg_mod.PluginConfig:
        data = cfg_mod.load_global(self._plugin_root)
        return cfg_mod.PluginConfig(
            enabled=bool(data.get("enabled", True)),
            add_emotion_field=bool(data.get("add_emotion_field", True)),
            active_character=str(data.get("active_character") or "").strip(),
            references=cfg_mod.load_all(self._plugin_root),
            whisper=dict(data.get("whisper") or {}),
            raw=data,
        )

    def current_config(self) -> cfg_mod.PluginConfig:
        sig = cfg_mod.signature(self._plugin_root)
        if sig != getattr(self, "_config_sig", None):
            self.config = self._build_config()
            self._config_sig = sig
        return self.config

    def _persist_global(self, **updates: Any) -> None:
        data = cfg_mod.load_global(self._plugin_root)
        data.update(updates)
        cfg_mod.save_global(self._plugin_root, data)
        # 强制下次 current_config() 重新加载（保存后内存里的引用要立刻反映新数据）
        self._config_sig = None

    def select_reference(self, character_name: str, emotion_signal: str):
        return cfg_mod.select_reference(self.current_config(), character_name, emotion_signal)

    # ------------------------------------------------------------------
    # 路径 / 进程
    # ------------------------------------------------------------------
    def _gsv_api_path(self) -> Path | None:
        try:
            data = yaml.safe_load(self._api_config_path.read_text(encoding="utf-8")) or {}
            raw = str(data.get("gpt_sovits_api_path") or "").strip()
            if not raw:
                return None
            base = Path(raw)
            if base.suffix.lower() == ".py":
                base = base.parent
            return base if base.is_dir() else None
        except Exception:
            return None

    def _ffmpeg_path(self) -> str:
        try:
            from shutil import which

            exe = which("ffmpeg")
            if exe:
                return exe
        except Exception:
            pass
        base = self._gsv_api_path()
        if base is None:
            return ""
        for name in ("ffmpeg.exe", "ffmpeg"):
            for sub in ("runtime", "", "bin"):
                cand = (base / sub / name) if sub else (base / name)
                try:
                    if cand.is_file():
                        return str(cand)
                except OSError:
                    continue
        return ""

    def _list_character_names(self) -> list[str]:
        names: list[str] = []
        try:
            data = yaml.safe_load(self._characters_path.read_text(encoding="utf-8")) or []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and str(item.get("name") or "").strip():
                        names.append(str(item["name"]).strip())
        except Exception:
            pass
        for name in cfg_mod.list_characters(self._plugin_root):
            if name and name not in names:
                names.append(name)
        try:
            for item in sorted((self._plugin_root / cfg_mod.DATA_DIR).iterdir()):
                if not item.is_dir():
                    continue
                name = "" if item.name == cfg_mod.ALL_DIR else item.name
                if name and name not in names:
                    names.append(name)
        except OSError:
            pass
        return names

    # ------------------------------------------------------------------
    # 音频检测 / 规范化 / 导入
    # ------------------------------------------------------------------
    def probe_audio(self, path: str) -> tuple[bool, float, str]:
        """检测音频能否被 ffmpeg 解码，返回 (ok, 时长秒, 错误)。"""
        raw = str(path or "").strip()
        if not raw:
            return False, 0.0, "没有选择音频文件"
        src = Path(raw).expanduser()
        try:
            if not src.is_file():
                return False, 0.0, f"找不到文件：{raw}"
        except OSError:
            return False, 0.0, f"找不到文件：{raw}"
        ffmpeg = self._ffmpeg_path()
        if not ffmpeg:
            # 没有 ffmpeg 时退化为「文件存在且非空」
            try:
                return src.stat().st_size > 0, 0.0, ""
            except OSError:
                return False, 0.0, "无法读取文件"
        try:
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(src)],
                capture_output=True,
                timeout=30,
            )
        except Exception as exc:
            return False, 0.0, f"ffmpeg 调用失败：{exc}"
        text = (proc.stderr or b"").decode("utf-8", errors="replace")
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
        if not match or "Audio:" not in text:
            return False, 0.0, "不是可识别的音频文件（ffmpeg 无法解码）"
        duration = (
            int(match.group(1)) * 3600
            + int(match.group(2)) * 60
            + float(match.group(3))
        )
        if duration < 1.0:
            return False, duration, f"音频太短（{duration:.2f}s），至少 1 秒"
        if duration > 300:
            return False, duration, f"音频太长（{duration:.1f}s），请选 3~10 秒的片段"
        return True, duration, ""

    def _normalize_to(self, source: Path, dest: Path) -> bool:
        ffmpeg = self._ffmpeg_path()
        if not ffmpeg:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                return True
            except Exception:
                return False
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-i",
                    str(source),
                    "-af",
                    "apad=whole_dur=3",
                    "-t",
                    "9.5",
                    str(dest),
                ],
                capture_output=True,
                timeout=120,
            )
        except Exception:
            return False
        try:
            return dest.is_file() and dest.stat().st_size > 0
        except OSError:
            return False

    def _import_audio(self, character: str, source: str) -> str:
        """把外部音频规范化后放进 data/<角色>/audio/，返回相对路径。"""
        src = Path(source).expanduser()
        if not src.is_file():
            raise ValueError(f"找不到音频文件：{source}")
        audio_dir = cfg_mod.char_audio_dir(self._plugin_root, character)
        try:
            inside = audio_dir.resolve() in src.resolve().parents
        except Exception:
            inside = False
        if inside and src.suffix.lower() == ".wav":
            return f"{cfg_mod.AUDIO_DIR}/{src.name}"
        stat = src.stat()
        digest = hashlib.sha1(
            f"{src.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:16]
        dest = audio_dir / f"{digest}.wav"
        if not (dest.is_file() and dest.stat().st_size > 0):
            if not self._normalize_to(src, dest):
                raise ValueError(f"音频转换失败：{source}")
        return f"{cfg_mod.AUDIO_DIR}/{dest.name}"

    def ensure_playable_audio(self, path: str) -> str:
        """运行时兜底：正常保存时已经规范化成 wav；万一不是（如 ffmpeg 当时缺失），
        这里再转一次到 data/_converted/ 并缓存。"""
        raw = str(path or "").strip()
        if not raw:
            return raw
        src = Path(raw).expanduser()
        try:
            if not src.is_file():
                return raw
        except OSError:
            return raw
        if src.suffix.lower() == ".wav":
            return raw
        try:
            stat = src.stat()
            digest = hashlib.sha1(
                f"{src.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
            ).hexdigest()[:16]
        except Exception:
            return raw
        dest = self._plugin_root / cfg_mod.DATA_DIR / "_converted" / f"{digest}.wav"
        try:
            if dest.is_file() and dest.stat().st_size > 0:
                return str(dest)
        except OSError:
            pass
        if not self._normalize_to(src, dest):
            return raw
        logger.info("normalized reference audio at runtime: %s -> %s", src, dest)
        return str(dest)

    def normalize_tool_audio(self, source: str) -> tuple[bool, float, str]:
        """音频工具页：检测并规范化到 data/_tools/，返回 (ok, 原始时长, 输出路径/错误)。"""
        ok, duration, error = self.probe_audio(source)
        if not ok:
            return False, duration, error
        src = Path(source).expanduser()
        stat = src.stat()
        digest = hashlib.sha1(
            f"{src.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:16]
        dest = self._plugin_root / cfg_mod.DATA_DIR / "_tools" / f"{digest}.wav"
        if not (dest.is_file() and dest.stat().st_size > 0):
            if not self._normalize_to(src, dest):
                return False, duration, "转换失败（ffmpeg 执行出错）"
        return True, duration, str(dest)

    # ------------------------------------------------------------------
    # 设置页（角色 / 参考语音）
    # ------------------------------------------------------------------
    def _build_schema(self) -> list[dict[str, Any]]:
        schema: list[dict[str, Any]] = [
            {
                "id": "main",
                "title": "基本设置",
                "fields": [
                    {
                        "key": "enabled",
                        "label": "启用插件",
                        "type": "boolean",
                        "defaultValue": True,
                        "description": "关闭后插件不接管角色台词，完全回退到宿主默认行为。",
                    },
                    {
                        "key": "add_emotion_field",
                        "label": "让 AI 输出 emotion 字段",
                        "type": "boolean",
                        "defaultValue": True,
                        "description": "给 LLM 输出增加一个 emotion 字段，让模型给每句台词标注情绪。此开关改动后需重启生效。",
                    },
                ],
            },
            {
                "id": "character",
                "title": "角色",
                "fields": [
                    {
                        "key": "character",
                        "label": "当前编辑角色",
                        "type": "select",
                        "defaultValue": "",
                        "options": [],
                        "description": "选择角色后点下方「切换角色」载入它的参考语音；直接点保存只保存当前已载入的角色。",
                    },
                    {
                        "key": "default_label",
                        "label": "该角色的兜底情绪",
                        "type": "text",
                        "defaultValue": "",
                        "placeholder": "例如：平静",
                        "description": "填该角色某条参考语音的情绪标注名；AI 的情绪没命中时用它。留空则不兜底（用角色默认参考音）。",
                    },
                ],
            },
        ]
        for i in range(MAX_SLOTS):
            schema.append(
                {
                    "id": f"ref_{i}",
                    "title": f"参考语音 {i + 1}",
                    "fields": [
                        {
                            "key": f"ref_{i}_label",
                            "label": "情绪标注",
                            "type": "text",
                            "defaultValue": "",
                            "placeholder": "开心/高兴/笑",
                            "description": "第一个词是主标注，后面是它的同义/近义词（用 / 或 ，分隔）。建议写成 AI 会输出的情绪词。",
                        },
                        {
                            "key": f"ref_{i}_audio",
                            "label": "参考语音",
                            "type": "file",
                            "pathKind": "file",
                            "defaultValue": "",
                            "span": "full",
                            "description": "上传参考音频（必须 3~10 秒、可被解码；保存时会自动转成 wav 并裁到 9.5 秒内，不可用则不保存）。",
                        },
                        {
                            "key": f"ref_{i}_text",
                            "label": "参考文本",
                            "type": "textarea",
                            "defaultValue": "",
                            "span": "full",
                            "description": "与参考音频逐字一致的文本，句末带标点。",
                        },
                        {
                            "key": f"ref_{i}_steps",
                            "label": "采样步数",
                            "type": "integer",
                            "defaultValue": 0,
                            "min": 0,
                            "max": 64,
                            "description": "这条参考语音单独用的 sample_steps；0 = 用全局/服务端默认（v5turbo 建议 4~16，v3/v4 建议 16~32）。",
                        },
                        {
                            "key": f"ref_{i}_no_prompt",
                            "label": "无参考文本（仅 v1/v2/v2Pro/v2ProPlus）",
                            "type": "boolean",
                            "defaultValue": False,
                            "description": (
                                "勾选后这条不发参考文本（prompt_text 留空），只用参考音频驱动。"
                                "仅 v1 / v2 / v2Pro / v2ProPlus 支持；"
                                "v3 / v4 / v5 / v5dev / v5turbo 走 vocoder，必须提供参考文本，"
                                "否则 GPT-SoVITS 会报 prompt_text cannot be empty when using SoVITS_V3。"
                            ),
                        },
                    ],
                }
            )
        return schema

    def _refresh_character_options(self, schema: list[dict[str, Any]]) -> None:
        active = str(self.config.active_character or "").strip()
        options = [{"label": "所有角色", "value": ""}]
        seen = {""}
        for name in self._list_character_names():
            if name not in seen:
                options.append({"label": name, "value": name})
                seen.add(name)
        if active and active not in seen:
            options.append({"label": active, "value": active})
        for group in schema:
            for field in group.get("fields", []):
                if field.get("key") == "character":
                    field["options"] = options

    @staticmethod
    def _format_label(entry: cfg_mod.ReferenceEntry) -> str:
        if not entry.label:
            return ""
        if entry.keywords:
            return entry.label + "/" + "/".join(entry.keywords)
        return entry.label

    def _collect_slots(self, values: dict[str, Any]) -> list[dict[str, Any]]:
        slots: list[dict[str, Any]] = []
        for i in range(MAX_SLOTS):
            label_raw = str(values.get(f"ref_{i}_label", "") or "").strip()
            audio = str(values.get(f"ref_{i}_audio", "") or "").strip()
            text = str(values.get(f"ref_{i}_text", "") or "").strip()
            steps_raw = values.get(f"ref_{i}_steps", 0)
            if not label_raw and not audio and not text:
                continue
            slots.append(
                {
                    "index": i,
                    "label_raw": label_raw,
                    "audio": audio,
                    "text": text,
                    "steps": cfg_mod._coerce_steps(steps_raw),
                    "no_prompt": bool(values.get(f"ref_{i}_no_prompt")),
                }
            )
        return slots

    # 不支持「无参考文本」的模型版本（走 vocoder，必须给参考文本）
    NO_PROMPT_UNSUPPORTED = {"v3", "v4", "v5", "v5dev", "v5turbo"}
    HEAD_VERSIONS = {
        b"00": "v1",
        b"01": "v2",
        b"02": "v3",
        b"03": "v3",
        b"04": "v4",
        b"05": "v2Pro",
        b"06": "v2ProPlus",
        b"07": "v5dev",
        b"08": "v5turbo",
    }

    def _character_sovits_path(self, character: str) -> Path | None:
        try:
            data = yaml.safe_load(self._characters_path.read_text(encoding="utf-8")) or []
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        for item in data:
            if not isinstance(item, dict) or str(item.get("name") or "") != character:
                continue
            raw = str(item.get("sovits_model_path") or "").strip()
            if not raw:
                return None
            path = Path(raw)
            if not path.is_absolute():
                path = self._plugin_root.resolve().parents[2] / path
            try:
                return path if path.is_file() else None
            except OSError:
                return None
        return None

    def _detect_sovits_version(self, character: str) -> str | None:
        """从 SoVITS 权重文件头 2 字节读版本（GPT-SoVITS 自己的约定）。"""
        path = self._character_sovits_path(character)
        if path is None:
            return None
        try:
            with open(path, "rb") as handle:
                head = handle.read(2)
        except OSError:
            return None
        return self.HEAD_VERSIONS.get(head)

    def _save_character_from_values(self, values: dict[str, Any], character: str) -> None:
        slots = self._collect_slots(values)
        # 第一阶段：只校验，不改任何文件
        problems: list[str] = []
        version = self._detect_sovits_version(character) if character else None
        for slot in slots:
            n = slot["index"] + 1
            if not slot["label_raw"]:
                problems.append(f"第 {n} 条：缺少情绪标注")
            if not slot["audio"]:
                problems.append(f"第 {n} 条：缺少参考语音")
                continue
            ok, duration, error = self.probe_audio(slot["audio"])
            if not ok:
                problems.append(f"第 {n} 条：{error}")
            if slot["no_prompt"] and version in self.NO_PROMPT_UNSUPPORTED:
                problems.append(
                    f"第 {n} 条：无参考文本模式不支持 {version} 模型"
                    "（仅 v1/v2/v2Pro/v2ProPlus）"
                )
        if problems:
            raise ValueError("不能保存，以下参考语音不可用：\n" + "\n".join(problems))
        # 第二阶段：导入音频 + 写角色配置
        default_label = str(values.get("default_label", "") or "").strip()
        entries: list[cfg_mod.ReferenceEntry] = []
        for slot in slots:
            label, keywords = cfg_mod.parse_label(slot["label_raw"])
            rel = self._import_audio(character, slot["audio"])
            entries.append(
                cfg_mod.ReferenceEntry(
                    character=character,
                    label=label,
                    keywords=keywords,
                    audio=rel,
                    prompt_text=slot["text"],
                    sample_steps=slot["steps"],
                    no_prompt=slot["no_prompt"],
                )
            )
        for entry in entries:
            if default_label and entry.label == default_label:
                entry.default = True
                break
        cfg_mod.save_character(self._plugin_root, character, entries)

    def _build_config_page(self) -> FrontendConfigContribution:
        plugin = self
        schema = self._build_schema()

        def load_values() -> dict[str, Any]:
            plugin._refresh_character_options(schema)
            cfg = plugin.current_config()
            character = cfg.active_character
            slots = cfg_mod.entries_for(cfg, character)
            values: dict[str, Any] = {
                "enabled": cfg.enabled,
                "add_emotion_field": cfg.add_emotion_field,
                "character": character,
                "default_label": next((e.label for e in slots if e.default), ""),
            }
            for i in range(MAX_SLOTS):
                entry = slots[i] if i < len(slots) else None
                values[f"ref_{i}_label"] = plugin._format_label(entry) if entry else ""
                values[f"ref_{i}_audio"] = entry.ref_audio_path if entry else ""
                values[f"ref_{i}_text"] = entry.prompt_text if entry else ""
                values[f"ref_{i}_steps"] = entry.sample_steps if entry else 0
                values[f"ref_{i}_no_prompt"] = bool(entry.no_prompt) if entry else False
            return values

        def save_values(values: Any) -> None:
            if not isinstance(values, dict):
                return
            cfg = plugin.current_config()
            character = cfg.active_character
            plugin._save_character_from_values(values, character)
            plugin._persist_global(
                enabled=bool(values.get("enabled", True)),
                add_emotion_field=bool(values.get("add_emotion_field", True)),
            )

        def switch_character(values: Any) -> dict[str, Any]:
            if isinstance(values, dict):
                cfg = plugin.current_config()
                target = str(values.get("character", "") or "").strip()
                plugin._save_character_from_values(values, cfg.active_character)
                plugin._persist_global(active_character=target)
            return {}

        def check(values: Any) -> dict[str, Any]:
            cfg = plugin.current_config()
            slots = cfg_mod.entries_for(cfg, cfg.active_character)
            return {
                "ok": True,
                "character": cfg.active_character,
                "count": len(slots),
                "labels": [e.label for e in slots if e.label],
            }

        return FrontendConfigContribution(
            page_id="gpt_sovits_emotion_voice.settings",
            title="GPT-SoVITS 多参考语音插件",
            kind="settings",
            description=(
                "先选一个角色，再为它配置多条带情绪标注的参考语音；对话时由 AI 自动切换。"
                "参考语音必须是 3~10 秒、可被解码的音频（保存时会自动转 wav 并裁到 9.5 秒内）。"
                "「无参考文本」模式仅 v1 / v2 / v2Pro / v2ProPlus 支持；"
                "v3 / v4 / v5 / v5dev / v5turbo 走 vocoder，必须提供参考文本。"
            ),
            restart_hint="emotion 字段开关与参考语音池的提示词需要重启后完全生效；参考语音本身保存后即时生效。",
            schema=schema,
            load_values=load_values,
            save_values=save_values,
            actions=[
                FrontendConfigAction(
                    id="switch_character",
                    label="切换角色",
                    description="保存当前角色的修改，并载入所选角色的参考语音。",
                    variant="primary",
                    order=10.0,
                    run=switch_character,
                ),
                FrontendConfigAction(
                    id="check",
                    label="自检",
                    description="检查当前角色的参考语音数量与情绪标注。",
                    variant="ghost",
                    order=20.0,
                    run=check,
                ),
            ],
            order=80.0,
        )

    # ------------------------------------------------------------------
    # 音频工具页
    # ------------------------------------------------------------------
    def _build_audio_tool_page(self) -> FrontendConfigContribution:
        plugin = self

        schema: list[dict[str, Any]] = [
            {
                "id": "input",
                "title": "音频检测 / 转换",
                "fields": [
                    {
                        "key": "audio_path",
                        "label": "音频文件",
                        "type": "file",
                        "pathKind": "file",
                        "defaultValue": "",
                        "span": "full",
                        "description": "选择要检测的音频；会检查能否被解码、时长是否合适，并转成 3~9.5 秒的 wav。",
                    },
                    {
                        "key": "ffmpeg",
                        "label": "ffmpeg",
                        "type": "text",
                        "defaultValue": "",
                        "span": "full",
                        "description": "自动探测到的 ffmpeg 路径（优先 GPT-SoVITS 自带）。",
                    },
                    {
                        "key": "result",
                        "label": "结果",
                        "type": "textarea",
                        "defaultValue": "",
                        "span": "full",
                        "description": "点「检测并转换」后在这里显示结果。",
                    },
                ],
            },
        ]

        def load_values() -> dict[str, Any]:
            return {
                "audio_path": plugin._audio_tool_path,
                "ffmpeg": plugin._ffmpeg_path(),
                "result": plugin._audio_tool_result,
            }

        def save_values(values: Any) -> None:
            if isinstance(values, dict):
                plugin._audio_tool_path = str(values.get("audio_path", "") or "").strip()

        def run_check(values: Any) -> dict[str, Any]:
            if not isinstance(values, dict):
                return {}
            source = str(values.get("audio_path", "") or "").strip()
            plugin._audio_tool_path = source
            if not source:
                plugin._audio_tool_result = "请先选择音频文件。"
                return {}
            ok, duration, message = plugin.normalize_tool_audio(source)
            if ok:
                plugin._audio_tool_result = (
                    f"✅ 可用\n原始时长：{duration:.2f}s\n"
                    f"已转换：{message}\n（保存到 data/_tools/，3~9.5 秒）"
                )
            else:
                plugin._audio_tool_result = f"❌ 不可用：{message}"
            return {}

        return FrontendConfigContribution(
            page_id="gpt_sovits_emotion_voice.audio_tool",
            title="音频检测 / 转换",
            kind="tools",
            description="检测音频能否被 GPT-SoVITS 使用；可用就转成 wav（3~9.5 秒），不可用会告诉你原因。",
            schema=schema,
            load_values=load_values,
            save_values=save_values,
            actions=[
                FrontendConfigAction(
                    id="check",
                    label="检测并转换",
                    description="检查选中的音频是否可用，并规范化成 3~9.5 秒的 wav。",
                    variant="primary",
                    order=10.0,
                    run=run_check,
                )
            ],
            order=85.0,
        )

    # ------------------------------------------------------------------
    # Whisper 页
    # ------------------------------------------------------------------
    def _system_whisper_model_size(self) -> str:
        try:
            path = (
                self._plugin_root.resolve().parents[2]
                / "data"
                / "config"
                / "system_config.yaml"
            )
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            size = str(data.get("asr_whisper_model_size") or "").strip()
            if size:
                return size
        except Exception:
            pass
        return "small"

    def _gsv_models_dir(self) -> Path | None:
        base = self._gsv_api_path()
        if base is None:
            return None
        models = base / "tools" / "asr" / "models"
        return models if models.is_dir() else None

    def _ensure_models_link(self) -> Path | None:
        link = self._plugin_root / "models"
        try:
            if link.is_dir():
                return link
        except OSError:
            pass
        if getattr(self, "_models_link_failed", False):
            return None
        target = self._gsv_models_dir()
        if target is None:
            self._models_link_failed = True
            return None
        try:
            os.symlink(str(target), str(link), target_is_directory=True)
            logger.info("linked GPT-SoVITS models via symlink: %s -> %s", link, target)
            return link
        except Exception:
            pass
        try:
            proc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
            )
            if proc.returncode == 0:
                logger.info("linked GPT-SoVITS models via junction: %s -> %s", link, target)
                return link
            detail = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
            logger.info("GPT-SoVITS models link unavailable (%s); using directory directly", detail)
        except Exception as exc:
            logger.info("GPT-SoVITS models link unavailable (%s); using directory directly", exc)
        self._models_link_failed = True
        return None

    def _whisper_model_roots(self) -> list[Path]:
        roots: list[Path] = []
        link = self._ensure_models_link()
        if link is not None:
            roots.append(link)
        gsv = self._gsv_models_dir()
        if gsv is not None and gsv not in roots:
            roots.append(gsv)
        return roots

    def _list_local_whisper_models(self) -> list[str]:
        for root in self._whisper_model_roots():
            sizes: list[str] = []
            try:
                for item in root.iterdir():
                    if item.name.startswith("faster-whisper-") and item.is_dir():
                        sizes.append(item.name[len("faster-whisper-") :])
            except Exception:
                continue
            if sizes:
                return sizes
        return []

    def _default_model_size(self) -> str:
        local = self._list_local_whisper_models()
        system_size = self._system_whisper_model_size()
        if system_size in local:
            return system_size
        for preferred in ("large-v3", "large-v3-turbo", "large-v2", "medium"):
            if preferred in local:
                return preferred
        return system_size

    @staticmethod
    def _hf_hub_candidates() -> list[Path]:
        roots: list[Path] = []
        hub = str(os.environ.get("HF_HUB_CACHE") or "").strip()
        if hub:
            roots.append(Path(hub))
        home = str(os.environ.get("HF_HOME") or "").strip()
        if home:
            roots.append(Path(home) / "hub")
        legacy = str(os.environ.get("HUGGINGFACE_HUB_CACHE") or "").strip()
        if legacy:
            roots.append(Path(legacy))
        roots.append(Path("data") / "cache" / "huggingface" / "hub")
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        unique: list[Path] = []
        for root in roots:
            if root not in unique:
                unique.append(root)
        return unique

    def _resolve_whisper_model_ref(self, model_size: str, custom_path: str) -> tuple[str, str]:
        custom = str(custom_path or "").strip()
        if custom:
            p = Path(custom).expanduser()
            if not p.is_absolute():
                p = self._plugin_root / p
            if p.exists():
                return str(p), f"自定义路径 {p}"
            return str(p), f"自定义路径（不存在）：{p}"
        size = str(model_size or "").strip() or self._default_model_size()
        for root in self._whisper_model_roots():
            candidate = root / f"faster-whisper-{size}"
            if candidate.is_dir():
                return str(candidate), f"GPT-SoVITS 自带模型 {candidate}"
        repo = f"models--Systran--faster-whisper-{size}"
        for root in self._hf_hub_candidates():
            try:
                snapshots = root / repo / "snapshots"
                if not snapshots.is_dir():
                    continue
                dirs = [d for d in snapshots.iterdir() if d.is_dir()]
                if dirs:
                    newest = max(dirs, key=lambda d: d.stat().st_mtime)
                    return str(newest), f"HF 缓存 {newest}"
            except Exception:
                continue
        return size, f"HF 模型名 Systran/faster-whisper-{size}（本地未缓存，首次会下载）"

    def _whisper_prefs(self) -> dict[str, Any]:
        w = self.current_config().whisper or {}
        return {
            "audio_path": str(w.get("audio_path") or ""),
            "model_size": str(w.get("model_size") or self._default_model_size()),
            "model_path": str(w.get("model_path") or ""),
            "language": str(w.get("language") or "auto"),
            "task": "translate" if str(w.get("task") or "").lower() == "translate" else "transcribe",
        }

    def _save_whisper_prefs(self, values: dict[str, Any]) -> None:
        self._persist_global(
            whisper={
                "audio_path": str(values.get("audio_path", "") or "").strip(),
                "model_size": str(values.get("model_size", "") or "").strip()
                or self._default_model_size(),
                "model_path": str(values.get("model_path", "") or "").strip(),
                "language": str(values.get("language", "auto") or "auto").strip() or "auto",
                "task": "translate"
                if str(values.get("task", "") or "").strip().lower() == "translate"
                else "transcribe",
            }
        )

    @staticmethod
    def _resolve_whisper_device() -> tuple[str, list[str]]:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", ["float16", "int8_float16", "int8", "float32"]
        except Exception:
            pass
        return "cpu", ["int8", "float32"]

    def _get_whisper_model(self, model_ref: str):
        key = str(model_ref or "small").strip() or "small"
        if self._whisper_model is not None and self._whisper_key == key:
            return self._whisper_model
        from faster_whisper import WhisperModel

        device, compute_chain = self._resolve_whisper_device()
        last_err: Exception | None = None
        for compute_type in compute_chain:
            try:
                model = WhisperModel(key, device=device, compute_type=compute_type)
            except Exception as exc:
                last_err = exc
                continue
            self._whisper_model = model
            self._whisper_key = key
            return model
        raise RuntimeError(f"加载 Whisper 模型失败（{key}）：{last_err}")

    def _run_transcription(
        self,
        audio_path: str,
        model_size: str,
        model_path: str,
        language: str,
        task: str,
    ) -> tuple[str, str]:
        path = str(audio_path or "").strip()
        if not path:
            return "", "请先选择要转写的音频文件。"
        try:
            if not Path(path).expanduser().is_file():
                return "", f"找不到音频文件：{path}"
        except Exception:
            pass
        model_ref, model_note = self._resolve_whisper_model_ref(model_size, model_path)
        self._last_model_used = model_note
        try:
            model = self._get_whisper_model(model_ref)
        except ImportError:
            return "", "未安装 faster-whisper（请启用 whisper_asr 插件，或手动 pip install faster-whisper）。"
        except Exception as exc:
            return "", f"加载 Whisper 失败：{exc}"
        kwargs: dict[str, Any] = {
            "beam_size": 5,
            "vad_filter": True,
            "without_timestamps": True,
        }
        lang = str(language or "auto").strip()
        if lang and lang.lower() != "auto":
            kwargs["language"] = lang
        if str(task or "").strip().lower() == "translate":
            kwargs["task"] = "translate"
        try:
            segments, _info = model.transcribe(path, **kwargs)
            text = "".join(getattr(seg, "text", "") for seg in segments).strip()
        except Exception as exc:
            return "", f"转写失败：{exc}"
        if not text:
            return "", "没有识别到语音内容（音频可能太短、音量太小或不是人声）。"
        return text, ""

    def _build_whisper_page(self) -> FrontendConfigContribution:
        plugin = self
        model_options = [{"label": label, "value": value} for value, label in self.WHISPER_MODELS]
        lang_options = [{"label": label, "value": value} for value, label in self.WHISPER_LANGS]
        schema: list[dict[str, Any]] = [
            {
                "id": "input",
                "title": "音频与参数",
                "fields": [
                    {
                        "key": "audio_path",
                        "label": "音频文件",
                        "type": "file",
                        "pathKind": "file",
                        "defaultValue": "",
                        "span": "full",
                        "description": "选择要转写成文本的音频（wav / mp3 / m4a / flac…）。",
                    },
                    {
                        "key": "model_size",
                        "label": "Whisper 模型",
                        "type": "select",
                        "defaultValue": "small",
                        "options": model_options,
                        "description": "越大越准也越慢。优先复用 GPT-SoVITS 自带的 tools/asr/models（自动建软链接），其次本地 HF 缓存。",
                    },
                    {
                        "key": "model_path",
                        "label": "自定义模型目录（可选）",
                        "type": "file",
                        "pathKind": "directory",
                        "defaultValue": "",
                        "span": "full",
                        "description": "留空则自动用 GPT-SoVITS 自带模型（插件目录下的 models 软链接）或 HF 缓存。也可手动选一个本地 faster-whisper 模型目录。",
                    },
                    {
                        "key": "language",
                        "label": "语音语言",
                        "type": "select",
                        "defaultValue": "auto",
                        "options": lang_options,
                        "description": "指定语言比自动识别更准更快。",
                    },
                    {
                        "key": "task",
                        "label": "任务",
                        "type": "select",
                        "defaultValue": "transcribe",
                        "options": [
                            {"label": "转写（保持原语言，推荐）", "value": "transcribe"},
                            {"label": "翻译成英文（Whisper translate）", "value": "translate"},
                        ],
                        "description": "给参考语音配参考文本时，用「转写」保持原语言逐字一致。",
                    },
                ],
            },
            {
                "id": "output",
                "title": "转写结果",
                "fields": [
                    {
                        "key": "result",
                        "label": "文本",
                        "type": "textarea",
                        "defaultValue": "",
                        "span": "full",
                        "description": "点「开始转写」后在这里显示结果；可复制去填参考文本。",
                    },
                    {
                        "key": "model_used",
                        "label": "本次使用的模型",
                        "type": "text",
                        "defaultValue": "",
                        "span": "full",
                        "description": "自动显示这次实际加载的模型目录或 HF 模型名。",
                    },
                ],
            },
        ]

        def load_values() -> dict[str, Any]:
            prefs = plugin._whisper_prefs()
            try:
                preview = plugin._resolve_whisper_model_ref(
                    prefs["model_size"], prefs["model_path"]
                )[1]
            except Exception:
                preview = ""
            return {
                "audio_path": prefs["audio_path"],
                "model_size": prefs["model_size"],
                "model_path": prefs["model_path"],
                "language": prefs["language"],
                "task": prefs["task"],
                "result": plugin._last_transcription,
                "model_used": plugin._last_model_used or preview,
            }

        def save_values(values: Any) -> None:
            if isinstance(values, dict):
                plugin._save_whisper_prefs(values)

        def transcribe(values: Any) -> dict[str, Any]:
            if isinstance(values, dict):
                plugin._save_whisper_prefs(values)
                text, error = plugin._run_transcription(
                    str(values.get("audio_path", "") or ""),
                    str(values.get("model_size", "") or ""),
                    str(values.get("model_path", "") or ""),
                    str(values.get("language", "auto") or "auto"),
                    str(values.get("task", "transcribe") or "transcribe"),
                )
                plugin._last_transcription = text or (f"❌ {error}" if error else "")
            return {}

        return FrontendConfigContribution(
            page_id="gpt_sovits_emotion_voice.whisper",
            title="Whisper 语音转文本",
            kind="tools",
            description="选一段音频，用本机 Whisper 转写成文本，方便填参考文本（prompt_text）。",
            restart_hint="首次使用会下载并加载 Whisper 模型，可能需要几十秒到几分钟。",
            schema=schema,
            load_values=load_values,
            save_values=save_values,
            actions=[
                FrontendConfigAction(
                    id="transcribe",
                    label="开始转写",
                    description="用 Whisper 把上面选中的音频转写成文本。",
                    variant="primary",
                    order=10.0,
                    run=transcribe,
                )
            ],
            order=90.0,
        )

    # ------------------------------------------------------------------
    # LLM 输出契约
    # ------------------------------------------------------------------
    def _build_contract_patch(self) -> OutputContractPatch:
        labels = [e.label for e in self.config.references if e.label and not e.default]
        label_hint = "、".join(labels[:12]) if labels else "开心、生气、悲伤、平静"
        return OutputContractPatch(
            id="gpt_sovits_emotion_voice.emotion_field",
            target_contract=CONTRACT_ID,
            priority=40.0,
            add_fields=(
                OutputFieldSpec(
                    key="emotion",
                    type="string",
                    description=(
                        f"本条台词的简短情绪标签（用于切换参考语音），例如：{label_hint}；"
                        "只输出真实存在的情绪，拿不准就省略该字段。"
                    ),
                    required=False,
                ),
            ),
            add_requirements=(
                RequirementSpec(
                    id="gpt_sovits_emotion_voice.emotion_label",
                    text=(
                        "可输出一个 emotion 字段，用简短词语标注本条台词的情绪（用于切换参考语音），"
                        "例如：开心、生气、悲伤、平静。只使用真实存在的情绪，拿不准就省略该字段。"
                    ),
                    order=175,
                ),
            ),
        )
