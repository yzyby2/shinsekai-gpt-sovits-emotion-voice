"""配置与数据存储 + 情绪匹配。

数据布局（全部在插件根目录下）：

::

    <plugin_root>/
      config.json              # 全局：enabled / add_emotion_field / active_character / whisper
      data/
        <角色名>/
          config.json          # 该角色的参考语音列表
          audio/
            <hash>.wav         # 已规范化（wav、3~9.5 秒）的参考音
        _all/                  # 角色名为「所有角色」时
          ...

``config.json`` 里的 ``audio`` 是相对角色目录的路径，运行时解析成绝对路径。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALL_DIR = "_all"
DATA_DIR = "data"
AUDIO_DIR = "audio"
MIN_SEC = 3.0
MAX_SEC = 9.5


@dataclass
class ReferenceEntry:
    """一条带情绪标注的参考语音。"""

    character: str = ""  # 所属角色；空 = 所有角色
    label: str = ""  # 情绪标注（同时是匹配词）
    keywords: list[str] = field(default_factory=list)  # 额外同义/近义词
    audio: str = ""  # 相对 data/<角色>/ 的音频路径
    ref_audio_path: str = ""  # 运行时解析出来的绝对路径
    prompt_text: str = ""
    prompt_lang: str = ""  # 空 = 用角色 prompt_lang
    sample_steps: int = 0  # 0 = 用全局/服务端默认
    no_prompt: bool = False  # 无参考文本模式（仅 v1/v2/v2Pro/v2ProPlus 支持）
    default: bool = False  # True = 该角色无情绪命中时的兜底


@dataclass
class PluginConfig:
    enabled: bool = True
    add_emotion_field: bool = True
    active_character: str = ""  # 设置页当前编辑的角色
    references: list[ReferenceEntry] = field(default_factory=list)
    whisper: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)  # 全局 config.json 原文


def parse_label(raw: str) -> tuple[str, list[str]]:
    """把「开心/高兴/笑」这类标注拆成 (主标注, 同义词列表)。"""
    text = str(raw or "").strip()
    if not text:
        return "", []
    tokens = [t.strip() for t in re.split(r"[/,，、;；]+", text) if t.strip()]
    if not tokens:
        return "", []
    return tokens[0], tokens[1:]


def _coerce_keywords(raw: Any) -> list[str]:
    if isinstance(raw, str):
        parts = [p.strip() for p in re.split(r"[,，、;；]+", raw) if p.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = [str(p).strip() for p in raw]
    else:
        parts = []
    return [p for p in parts if p]


def _coerce_steps(raw: Any) -> int:
    try:
        value = int(raw if raw not in (None, "") else 0)
    except (TypeError, ValueError):
        value = 0
    return max(0, value)


# ----------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------
def char_dir(root: Path, character: str) -> Path:
    return Path(root) / DATA_DIR / (str(character or "").strip() or ALL_DIR)


def char_config_path(root: Path, character: str) -> Path:
    return char_dir(root, character) / "config.json"


def char_audio_dir(root: Path, character: str) -> Path:
    return char_dir(root, character) / AUDIO_DIR


def global_config_path(root: Path) -> Path:
    return Path(root) / "config.json"


# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
def load_global(root: Path) -> dict[str, Any]:
    path = global_config_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def save_global(root: Path, data: dict[str, Any]) -> None:
    path = global_config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# 角色配置
# ----------------------------------------------------------------------
def load_character(root: Path, character: str) -> list[ReferenceEntry]:
    path = char_config_path(root, character)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    base = char_dir(root, character)
    entries: list[ReferenceEntry] = []
    for item in data.get("references") or []:
        if not isinstance(item, dict):
            continue
        label, keywords = parse_label(str(item.get("label") or ""))
        keywords = keywords + _coerce_keywords(item.get("keywords"))
        audio = str(item.get("audio") or "").strip()
        entries.append(
            ReferenceEntry(
                character=character,
                label=label,
                keywords=keywords,
                audio=audio,
                ref_audio_path=str((base / audio).resolve()) if audio else "",
                prompt_text=str(item.get("prompt_text") or ""),
                prompt_lang=str(item.get("prompt_lang") or ""),
                sample_steps=_coerce_steps(item.get("sample_steps")),
                no_prompt=bool(item.get("no_prompt")),
                default=bool(item.get("default")),
            )
        )
    return entries


def save_character(root: Path, character: str, entries: list[ReferenceEntry]) -> None:
    path = char_config_path(root, character)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "character": character,
        "references": [
            {
                "label": e.label,
                "keywords": list(e.keywords),
                "audio": e.audio,
                "prompt_text": e.prompt_text,
                "prompt_lang": e.prompt_lang,
                "sample_steps": int(e.sample_steps),
                "no_prompt": bool(e.no_prompt),
                "default": bool(e.default),
            }
            for e in entries
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_characters(root: Path) -> list[str]:
    data_root = Path(root) / DATA_DIR
    if not data_root.is_dir():
        return []
    found: list[str] = []
    for item in sorted(data_root.iterdir()):
        if item.is_dir() and (item / "config.json").is_file():
            found.append("" if item.name == ALL_DIR else item.name)
    return found


def load_all(root: Path) -> list[ReferenceEntry]:
    refs: list[ReferenceEntry] = []
    for character in list_characters(root):
        refs.extend(load_character(root, character))
    return refs


def signature(root: Path) -> tuple:
    """用来感知跨进程/跨次保存的变化。"""
    items: list[tuple[str, int, int]] = []
    try:
        path = global_config_path(root)
        st = path.stat()
        items.append(("config.json", st.st_mtime_ns, st.st_size))
    except OSError:
        items.append(("config.json", 0, 0))
    data_root = Path(root) / DATA_DIR
    try:
        for item in sorted(data_root.glob("*/config.json")):
            try:
                st = item.stat()
                items.append((item.parent.name, st.st_mtime_ns, st.st_size))
            except OSError:
                continue
    except OSError:
        pass
    return tuple(items)


# ----------------------------------------------------------------------
# 匹配
# ----------------------------------------------------------------------
def entries_for(cfg: PluginConfig, character: str) -> list[ReferenceEntry]:
    target = str(character or "").strip()
    return [e for e in cfg.references if str(e.character or "").strip() == target]


def _applicable(entry: ReferenceEntry, character_name: str) -> bool:
    return (not entry.character) or entry.character == character_name


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def _tokens(value: str) -> list[str]:
    return [t for t in re.split(r"[/,，、;；\s|·.。!！?？~～]+", value) if t]


def _hits(entry: ReferenceEntry, emotion_signal: str) -> bool:
    hay = _norm(emotion_signal)
    if not hay:
        return False
    for raw in [entry.label, *entry.keywords]:
        cand = _norm(raw)
        if not cand:
            continue
        if cand in hay or hay in cand:
            return True
        for token in _tokens(hay):
            if len(token) >= 2 and token in cand:
                return True
        for token in _tokens(cand):
            if len(token) >= 2 and token in hay:
                return True
    return False


def select_reference(
    cfg: PluginConfig,
    character_name: str,
    emotion_signal: str,
) -> ReferenceEntry | None:
    """按情绪选一条参考语音；无命中回退该角色的 default 条目，否则 None。"""
    if not cfg.enabled:
        return None
    for entry in cfg.references:
        if not entry.ref_audio_path or entry.default:
            continue
        if not _applicable(entry, character_name):
            continue
        if _hits(entry, emotion_signal):
            return entry
    for entry in cfg.references:
        if not entry.ref_audio_path or not entry.default:
            continue
        if _applicable(entry, character_name):
            return entry
    return None


def has_applicable_reference(cfg: PluginConfig, character_name: str) -> bool:
    if not cfg.enabled:
        return False
    return any(
        e.ref_audio_path and _applicable(e, character_name)
        for e in cfg.references
    )
