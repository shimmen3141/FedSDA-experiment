"""長い実験条件を安全な短い成果物名へ写像する。"""

from __future__ import annotations

import hashlib
import json
import re


DEFAULT_STEM_LIMIT = 72


def _safe_text(value):
    return re.sub(
        r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", str(value))
    ).strip("_")


def bounded_artifact_stem(value, *, hint=None, limit=DEFAULT_STEM_LIMIT):
    """意味を残しつつ、長いstemを内容ハッシュ付きで上限内へ収める。"""
    safe = _safe_text(value) or "artifact"
    if len(safe) <= limit:
        return safe
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
    prefix = _safe_text(hint) if hint else safe
    prefix = prefix[:max(1, limit - len(digest) - 1)].rstrip("_")
    return f"{prefix}_{digest}"


def raw_run_filename(identity, *, dataset, seed):
    """設定本体はNPZ内へ保存し、ファイル名は短いrun指紋だけにする。"""
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    dataset_hint = _safe_text(dataset)[:16] or "data"
    return f"run_{dataset_hint}_s{seed}_{digest}.npz"
