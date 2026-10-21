"""1 runを作ったコードの来歴（gitの版と、期待する動作の基準版）。

要約の`run_provenance`の辞書との変換を持つ。gitの情報の取り方は、ここでは扱わない。
"""

from dataclasses import dataclass
from typing import cast

# 今のコードが満たす、期待する動作の基準版（重要動作goldenの現在の版と同じ番号）。goldenの版を上げるときに、
# 一緒に上げる（testが、goldenのhelperの現在の版との一致を確かめる）。runの設定が選んだ扱い（前の版の扱いを、
# 設定で選ぶ場合を含む）とは関係しない。
BEHAVIOR_BASELINE_VERSION = 10
_RUN_PROVENANCE_KEYS = (
    "code_commit",
    "has_uncommitted_changes",
    "code_revision_unavailable_reason",
    "behavior_baseline_version",
)


@dataclass(frozen=True, kw_only=True)
class RunProvenance:
    """1 runの来歴。

    gitの情報が取れたときは、`code_commit`（commitのhash）と`has_uncommitted_changes`（gitが追跡するファイルの
    変更の有無）を持ち、`code_revision_unavailable_reason`はNone。取れなかったときは、2つがNoneで、理由を持つ。
    """

    code_commit: str | None
    has_uncommitted_changes: bool | None
    code_revision_unavailable_reason: str | None
    behavior_baseline_version: int

    def __post_init__(self) -> None:
        if type(self.behavior_baseline_version) is not int:
            raise TypeError("behavior_baseline_version must be builtin int")
        if self.behavior_baseline_version < 1:
            raise ValueError("behavior_baseline_version must be positive")
        if self.code_commit is None:
            if self.has_uncommitted_changes is not None:
                raise ValueError("has_uncommitted_changes must be None without code_commit")
            if type(self.code_revision_unavailable_reason) is not str:
                raise TypeError(
                    "code_revision_unavailable_reason must be builtin str without code_commit"
                )
            if not self.code_revision_unavailable_reason:
                raise ValueError("code_revision_unavailable_reason must not be empty")
            return
        if type(self.code_commit) is not str:
            raise TypeError("code_commit must be builtin str or None")
        if not self.code_commit:
            raise ValueError("code_commit must not be empty")
        if type(self.has_uncommitted_changes) is not bool:
            raise TypeError("has_uncommitted_changes must be builtin bool with code_commit")
        if self.code_revision_unavailable_reason is not None:
            raise ValueError("code_revision_unavailable_reason must be None with code_commit")


def convert_run_provenance_to_mapping(run_provenance: RunProvenance) -> dict[str, object]:
    """要約へ書く辞書（4つのキー）にする。"""
    if type(run_provenance) is not RunProvenance:
        raise TypeError("run_provenance must be exact RunProvenance")
    return {key: getattr(run_provenance, key) for key in _RUN_PROVENANCE_KEYS}


def read_run_provenance(run_provenance_mapping: object) -> RunProvenance:
    """要約の`run_provenance`の辞書を読む。キーが4つちょうどでない、値が不正なら、拒否する。"""
    if not isinstance(run_provenance_mapping, dict):
        raise TypeError("run_provenance must be a JSON object")
    # 値の型は、`RunProvenance`の検査が確かめる。
    provenance_values = cast("dict[str, object]", run_provenance_mapping)
    if set(provenance_values) != set(_RUN_PROVENANCE_KEYS):
        raise ValueError(
            f"run_provenance must hold exactly {', '.join(_RUN_PROVENANCE_KEYS)}: "
            f"{sorted(map(str, provenance_values))}"
        )
    return RunProvenance(
        code_commit=cast("str | None", provenance_values["code_commit"]),
        has_uncommitted_changes=cast("bool | None", provenance_values["has_uncommitted_changes"]),
        code_revision_unavailable_reason=cast(
            "str | None", provenance_values["code_revision_unavailable_reason"]
        ),
        behavior_baseline_version=cast("int", provenance_values["behavior_baseline_version"]),
    )
