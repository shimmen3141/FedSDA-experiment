"""データセット名の正規形と、過去結果向けの互換変換を管理する。"""

import re


LEGACY_DATASET_NAMES = {
    "sea": "sea4",
    "circle": "circle2",
    "sine": "sine2",
}


def normalize_dataset_name(dataset):
    """旧名を論文上の概念数が明示された正規名へ変換する。"""
    if dataset is None:
        return None
    return LEGACY_DATASET_NAMES.get(str(dataset), str(dataset))


def dataset_cli_choices(canonical_names):
    """CLIで受理する正規名と後方互換名を返す。"""
    return tuple(canonical_names) + tuple(LEGACY_DATASET_NAMES)


def normalize_dataset_in_text(text):
    """ファイル名などに独立トークンとして含まれる旧名を変換する。"""
    normalized = str(text)
    for old_name, new_name in LEGACY_DATASET_NAMES.items():
        normalized = re.sub(
            rf"(?<![0-9A-Za-z]){re.escape(old_name)}(?![0-9A-Za-z])",
            new_name,
            normalized,
        )
    return normalized
