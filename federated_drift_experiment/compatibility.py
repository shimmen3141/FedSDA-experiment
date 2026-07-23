"""旧手法名・旧データセット名を読み込むための後方互換定義。"""

import re


LEGACY_DATASET_NAMES = {
    "sea": "sea4",
    "circle": "circle2",
    "sine": "sine2",
}

LEGACY_MODE_NAMES = {
    "FedSDA": "FedSDA_Legacy",
    "FedSDA_v2": "FedSDA_NoCached_ADWIN",
    "FedSDA_v2.1": "FedSDA_NoCached_ClassADWIN",
    "FedSDA_v2.2": "FedSDA_NoCached_ESR",
    "FedSDA_v2.3": "FedSDA_NoCached_ClassESR",
    "FedSDA_v3": "FedSDA_Cached_ADWIN",
    "FedSDA_v3.1": "FedSDA_Cached_ClassADWIN",
    "FedSDA_v3.2": "FedSDA_Cached_ESR",
    "FedSDA_v3.3": "FedSDA_Cached_ClassESR",
    "FedDrift_v2": "FedDrift",
}


def normalize_dataset_name(dataset):
    """旧データセット名を論文上の概念数を示す正規名へ変換する。"""
    if dataset is None:
        return None
    return LEGACY_DATASET_NAMES.get(str(dataset), str(dataset))


def dataset_cli_choices(canonical_names):
    """CLIで受理する正規名と後方互換名を返す。"""
    return tuple(canonical_names) + tuple(LEGACY_DATASET_NAMES)


def normalize_dataset_in_text(text):
    """独立トークンとして含まれる旧データセット名を変換する。"""
    normalized = str(text)
    for old_name, new_name in LEGACY_DATASET_NAMES.items():
        normalized = re.sub(
            rf"(?<![0-9A-Za-z]){re.escape(old_name)}(?![0-9A-Za-z])",
            new_name,
            normalized,
        )
    return normalized


def normalize_legacy_mode(mode):
    """旧バージョン名を現在の手法名へ変換する。"""
    return LEGACY_MODE_NAMES.get(mode, mode)


def normalize_method_in_text(text):
    """文章・ファイル名中の旧手法名だけを現在名へ変換する。"""
    normalized = str(text)
    for old_mode, new_mode in sorted(
        LEGACY_MODE_NAMES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if old_mode == "FedSDA":
            # 現行名（FedSDA_Cached_*等）の接頭辞は旧v1名とみなさない。
            pattern = rf"(?<![0-9A-Za-z_]){re.escape(old_mode)}(?![0-9A-Za-z_])"
            normalized = re.sub(pattern, new_mode, normalized)
        else:
            # バージョン付き旧名はファイル名の区切り「_」の前でも置換する。
            normalized = normalized.replace(old_mode, new_mode)
    return normalized


def normalize_names_in_text(text):
    """手法名・データセット名だけを現在表記へ変換する。"""
    return normalize_dataset_in_text(normalize_method_in_text(text))

