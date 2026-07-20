"""実験モード名と旧結果ファイルの互換変換を一元管理する。"""

FEDSDA_MODES = (
    "FedSDA_NoCached_ADWIN",
    "FedSDA_NoCached_ClassADWIN",
    "FedSDA_NoCached_ESR",
    "FedSDA_NoCached_ClassESR",
    "FedSDA_Cached_ADWIN",
    "FedSDA_Cached_ClassADWIN",
    "FedSDA_Cached_ESR",
    "FedSDA_Cached_ClassESR",
    "FedSDA_NoCached_ESR_UCB",
    "FedSDA_NoCached_ClassESR_UCB",
    "FedSDA_Cached_ESR_UCB",
    "FedSDA_Cached_ClassESR_UCB",
)

FEDDRIFT_MODES = ("FedDrift",)
BASELINE_MODES = ("FedSDA_without_server", "Oblivious")

# 過去のCSV・NPZを新しい解析コードで引き続き利用するための読み込み専用変換。
# v1は新しい正式手法と混同しないようLegacyとして明示する。
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
    "FedSDA_v2.2_ucb": "FedSDA_NoCached_ESR_UCB",
    "FedSDA_v2.3_ucb": "FedSDA_NoCached_ClassESR_UCB",
    "FedSDA_v3.2_ucb": "FedSDA_Cached_ESR_UCB",
    "FedSDA_v3.3_ucb": "FedSDA_Cached_ClassESR_UCB",
    "FedDrift_v2": "FedDrift",
}


def normalize_legacy_mode(mode):
    """旧バージョン名を現在の結果表示名へ変換する。"""
    return LEGACY_MODE_NAMES.get(mode, mode)


def normalize_legacy_series(series, old_mode, new_mode):
    """系列名の先頭に含まれる旧モード名だけを置換する。"""
    if not series or old_mode == new_mode:
        return series
    if series == old_mode:
        return new_mode
    prefix = f"{old_mode} "
    return f"{new_mode} {series[len(prefix):]}" if series.startswith(prefix) else series
