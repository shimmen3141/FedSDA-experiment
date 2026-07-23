"""現在の実験モード名と表示用系列名を管理する。"""

from .compatibility import LEGACY_MODE_NAMES, normalize_legacy_mode

FEDSDA_MODES = (
    "FedSDA_NoCached_ADWIN",
    "FedSDA_NoCached_ClassADWIN",
    "FedSDA_NoCached_ESR",
    "FedSDA_NoCached_ClassESR",
    "FedSDA_NoCached_HDDMA",
    "FedSDA_NoCached_ClassHDDMA",
    "FedSDA_NoCached_HDDMW",
    "FedSDA_Cached_ADWIN",
    "FedSDA_Cached_ClassADWIN",
    "FedSDA_Cached_ESR",
    "FedSDA_Cached_ClassESR",
    "FedSDA_Cached_HDDMA",
    "FedSDA_Cached_ClassHDDMA",
    "FedSDA_Cached_HDDMW",
)

FEDDRIFT_MODES = ("FedDrift",)
BASELINE_MODES = ("FedSDA_without_server", "Oblivious")


def fedsda_detector_name(mode):
    """FedSDAモード名から検出器部分を返す。"""
    if mode == "FedSDA_without_server":
        return "ADWIN"
    if mode not in FEDSDA_MODES:
        return None
    detector = mode.rsplit("_", 1)[-1]
    return detector


def is_adwin_mode(mode):
    return fedsda_detector_name(mode) in {"ADWIN", "ClassADWIN"}


def is_esr_mode(mode):
    return fedsda_detector_name(mode) in {"ESR", "ClassESR"}


def is_hddm_mode(mode):
    return fedsda_detector_name(mode) in {"HDDMA", "ClassHDDMA", "HDDMW"}

def normalize_legacy_series(series, old_mode, new_mode):
    """系列名の先頭に含まれる旧モード名だけを置換する。"""
    if not series or old_mode == new_mode:
        return series
    if series == old_mode:
        return new_mode
    prefix = f"{old_mode} "
    return f"{new_mode} {series[len(prefix):]}" if series.startswith(prefix) else series


def normalize_series_notation(series):
    """過去の掃引系列表記を現在の論文・凡例記号へ統一する。"""
    if not series:
        return series
    normalized = str(series)
    normalized = normalized.replace("AGG_INTERVAL sweep", "A sweep")
    normalized = normalized.replace("δ_adwin sweep", "δ_ADWIN sweep")
    normalized = normalized.replace("δ_adwin=", "δ_ADWIN=")
    if normalized.startswith("FedDrift"):
        normalized = normalized.replace("batch sweep", "B_detect sweep")
        normalized = normalized.replace(" δ sweep", " δ_FedDrift sweep")
        normalized = normalized.replace(" δ_drift sweep", " δ_FedDrift sweep")
        normalized = normalized.replace("batch=", "B_detect=")
        normalized = normalized.replace("(δ=", "(δ_FedDrift=")
        normalized = normalized.replace("(δ_drift=", "(δ_FedDrift=")
    return normalized
