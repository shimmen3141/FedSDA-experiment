"""現在の実験モード名と表示用系列名を管理する。"""

FEDSDA_MODES = (
    "FedSDA_NoCached_ADWIN",
    "FedSDA_NoCached_ClassADWIN",
    "FedSDA_NoCached_ESR",
    "FedSDA_NoCached_ClassESR",
    "FedSDA_NoCached_ClassESR_RestartingSoftRouting",
    "FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting",
    "FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
    "FedSDA_NoCached_ClassESR_ProtectedSoftRouting",
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
    detector_names = {
        "ADWIN", "ClassADWIN", "ESR", "ClassESR",
        "HDDMA", "ClassHDDMA", "HDDMW",
    }
    return next(
        (part for part in mode.split("_") if part in detector_names),
        None,
    )


def is_adwin_mode(mode):
    return fedsda_detector_name(mode) in {"ADWIN", "ClassADWIN"}


def is_esr_mode(mode):
    return fedsda_detector_name(mode) in {
        "ESR", "ClassESR",
    }


def is_hddm_mode(mode):
    return fedsda_detector_name(mode) in {"HDDMA", "ClassHDDMA", "HDDMW"}


def is_shared_representation_mode(mode):
    """複数の概念モデルが同じ特徴抽出部を共有するmodeかを返す。"""
    return any(token in mode for token in (
        "_SharedBackbone_", "_ResidualAdapter_",
    ))
