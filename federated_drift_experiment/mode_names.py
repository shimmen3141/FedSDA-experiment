"""現在の実験モード名と表示用系列名を管理する。"""

FEDSDA_MODES = (
    "FedSDA_NoCached_ADWIN",
    "FedSDA_NoCached_ClassADWIN",
    "FedSDA_NoCached_ESR",
    "FedSDA_NoCached_ClassESR",
    "FedSDA_NoCached_HierarchicalClassESR",
    "FedSDA_NoCached_ClassESR_SoftRouting",
    "FedSDA_NoCached_ClassESR_RestartingSoftRouting",
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
        "HierarchicalClassESR", "HDDMA", "ClassHDDMA", "HDDMW",
    }
    return next(
        (part for part in mode.split("_") if part in detector_names),
        None,
    )


def is_adwin_mode(mode):
    return fedsda_detector_name(mode) in {"ADWIN", "ClassADWIN"}


def is_esr_mode(mode):
    return fedsda_detector_name(mode) in {
        "ESR", "ClassESR", "HierarchicalClassESR",
    }


def is_hddm_mode(mode):
    return fedsda_detector_name(mode) in {"HDDMA", "ClassHDDMA", "HDDMW"}
