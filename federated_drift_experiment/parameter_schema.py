"""コード・保存形式・CLI・論文表記を結ぶパラメータスキーマ。"""

from dataclasses import dataclass


PARAMETER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ParameterSpec:
    """一つの実験パラメータに対する各表現の対応。"""

    id: str
    code_name: str
    paper_symbol: str
    cli_name: str | None
    methods: tuple[str, ...]

    @property
    def csv_name(self):
        return self.id

    def format_assignment(self, value):
        return f"{self.paper_symbol}={value:g}"


PARAMETERS = (
    ParameterSpec(
        id="aggregation_interval",
        code_name="AGGREGATION_INTERVAL",
        paper_symbol="A",
        cli_name="aggregation-intervals",
        methods=("FedSDA", "Oblivious"),
    ),
    ParameterSpec(
        id="feddrift_detection_batch_size",
        code_name="FEDDRIFT_DETECTION_BATCH_SIZE",
        paper_symbol="B_detect",
        cli_name="feddrift-detection-batch-sizes",
        methods=("FedDrift",),
    ),
    ParameterSpec(
        id="fedsda_distance_threshold",
        code_name="FEDSDA_DISTANCE_THRESHOLD",
        paper_symbol="γ",
        cli_name="fedsda-distance-threshold",
        methods=("FedSDA",),
    ),
    ParameterSpec(
        id="feddrift_distance_threshold",
        code_name="FEDDRIFT_DISTANCE_THRESHOLD",
        paper_symbol="δ_FedDrift",
        cli_name="feddrift-distance-thresholds",
        methods=("FedDrift",),
    ),
    ParameterSpec(
        id="adwin_delta",
        code_name="ADWIN_DELTA",
        paper_symbol="δ_ADWIN",
        cli_name="adwin-deltas",
        methods=("FedSDA_ADWIN",),
    ),
    ParameterSpec(
        id="e_detector_alpha",
        code_name="E_DETECTOR_ALPHA",
        paper_symbol="α_e",
        cli_name=None,
        methods=("FedSDA_ESR",),
    ),
    ParameterSpec(
        id="hddm_drift_confidence",
        code_name="HDDM_DRIFT_CONFIDENCE",
        paper_symbol="δ_HDDM",
        cli_name=None,
        methods=("FedSDA_HDDM",),
    ),
    ParameterSpec(
        id="local_update_interval",
        code_name="LOCAL_UPDATE_INTERVAL",
        paper_symbol="τ",
        cli_name=None,
        methods=("FedSDA", "Oblivious"),
    ),
    ParameterSpec(
        id="updates_per_sample",
        code_name="UPDATES_PER_SAMPLE",
        paper_symbol="L",
        cli_name=None,
        methods=("FedSDA", "FedDrift", "Oblivious"),
    ),
)

PARAMETERS_BY_ID = {parameter.id: parameter for parameter in PARAMETERS}


def parameter(parameter_id):
    """正規IDからパラメータ定義を返す。"""
    try:
        return PARAMETERS_BY_ID[parameter_id]
    except KeyError as exc:
        raise KeyError(f"Unknown parameter id: {parameter_id}") from exc


def paper_symbol(parameter_id):
    return parameter(parameter_id).paper_symbol


def cli_option(parameter_id):
    name = parameter(parameter_id).cli_name
    return None if name is None else f"--{name}"
