"""正解率を受け取るDriftSurfの役割と時刻窓の状態遷移。"""

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True, kw_only=True)
class DriftSurfSettings:
    """δを設定し、反応3時刻・窓10時刻を固定する。"""

    accuracy_drop_threshold: float = field(
        default=0.10, metadata={"omit_default_from_saved_settings": True}
    )

    def __post_init__(self) -> None:
        if (
            type(self.accuracy_drop_threshold) is not float
            or not isfinite(self.accuracy_drop_threshold)
            or self.accuracy_drop_threshold <= 0
        ):
            raise ValueError("DriftSurfのδは正で有限のbuiltin floatが必要です。")


@dataclass(frozen=True, kw_only=True)
class DriftSurfTransition:
    """runtimeが実行する役割の作り直しと昇格。"""

    reset_roles: tuple[str, ...]
    entered_reactive: bool
    promoted: bool


class DriftSurfState:
    """公式の状態・窓を所有する。モデルの実体とIDはruntimeが所有する。"""

    def __init__(self, *, settings: DriftSurfSettings) -> None:
        if type(settings) is not DriftSurfSettings:
            raise TypeError("exact DriftSurfSettingsが必要です。")
        settings.__post_init__()
        self.delta = settings.accuracy_drop_threshold
        self.state = "stab"
        self.train_data_dict: dict[str, list[int] | None] = {"pred": [0], "stab": [0], "reac": None}
        self.train_keys = ["pred", "stab"]
        self.model_key = "pred"
        self.acc_best = 0.0
        self.reac_ctr: int | None = None
        self.acc_dict: dict[str, list[float]] | None = None
        self._last_timestep = 0

    @property
    def scoring_roles(self) -> tuple[str, ...]:
        """公式がこの状態で_scoreを呼ぶ役割だけ。"""
        return (
            ("pred",)
            + (("stab",) if self.state == "stab" and self.train_data_dict["stab"] else ())
            + (("reac",) if self.state == "reac" and self.reac_ctr else ())
        )

    def _append(self, role: str, timestep: int) -> None:
        window = self.train_data_dict[role]
        assert window is not None
        window.append(timestep)
        if len(window) > 10:
            window.pop(0)

    def advance(self, *, timestep_index: int, accuracies: dict[str, float]) -> DriftSurfTransition:
        """次の時刻へ進める。未計測の役割は要求しない。"""
        if type(timestep_index) is not int or timestep_index != self._last_timestep + 1:
            raise ValueError("時刻は1から連続している必要があります。")
        for role in self.scoring_roles:
            if (
                role not in accuracies
                or not isfinite(accuracies[role])
                or not 0 <= accuracies[role] <= 1
            ):
                raise ValueError("必要な役割の正解率は有限の0〜1が必要です。")
        pred = accuracies["pred"]
        self.acc_best = max(self.acc_best, pred)
        resets = []
        entered = promoted = False
        if self.state == "stab":
            stab = accuracies["stab"] if self.train_data_dict["stab"] else 0.0
            if pred < self.acc_best - self.delta or pred < stab - self.delta / 2:
                self.state = "reac"
                self.train_data_dict["reac"] = []
                resets.append("reac")
                entered = True
                self.reac_ctr = 0
                self.acc_dict = {"pred": [0.0] * 3, "reac": [0.0] * 3}
            else:
                self._append("pred", timestep_index)
                self._append("stab", timestep_index)
                self.train_keys = ["pred", "stab"]
        if self.state == "reac":
            assert self.reac_ctr is not None and self.acc_dict is not None
            if self.reac_ctr > 0:
                reac = accuracies["reac"]
                self.acc_dict["pred"][self.reac_ctr - 1] = pred
                self.acc_dict["reac"][self.reac_ctr - 1] = reac
                self.model_key = "reac" if reac > pred else "pred"
            self._append("pred", timestep_index)
            self._append("reac", timestep_index)
            self.train_keys = ["pred", "reac"]
            self.reac_ctr += 1
            if self.reac_ctr == 3:
                self.state = "stab"
                self.train_data_dict["stab"] = []
                resets.append("stab")
                if sum(self.acc_dict["pred"]) / 3 < sum(self.acc_dict["reac"]) / 3:
                    self.train_data_dict["pred"] = self.train_data_dict["reac"]
                    self.acc_best = max(self.acc_dict["reac"])
                    promoted = True
                self.model_key = "pred"
                self.acc_dict = None
                self.reac_ctr = None
        self._last_timestep = timestep_index
        return DriftSurfTransition(
            reset_roles=tuple(resets), entered_reactive=entered, promoted=promoted
        )
