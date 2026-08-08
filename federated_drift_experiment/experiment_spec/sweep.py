"""掃引軸と固定値から単一runのExperimentConfiguration列を生成する。"""

from dataclasses import dataclass

from .. import config
from .configuration import (
    AlgorithmOptions,
    ExperimentConfiguration,
    ParameterAssignment,
)
from ..mode_names import is_adwin_mode, is_esr_mode, is_hddm_mode


ADWIN_DELTA = "adwin_delta"
AGGREGATION_INTERVAL = "aggregation_interval"
FEDSDA_DISTANCE_THRESHOLD = "fedsda_distance_threshold"
FEDDRIFT_DETECTION_BATCH_SIZE = "feddrift_detection_batch_size"
FEDDRIFT_DISTANCE_THRESHOLD = "feddrift_distance_threshold"


@dataclass(frozen=True)
class SweepAxis:
    """変化させる一パラメータと、その掃引中に固定する値。"""

    parameter_id: str
    values: tuple[int | float, ...]
    applies_to: str
    fixed_values: tuple[ParameterAssignment, ...] = ()

    def applies_to_mode(self, mode):
        if self.applies_to == "fedsda_adwin":
            return mode.startswith("FedSDA_") and is_adwin_mode(mode)
        if self.applies_to == "fedsda":
            return mode.startswith("FedSDA_") and mode != "FedSDA_without_server"
        if self.applies_to == "feddrift":
            return mode == "FedDrift"
        return False


@dataclass(frozen=True)
class SweepPlan:
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    fedsda_modes: tuple[str, ...]
    feddrift_modes: tuple[str, ...]
    baseline_modes: tuple[str, ...]
    concept_schedule: str
    algorithm: AlgorithmOptions
    axes: tuple[SweepAxis, ...]
    baseline_parameters: tuple[ParameterAssignment, ...]

    def iter_experiments(self):
        for dataset in self.datasets:
            for seed in self.seeds:
                for mode in self.fedsda_modes:
                    for axis in self.axes:
                        if not axis.applies_to_mode(mode):
                            continue
                        for value in axis.values:
                            yield self._experiment(
                                mode, dataset, seed, axis, value,
                            )
                for mode in self.baseline_modes:
                    yield ExperimentConfiguration(
                        mode=mode, dataset=dataset, seed=seed,
                        concept_schedule=self.concept_schedule,
                        series=mode, sweep_parameter=None, sweep_value=None,
                        parameters=self.baseline_parameters,
                        algorithm=self.algorithm,
                    )
                for mode in self.feddrift_modes:
                    for axis in self.axes:
                        if not axis.applies_to_mode(mode):
                            continue
                        for value in axis.values:
                            yield self._experiment(
                                mode, dataset, seed, axis, value,
                            )

    def _experiment(self, mode, dataset, seed, axis, value):
        assignments = tuple(
            item for item in axis.fixed_values
            if item.parameter_id != axis.parameter_id
            and not (
                item.parameter_id == ADWIN_DELTA
                and not is_adwin_mode(mode)
            )
        ) + (ParameterAssignment(axis.parameter_id, value),)
        return ExperimentConfiguration(
            mode=mode, dataset=dataset, seed=seed,
            concept_schedule=self.concept_schedule,
            series=self._series(mode, axis),
            sweep_parameter=axis.parameter_id,
            sweep_value=value,
            parameters=assignments,
            algorithm=self.algorithm,
        )

    @staticmethod
    def _fixed(axis, parameter_id):
        return next(
            item.value for item in axis.fixed_values
            if item.parameter_id == parameter_id
        )

    def _series(self, mode, axis):
        if axis.parameter_id == ADWIN_DELTA:
            fixed_a = self._fixed(axis, AGGREGATION_INTERVAL)
            gamma = self._fixed(axis, FEDSDA_DISTANCE_THRESHOLD)
            return f"{mode} δ_ADWIN sweep (A={fixed_a}, γ={gamma})"
        if axis.parameter_id == AGGREGATION_INTERVAL:
            gamma = self._fixed(axis, FEDSDA_DISTANCE_THRESHOLD)
            if is_esr_mode(mode):
                detector = f"alpha_e={config.E_DETECTOR_ALPHA}"
            elif is_hddm_mode(mode):
                detector = f"confidence={config.HDDM_DRIFT_CONFIDENCE}"
            else:
                detector = f"δ_ADWIN={self._fixed(axis, ADWIN_DELTA)}"
            return f"{mode} A sweep ({detector}, γ={gamma})"
        if axis.parameter_id == FEDDRIFT_DETECTION_BATCH_SIZE:
            delta = self._fixed(axis, FEDDRIFT_DISTANCE_THRESHOLD)
            return f"{mode} B_detect sweep (δ_FedDrift={delta})"
        if axis.parameter_id == FEDDRIFT_DISTANCE_THRESHOLD:
            batch = self._fixed(axis, FEDDRIFT_DETECTION_BATCH_SIZE)
            return f"{mode} δ_FedDrift sweep (B_detect={batch})"
        raise ValueError(f"Unknown sweep parameter: {axis.parameter_id}")

    @property
    def run_count(self):
        return sum(1 for _ in self.iter_experiments())

    def describe(self):
        """実行せずに対象・掃引軸・固定値を確認できる文字列を返す。"""
        lines = [
            f"datasets: {', '.join(self.datasets) or '(none)'}",
            f"seeds: {', '.join(map(str, self.seeds)) or '(none)'}",
            f"FedSDA modes: {', '.join(self.fedsda_modes) or '(disabled)'}",
            f"FedDrift modes: {', '.join(self.feddrift_modes) or '(disabled)'}",
            f"baselines: {', '.join(self.baseline_modes) or '(disabled)'}",
            "sweep axes:",
        ]
        for axis in self.axes:
            if not axis.values:
                lines.append(f"  - {axis.parameter_id}: disabled")
                continue
            applicable_modes = tuple(
                mode for mode in self.fedsda_modes + self.feddrift_modes
                if axis.applies_to_mode(mode)
            )
            if not applicable_modes:
                lines.append(
                    f"  - {axis.parameter_id}: inactive (no applicable mode)"
                )
                continue
            fixed = ", ".join(
                f"{item.parameter_id}={item.value}" for item in axis.fixed_values
                if not (
                    item.parameter_id == ADWIN_DELTA
                    and not any(is_adwin_mode(mode) for mode in applicable_modes)
                )
            ) or "none"
            values = ", ".join(map(str, axis.values))
            lines.append(
                f"  - {axis.parameter_id}: [{values}] "
                f"(modes={', '.join(applicable_modes)}; fixed: {fixed})"
            )
        lines.append(f"total runs: {self.run_count}")
        return "\n".join(lines)


def create_sweep_plan(
    *, datasets, seeds, fedsda_modes, feddrift_modes, baseline_modes,
    concept_schedule, algorithm, adwin_deltas, aggregation_intervals,
    feddrift_batches, feddrift_deltas, fixed_adwin, fixed_aggregation,
    fixed_fedsda_distance, fixed_feddrift_distance, fixed_feddrift_batch,
):
    """CLI等で選ばれた掃引値を、固定値付きのSweepAxisへ解決する。"""
    axes = (
        SweepAxis(
            ADWIN_DELTA, tuple(adwin_deltas), "fedsda_adwin",
            (
                ParameterAssignment(AGGREGATION_INTERVAL, fixed_aggregation),
                ParameterAssignment(FEDSDA_DISTANCE_THRESHOLD, fixed_fedsda_distance),
            ),
        ),
        SweepAxis(
            AGGREGATION_INTERVAL, tuple(aggregation_intervals), "fedsda",
            (
                ParameterAssignment(ADWIN_DELTA, fixed_adwin),
                ParameterAssignment(FEDSDA_DISTANCE_THRESHOLD, fixed_fedsda_distance),
            ),
        ),
        SweepAxis(
            FEDDRIFT_DETECTION_BATCH_SIZE, tuple(feddrift_batches), "feddrift",
            (ParameterAssignment(
                FEDDRIFT_DISTANCE_THRESHOLD, fixed_feddrift_distance,
            ),),
        ),
        SweepAxis(
            FEDDRIFT_DISTANCE_THRESHOLD, tuple(feddrift_deltas), "feddrift",
            (ParameterAssignment(
                FEDDRIFT_DETECTION_BATCH_SIZE, fixed_feddrift_batch,
            ),),
        ),
    )
    return SweepPlan(
        datasets=tuple(datasets), seeds=tuple(seeds),
        fedsda_modes=tuple(fedsda_modes), feddrift_modes=tuple(feddrift_modes),
        baseline_modes=tuple(baseline_modes), concept_schedule=concept_schedule,
        algorithm=algorithm, axes=axes,
        baseline_parameters=(
            ParameterAssignment(ADWIN_DELTA, config.ADWIN_DELTA),
            ParameterAssignment(AGGREGATION_INTERVAL, config.AGGREGATION_INTERVAL),
        ),
    )
