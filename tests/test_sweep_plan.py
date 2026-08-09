from federated_drift_experiment.experiment_spec.configuration import AlgorithmOptions
from federated_drift_experiment.experiment_spec.sweep import (
    ADWIN_DELTA,
    AGGREGATION_INTERVAL,
    FEDSDA_DISTANCE_THRESHOLD,
    FEDDRIFT_DETECTION_BATCH_SIZE,
    FEDDRIFT_DISTANCE_THRESHOLD,
    create_sweep_plan,
)
from federated_drift_experiment import config


def _algorithm():
    return AlgorithmOptions(
        clustering_policy="on_new_model",
        clustering_decision="distance",
        detection_episodes=False,
        new_model_creation_policy="forward_persistent",
        fifo_size=30,
        new_model_validation_fraction=0.2,
        new_model_forward_validation_samples=10,
        shared_backbone_training="sequential",
        shared_backbone_routing_recalibration="none",
    )


def _plan():
    return create_sweep_plan(
        datasets=["sea4"], seeds=[0],
        fedsda_modes=["FedSDA_NoCached_ADWIN", "FedSDA_NoCached_ESR"],
        feddrift_modes=["FedDrift"], baseline_modes=["Oblivious"],
        concept_schedule="random", algorithm=_algorithm(),
        adwin_deltas=[0.05, 0.1], aggregation_intervals=[50, 100],
        feddrift_batches=[50], feddrift_deltas=[0.1],
        fixed_adwin=0.05, fixed_aggregation=50,
        fixed_fedsda_distance=0.1, fixed_feddrift_distance=0.1,
        fixed_feddrift_batch=50,
    )


def test_sweep_plan_separates_axes_from_resolved_runs():
    plan = _plan()
    runs = list(plan.iter_experiments())

    # ADWIN: δ掃引2 + A掃引2、ESR: A掃引2、FedDrift: 2掃引、baseline: 1。
    assert plan.run_count == len(runs) == 9
    assert all(run.algorithm == _algorithm() for run in runs)
    assert not any(
        run.mode.endswith("_ESR") and run.sweep_parameter == ADWIN_DELTA
        for run in runs
    )
    esr_aggregation = next(
        run for run in runs
        if run.mode.endswith("_ESR")
        and run.sweep_parameter == AGGREGATION_INTERVAL
    )
    assert esr_aggregation.parameter_value(ADWIN_DELTA) is None


def test_each_axis_carries_its_own_fixed_values():
    runs = list(_plan().iter_experiments())
    adwin_run = next(run for run in runs if run.sweep_parameter == ADWIN_DELTA)
    assert adwin_run.parameter_value(AGGREGATION_INTERVAL) == 50
    assert adwin_run.parameter_value(FEDSDA_DISTANCE_THRESHOLD) == 0.1

    aggregation_run = next(
        run for run in runs
        if run.mode.endswith("_ADWIN")
        and run.sweep_parameter == AGGREGATION_INTERVAL
    )
    assert aggregation_run.parameter_value(ADWIN_DELTA) == 0.05

    batch_run = next(
        run for run in runs
        if run.sweep_parameter == FEDDRIFT_DETECTION_BATCH_SIZE
    )
    assert batch_run.parameter_value(FEDDRIFT_DISTANCE_THRESHOLD) == 0.1
    delta_run = next(
        run for run in runs
        if run.sweep_parameter == FEDDRIFT_DISTANCE_THRESHOLD
    )
    assert delta_run.parameter_value(FEDDRIFT_DETECTION_BATCH_SIZE) == 50


def test_plan_description_exposes_axes_fixed_values_and_run_count():
    description = _plan().describe()
    assert "adwin_delta: [0.05, 0.1]" in description
    assert "aggregation_interval=50" in description
    assert "fedsda_distance_threshold=0.1" in description
    assert "total runs: 9" in description


def test_experiment_activation_is_scoped_and_restores_config():
    run = next(
        item for item in _plan().iter_experiments()
        if item.sweep_parameter == ADWIN_DELTA
    )
    before = (
        config.DATASET,
        config.CONCEPT_SCHEDULE,
        config.ADWIN_DELTA,
        config.AGGREGATION_INTERVAL,
        config.NEW_MODEL_CREATION_POLICY,
    )

    with run.activated():
        assert config.DATASET == "sea4"
        assert config.CONCEPT_SCHEDULE == "random"
        assert config.ADWIN_DELTA == 0.05
        assert config.AGGREGATION_INTERVAL == 50
        assert config.NEW_MODEL_CREATION_POLICY == "forward_persistent"

    assert (
        config.DATASET,
        config.CONCEPT_SCHEDULE,
        config.ADWIN_DELTA,
        config.AGGREGATION_INTERVAL,
        config.NEW_MODEL_CREATION_POLICY,
    ) == before


def test_experiment_activation_restores_config_after_error():
    run = next(iter(_plan().iter_experiments()))
    before_dataset = config.DATASET

    try:
        with run.activated():
            raise RuntimeError("test")
    except RuntimeError:
        pass

    assert config.DATASET == before_dataset
