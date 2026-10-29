"""IMPROVE-033の初期値・seed・選択との結び付けと、小さい全体runの照合。"""

import dataclasses
import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.runtime import fedsda_experiment_run as product
from federated_learning_experiments.runtime.fedsda_run_record_tables import (
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)
from tests.refactoring.test_alarm_interval_minimum_description_length_selection import (
    OPTIMIZER_SETTINGS,
    CurrentTrainingModelAssignment,
    ModelAndClassLossStatisticsStore,
    _assert_random_states_equal,
    _observations,
    _random_states,
    _registry,
)
from tools.research_provisional_candidate_expert import (
    candidate_initialization_follows_learner as policy,
)
from tools.research_provisional_candidate_expert import run as research

REVERSE = policy.EVERY_SAMPLE_WITH_REVERSE_PREQUENTIAL_CODE_SEARCH_NAME
FORWARD = policy.MINIMUM_INTERVAL_MULTIPLES_WITH_FORWARD_PREQUENTIAL_CODE_SEARCH_NAME
REINITIALIZED = policy.selection.REINITIALIZED_LEARNER_NAME
WARM = policy.selection.WARM_STARTED_LEARNER_NAME


@pytest.fixture
def harness(monkeypatch):
    registry = _registry(class_count=4, model_ids=(4, 9))
    observations = _observations(count=15, class_count=4)
    select_arguments = dict(
        client_id=1,
        alarm_sample_index=observations[-1].sample_index,
        pending_sample_observations=observations,
        held_model_training_state_registry=registry,
        loss_statistics_store=ModelAndClassLossStatisticsStore(),
        current_training_model_assignment=CurrentTrainingModelAssignment(initial_model_id=9),
        parameter_optimizer_settings=OPTIMIZER_SETTINGS,
        switching_probability=1 / 120,
        change_interval_grid_sample_count=5,
        new_model_hypothesis_learners="reinitialized_and_warm_started_learners",
        change_interval_search=REVERSE,
    )
    snapshots = {
        s.model_id: snapshot_classifier_parameters(classifier=s.classifier)
        for s in registry.snapshot_ordered_held_model_training_states()
    }
    init_arguments = dict(
        settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="lowest_evaluated_mean_loss_model"
        ),
        available_parameter_snapshots_by_model_id=snapshots,
        current_training_model_id=9,
        evaluated_mean_losses_by_model_id=((4, 0.1), (9, 0.3)),
    )
    real_select = policy.processing.select_alarm_interval_assignment
    chosen_learner = [REINITIALIZED]
    hypothesis = ["new_model"]
    selected_count = [10]
    result_override = [None]

    def select(**arguments):
        # 符号長の学習・捕捉は実物。選択の分岐だけ固定して、両学習者と各seed経路を必ず通す。
        result = real_select(**arguments)
        return result_override[0] or dataclasses.replace(
            result,
            selected_hypothesis=hypothesis[0],
            destination_model_id=None,
            change_interval_sample_count=selected_count[0],
            new_model_learner_name=chosen_learner[0]
            if hypothesis[0] == "new_model"
            else policy.selection.NO_NEW_MODEL_LEARNER_NAME,
        )

    def train(**arguments):
        return SimpleNamespace(
            candidate_epoch_training_result=SimpleNamespace(completed_epoch_count=7)
        )

    def resolve(**arguments):
        if arguments["alarm_interval_assignment_selection"].selected_hypothesis != "new_model":
            return None
        value = policy.resolution.select_candidate_initial_parameter_snapshot(**init_arguments)
        policy.resolution.train_candidate_on_alarm_change_interval()
        return value

    monkeypatch.setattr(policy.processing, "select_alarm_interval_assignment", select)
    monkeypatch.setattr(policy.response, "resolve_alarm_change_interval", resolve)
    monkeypatch.setattr(policy.resolution, "train_candidate_on_alarm_change_interval", train)

    def resolve_arguments(chosen, **updates):
        return (
            dict(
                alarm_interval_assignment_selection=chosen,
                proposal_sample_index=select_arguments["alarm_sample_index"],
                change_interval_training_samples=tuple(
                    o.training_sample for o in observations[-selected_count[0] :]
                ),
                held_model_training_state_registry=registry,
                parameter_optimizer_settings=OPTIMIZER_SETTINGS,
            )
            | updates
        )

    return SimpleNamespace(
        select=select_arguments,
        initialize=init_arguments,
        resolve=resolve_arguments,
        learner=chosen_learner,
        hypothesis=hypothesis,
        count=selected_count,
        registry=registry,
        snapshots=snapshots,
        result_override=result_override,
    )


@pytest.mark.parametrize("search", [REVERSE, FORWARD])
@pytest.mark.parametrize("count", [5, 10])
@pytest.mark.parametrize("learner", [REINITIALIZED, WARM])
@pytest.mark.parametrize("variant", policy.CANDIDATE_INITIALIZATION_VARIANTS)
def test_initial_values_seed_paths_rng_and_restoration(harness, search, count, learner, variant):
    h = harness
    h.select["change_interval_search"] = search
    h.learner[0] = learner
    h.count[0] = count
    originals = (
        policy.processing.select_alarm_interval_assignment,
        policy.response.resolve_alarm_change_interval,
        policy.resolution.select_candidate_initial_parameter_snapshot,
        policy.resolution.train_candidate_on_alarm_change_interval,
        policy.training._compute_reinitialized_learner_prequential_code_length,
        policy.training._accumulate_prequential_code_length,
    )
    diagnostics = {}
    before = _random_states()
    with policy.candidate_initialization_policy(variant=variant, diagnostics=diagnostics):
        chosen = policy.processing.select_alarm_interval_assignment(**h.select)
        actual = policy.response.resolve_alarm_change_interval(**h.resolve(chosen))
    _assert_random_states_equal(before, _random_states())
    assert originals == (
        policy.processing.select_alarm_interval_assignment,
        policy.response.resolve_alarm_change_interval,
        policy.resolution.select_candidate_initial_parameter_snapshot,
        policy.resolution.train_candidate_on_alarm_change_interval,
        policy.training._compute_reinitialized_learner_prequential_code_length,
        policy.training._accumulate_prequential_code_length,
    )
    (entry,) = diagnostics["candidates"]
    assert entry["completed_epoch_count"] == 7
    assert entry["lowest_evaluated_mean_loss_model_id"] == 4
    expected_seed = (
        policy.mdl.derive_reinitialized_learner_single_pass_initialization_seed(
            client_id=1, alarm_sample_index=h.select["alarm_sample_index"]
        )
        if search == REVERSE
        else policy.mdl.derive_reinitialized_learner_initialization_seed(
            client_id=1,
            alarm_sample_index=h.select["alarm_sample_index"],
            change_interval_sample_count=count,
        )
    )
    assert entry["reinitialized_learner_initialization_seed"] == expected_seed
    reinitialized = (
        variant == policy.CANDIDATE_INITIALIZATION_VARIANTS[0] and learner == REINITIALIZED
    )
    if reinitialized:
        assert entry["initialization_seed"] == expected_seed
        assert entry["matches_selected_learner_initial_parameters"] is True
        assert (
            entry["concept_specific_parameter_digest"]
            == entry["reinitialized_learner_initial_parameter_digest"]
        )
        assert entry["initialization_kind"] == "reinitialized_concept_specific_part"
        assert any(
            not actual[k].equal(v)
            for k, v in h.snapshots[9].items()
            if not k.startswith("feature_extractor.")
        )
    else:
        assert entry["initialization_seed"] is None
        assert entry["initialization_kind"] == "current_model_copy"
        assert policy._equal(actual, h.snapshots[9])
    for name, value in actual.items():
        assert (
            value.shape == h.snapshots[9][name].shape and value.dtype == h.snapshots[9][name].dtype
        )
        assert value.data_ptr() != h.snapshots[9][name].data_ptr()
        if name.startswith("feature_extractor."):
            assert value.equal(h.snapshots[9][name])
    assert policy._equal(
        h.snapshots[9],
        snapshot_classifier_parameters(
            classifier=h.registry.get_held_model_training_state(model_id=9).classifier
        ),
    )


@pytest.mark.parametrize("search", [REVERSE, FORWARD])
@pytest.mark.parametrize("second_learner", [REINITIALIZED, WARM])
def test_interleaved_clients_use_their_own_selection_learner_and_seed(
    harness, search, second_learner
):
    h = harness
    h.select["change_interval_search"] = search
    diagnostics = {}
    with policy.candidate_initialization_policy(
        variant=policy.CANDIDATE_INITIALIZATION_VARIANTS[0], diagnostics=diagnostics
    ):
        a = policy.processing.select_alarm_interval_assignment(**h.select)
        h.learner[0] = second_learner
        b = policy.processing.select_alarm_interval_assignment(**(h.select | {"client_id": 2}))
        policy.response.resolve_alarm_change_interval(**h.resolve(a))
        policy.response.resolve_alarm_change_interval(**h.resolve(b))
    entries = diagnostics["candidates"]
    assert [e["client_id"] for e in entries] == [1, 2]
    assert [e["new_model_learner_name"] for e in entries] == [REINITIALIZED, second_learner]
    assert [e["initialization_kind"] for e in entries] == [
        "reinitialized_concept_specific_part",
        "reinitialized_concept_specific_part"
        if second_learner == REINITIALIZED
        else "current_model_copy",
    ]
    assert (
        entries[0]["reinitialized_learner_initialization_seed"]
        != entries[1]["reinitialized_learner_initialization_seed"]
    )


@pytest.mark.parametrize(
    "failure",
    [
        "different_object",
        "different_alarm",
        "outside",
        "pending",
        "duplicate",
        "wrong_length",
        "mismatch",
    ],
)
def test_binding_failures_raise_and_restore(harness, monkeypatch, failure):
    h = harness
    original_select = policy.processing.select_alarm_interval_assignment
    original_init = policy.resolution.select_candidate_initial_parameter_snapshot
    expected_messages = {
        "different_object": "解決に渡された選択が記憶した選択と一致しません",
        "different_alarm": "解決に渡された警報の標本番号が一致しません",
        "outside": "候補の初期化は選択と対応した解決の中だけで実行できます",
        "pending": "contextの終了時に未解決の新しいモデルの選択が残っています",
        "duplicate": "未解決の選択を上書きできません",
        "wrong_length": "解決に渡された変化区間の長さが一致しません",
        "mismatch": "候補とMDL学習者の最初の概念固有部が一致しません",
    }
    with (
        pytest.raises(ValueError, match=expected_messages[failure]),
        policy.candidate_initialization_policy(
            variant=policy.CANDIDATE_INITIALIZATION_VARIANTS[0], diagnostics={}
        ),
    ):
        if failure == "outside":
            policy.resolution.select_candidate_initial_parameter_snapshot(**h.initialize)
        chosen = policy.processing.select_alarm_interval_assignment(**h.select)
        if failure == "pending":
            pass
        elif failure == "duplicate":
            h.result_override[0] = chosen
            policy.processing.select_alarm_interval_assignment(**h.select)
        else:
            updates = {}
            if failure == "different_object":
                chosen = dataclasses.replace(chosen)
            if failure == "different_alarm":
                updates["proposal_sample_index"] = h.select["alarm_sample_index"] + 1
            if failure == "wrong_length":
                updates["change_interval_training_samples"] = ()
            if failure == "mismatch":
                monkeypatch.setattr(
                    policy,
                    "reinitialized_candidate_parameter_snapshot",
                    lambda **kw: h.snapshots[9],
                )
            policy.response.resolve_alarm_change_interval(**h.resolve(chosen, **updates))
    assert policy.processing.select_alarm_interval_assignment is original_select
    assert policy.resolution.select_candidate_initial_parameter_snapshot is original_init


def test_non_new_model_selections_are_not_retained(harness):
    h = harness
    h.hypothesis[0] = "current_model"
    with policy.candidate_initialization_policy(
        variant=policy.CANDIDATE_INITIALIZATION_VARIANTS[0], diagnostics={}
    ):
        chosen = policy.processing.select_alarm_interval_assignment(**h.select)
        assert policy.response.resolve_alarm_change_interval(**h.resolve(chosen)) is None


def _product_summary(run):
    tables = build_fedsda_run_record_tables(experiment_run=run)
    clients = run.measured_run.participants.client_operations
    records = [
        r
        for c in clients
        for r in c.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
    ]
    repository = run.measured_run.participants.server_operations.owners.global_model_repository
    volume = run.run_metrics.communication_volume
    return dict(
        accuracy=run.run_metrics.prediction_accuracy,
        uploaded_bytes=volume.uploaded_byte_count,
        downloaded_bytes=volume.downloaded_byte_count,
        communication_volume=dataclasses.asdict(volume),
        prediction_record_digest=hashlib.sha256(
            json.dumps([r.combined_prediction_is_correct for r in records]).encode()
        ).hexdigest(),
        server_parameter_digest=research._digest(
            {
                f"{mid}.{k}": v
                for mid in repository.global_model_ids
                for k, v in repository.get_global_model_parameters(model_id=mid).items()
            }
        ),
        model_computation_counts=dataclasses.asdict(run.run_metrics.model_computation_counts),
        alarm_interval_assignment_selection={
            k: v.tolist()
            for k, v in tables.items()
            if k.startswith("alarm_interval_assignment_selection.")
        },
        candidate_adoption_count=sum(
            v == "alarm_interval_candidate_adopted_as_new_model"
            for v in tables["adaptation.adaptation_outcome"]
        ),
    )


@pytest.fixture(scope="module", params=["sine2", "mnist4"])
def small_runs(request):
    dataset = request.param
    job = dict(dataset=dataset, interval=50, seed=0, sample_count=1000, check=False)
    results = {
        v: research._execute_job(job | {"variant": v})
        for v in ("product-default", *policy.CANDIDATE_INITIALIZATION_VARIANTS)
    }
    settings = build_experiment_run_settings(
        configuration_name="fedsda",
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset,
            random_seed=0,
            client_count=10,
            per_client_sample_count=1000,
            server_aggregation_interval_per_client_samples=50,
        ),
    )
    plain = _product_summary(product.execute_fedsda_experiment_run(run_settings=settings))
    return dataset, results, plain


def test_small_runs_product_default_exactly_matches_unpatched_product(small_runs):
    dataset, results, plain = small_runs
    json.dumps(results)
    for name, expected in plain.items():
        assert results["product-default"][name] == expected, name
    assert plain["candidate_adoption_count"] > 0
    assert len(plain["alarm_interval_assignment_selection"]) >= 8
    baseline = results["product-default"]
    print(
        json.dumps(
            dict(
                dataset=dataset,
                variant="product-default",
                exact_product_match=True,
                accuracy=baseline["accuracy"],
                uploaded_bytes=baseline["uploaded_bytes"],
                downloaded_bytes=baseline["downloaded_bytes"],
                candidate_adoption_count=baseline["candidate_adoption_count"],
                model_computation_counts=baseline["model_computation_counts"],
                seconds=baseline["seconds"],
            )
        )
    )


def test_small_runs_variants_complete_and_report_initialization_counts(small_runs):
    dataset, results, _ = small_runs
    for variant in policy.CANDIDATE_INITIALIZATION_VARIANTS:
        result = results[variant]
        entries = result["candidate_initialization"]["candidates"]
        counts = dict(Counter(e["initialization_kind"] for e in entries))
        print(
            json.dumps(
                dict(
                    dataset=dataset,
                    variant=variant,
                    initialization_counts=counts,
                    accuracy=result["accuracy"],
                    uploaded_bytes=result["uploaded_bytes"],
                    downloaded_bytes=result["downloaded_bytes"],
                    model_computation_counts=result["model_computation_counts"],
                    seconds=result["seconds"],
                )
            )
        )
        assert len(entries) == result["candidate_adoption_count"] > 0
        assert all(e["completed_epoch_count"] is not None for e in entries)
        if variant == policy.CANDIDATE_INITIALIZATION_VARIANTS[0]:
            assert all(e["matches_selected_learner_initial_parameters"] for e in entries)
            missing = {"reinitialized_concept_specific_part", "current_model_copy"} - counts.keys()
            if missing:
                print(
                    f"未出現の初期値 {sorted(missing)}: {dataset}, interval=50, seed=0, sample_count=1000, clients=10"
                )
