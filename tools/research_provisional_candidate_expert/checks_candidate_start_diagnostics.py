"""13候補の隔離、製品との完全一致、生成器のラベルを検査する。"""

import json
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.refactoring.test_alarm_interval_minimum_description_length_selection import (
    OPTIMIZER_SETTINGS,
    _assert_random_states_equal,
    _random_states,
    _registry,
)
from tools.research_provisional_candidate_expert import (
    candidate_start_diagnostics as diagnostic,
)
from tools.research_provisional_candidate_expert import run as research


@pytest.mark.parametrize("dataset", ["sine2", "mnist4", "rotmnist2"])
def test_small_run_exact_product_trajectory(dataset):
    job = dict(dataset=dataset, interval=50, seed=0, sample_count=1000, check=False)
    baseline = research._execute_job(job | dict(variant="product-default"))
    baseline_random_states = _random_states()
    original = (
        diagnostic.processing.select_alarm_interval_assignment,
        diagnostic.response.resolve_alarm_change_interval,
        diagnostic.resolution.train_candidate_on_alarm_change_interval,
        diagnostic.streams.build_client_observed_streams,
    )
    measured = research._execute_job(job | dict(variant=diagnostic.VARIANT))
    _assert_random_states_equal(baseline_random_states, _random_states())
    assert original == (
        diagnostic.processing.select_alarm_interval_assignment,
        diagnostic.response.resolve_alarm_change_interval,
        diagnostic.resolution.train_candidate_on_alarm_change_interval,
        diagnostic.streams.build_client_observed_streams,
    )
    for key in baseline.keys() - {"variant", "seconds"}:
        assert measured[key] == baseline[key], key
    alarms = measured["candidate_start_diagnostics"]["alarms"]
    assert len(alarms) == measured["candidate_adoption_count"] > 0
    for alarm in alarms:
        candidates = alarm["candidates"]
        assert len(candidates) == 13
        assert (
            candidates["A-T30"]["parameter_digest"] == alarm["product_candidate_digest"]
        )
        assert alarm["c_initial_digest"] == alarm["learner_initial_digest"]
        for learner in alarm["learners"]:
            assert sum(learner["losses"]) == learner["product_code_length"]
            assert learner["concepts"] == list(reversed(alarm["interval_concepts"]))
            assert (
                len(learner["labels"])
                == len(learner["losses"])
                == alarm["change_interval_sample_count"]
            )
        assert len(set(alarm["evaluation_seeds"].values())) == 3
        for name, candidate in candidates.items():
            assert (
                sum(c["count"] for c in candidate["static"]["class_counts"].values())
                == 500
            )
            assert len(candidate["sequential"]["cumulative_log_loss"]) == 200
            assert candidate["best_validation_loss"] is not None
            assert candidate["last_training_loss"] is not None
            assert (
                candidate["training_sample_indices"]
                == candidates["A-T30"]["training_sample_indices"]
            )
            if "E100" in name:
                assert candidate["completed_epoch_count"] == 100
            if "E300" in name:
                assert candidate["completed_epoch_count"] == 300
            if "S4" in name:
                counts = Counter(
                    alarm["interval_concepts"][i]
                    for i in candidate["training_sample_indices"]
                )
                assert candidate["added_sample_counts_by_concept"] == {
                    c: n * 3 for c, n in counts.items()
                }
                assert candidate["added_sample_count"] == sum(counts.values()) * 3
            else:
                assert candidate["added_sample_count"] == 0
    print(
        json.dumps(
            dict(
                dataset=dataset,
                alarms=len(alarms),
                baseline_seconds=baseline["seconds"],
                diagnostic_seconds=measured["seconds"],
                increase_seconds=measured["seconds"] - baseline["seconds"],
                accuracy=measured["accuracy"],
                uploaded_bytes=measured["uploaded_bytes"],
                downloaded_bytes=measured["downloaded_bytes"],
                model_computation_counts=measured["model_computation_counts"],
            )
        )
    )


@pytest.mark.parametrize("dataset", ["sine2", "mnist4", "rotmnist2"])
def test_sample_generation_deterministic_and_isolated(dataset):
    arguments = dict(
        dataset=dataset,
        run_seed=0,
        client_id=3,
        alarm_sample_index=701,
        purpose="static-evaluation",
        concepts=[0, 1] * 10,
    )
    before = _random_states()
    first = diagnostic._generate_samples(**arguments)
    _assert_random_states_equal(before, _random_states())
    second = diagnostic._generate_samples(**arguments)
    assert all(a.equal(b) for a, b in zip(first[:2], second[:2], strict=True))
    assert first[2] == second[2]
    other = diagnostic._generate_samples(**(arguments | dict(purpose="s4-training")))
    assert other[2] != first[2]
    assert not other[0].equal(first[0])
    _assert_random_states_equal(before, _random_states())


@pytest.mark.parametrize("dataset", ["mnist4", "fmnist4"])
@pytest.mark.parametrize("previous,current", [(0, 1), (1, 2), (2, 2), (3, 0)])
def test_changed_classes_match_generator_labels(dataset, previous, current):
    # 同じ画像列を2概念で生成し、ラベルの対応から定義を独立に検査する。
    arguments = dict(
        dataset=dataset,
        run_seed=0,
        client_id=0,
        alarm_sample_index=0,
        purpose="label-contract",
    )
    x, p, _ = diagnostic._generate_samples(**arguments, concepts=[previous] * 500)
    y, q, _ = diagnostic._generate_samples(**arguments, concepts=[current] * 500)
    assert torch.equal(x, y)
    changed = set(p[p != q].long().tolist())
    assert changed == set(diagnostic._changed_classes(dataset, previous, current, 10))
    assert set(p.reshape(-1).long().tolist()) == set(range(10))


def test_patch_restores_on_failure():
    original = diagnostic.processing.select_alarm_interval_assignment
    with pytest.raises(ValueError, match="実体"):
        with diagnostic.candidate_start_diagnostics(
            dataset="sine2", run_seed=0, diagnostics={}
        ):
            diagnostic.response.resolve_alarm_change_interval(
                alarm_interval_assignment_selection=None
            )
    assert diagnostic.processing.select_alarm_interval_assignment is original


@pytest.mark.parametrize("message", ["A・T30と製品", "Cと学習者", "学習者の符号長"])
def test_mismatch_guard_rejects_different_values(message):
    with pytest.raises(ValueError, match=message):
        diagnostic._require_equal("corrupted", "expected", message)


def test_owner_digest_detects_tensor_and_scalar_mutations():
    value = dict(parameters=torch.ones(2), nested=dict(count=1), array=np.ones(2))
    before = diagnostic._state_digest(value)
    value["parameters"][0] += 1
    assert before != diagnostic._state_digest(value)


def test_held_concept_evidence_pools_clients_and_keeps_ties_unknown():
    def client(counts):
        return SimpleNamespace(
            get_model_assigned_sample_concept_counts=lambda *, model_id: counts.get(
                model_id, {}
            )
        )

    clients = [client({4: {1: 3}, 9: {1: 2}}), client({4: {2: 5}, 9: {2: 2}})]
    counts, concepts, unknown, present = diagnostic._held_concept_evidence(
        clients, (0, 4, 9), 2
    )
    assert counts[4] == {1: 3, 2: 5}
    assert concepts == {0: 0, 4: 2}
    assert unknown == [9]
    assert present is True
    assert diagnostic._held_concept_evidence(clients, (0, 4, 9), 3)[-1] is None
    assert diagnostic._held_concept_evidence(clients, (0, 4), 3)[-1] is False
    clients[0].client_id = 0
    clients[1].client_id = 1
    clients[0].get_model_assigned_sample_concept_counts = lambda **kw: {1: 2}
    clients[1].get_model_assigned_sample_concept_counts = lambda **kw: {2: 20}
    assert (
        diagnostic._held_concept_evidence(clients, (-101,), 1, local_client_id=0)[-1]
        is True
    )
    assert (
        diagnostic._held_concept_evidence(clients, (-101,), 1, local_client_id=1)[-1]
        is False
    )


def test_diagnostic_gate_leaves_computation_meter_unchanged():
    registry = _registry(class_count=4, model_ids=(9,))
    state = registry.get_held_model_training_state(model_id=9)
    features = torch.ones(2, 3)
    meter_type = diagnostic.computation.ModelComputationMeter
    original = meter_type._observe_forward_start
    with diagnostic._model_computation_gate() as paused:
        with diagnostic.computation.measure_model_computation() as meter:
            state.classifier(features)
            before = meter.get_model_computation_counts()
            references = dict(meter._observed_shared_parameter_references)
            paused[0] = True
            state.classifier(features)
            state.concept_specific_parameter_optimizer_state.parameter_optimizer.step()
            assert meter.get_model_computation_counts() == before
            assert meter._observed_shared_parameter_references == references
            assert meter._shared_part_forward_depth == 0
            paused[0] = False
            state.classifier(features)
            assert (
                meter.get_model_computation_counts().shared_part_training_example_count
                == 2 * before.shared_part_training_example_count
            )
    assert meter_type._observe_forward_start is original


@pytest.fixture
def guarded_alarm(monkeypatch):
    """選択の周囲だけを小さく組み、実物の学習者と各照合を通す。"""
    registry = _registry(class_count=4, model_ids=(9,))
    monkeypatch.setattr(
        diagnostic.streams, "build_client_observed_streams", lambda **kw: ()
    )
    classifier = registry.get_held_model_training_state(model_id=9).classifier
    initial = diagnostic.initialization.snapshot_classifier_parameters(
        classifier=classifier
    )
    select_arguments = dict(
        client_id=1,
        alarm_sample_index=2,
        change_interval_grid_sample_count=1,
        change_interval_search=diagnostic.initialization.EVERY_SAMPLE_WITH_REVERSE_PREQUENTIAL_CODE_SEARCH_NAME,
        pending_sample_observations=tuple(
            SimpleNamespace(
                sample_index=i,
                training_sample=SimpleNamespace(
                    observed_class_labels=torch.zeros(1, 1)
                ),
                observed_concept_id=0,
            )
            for i in range(3)
        ),
    )
    reinitialized = (
        diagnostic.initialization.reinitialized_candidate_parameter_snapshot(
            classifier=classifier,
            initialization_seed=diagnostic.initialization._seed(
                client_id=1,
                alarm_sample_index=2,
                count=3,
                search=select_arguments["change_interval_search"],
            ),
            optimizer_settings=OPTIMIZER_SETTINGS,
        )
    )

    def select(**kwargs):
        prefixes_by_name = []
        for name, snapshot in [
            (diagnostic.selection.REINITIALIZED_LEARNER_NAME, reinitialized),
            (diagnostic.selection.WARM_STARTED_LEARNER_NAME, initial),
        ]:
            learner = deepcopy(classifier)
            learner.load_state_dict(snapshot)
            prefixes = []
            diagnostic.training._accumulate_prequential_code_length(
                learner=learner,
                input_features=torch.ones(
                    3, classifier.feature_extractor.input_feature_count
                ),
                observed_class_labels=torch.zeros(3, 1),
                parameter_optimizer_settings=OPTIMIZER_SETTINGS,
                prefixes=prefixes,
            )
            prefixes_by_name.append((name, prefixes))
        choices = {
            k: diagnostic.mdl.select_new_model_learner(
                prequential_code_lengths_by_learner_name=tuple(
                    (name, values[k - 1]) for name, values in prefixes_by_name
                )
            )
            for k in range(1, 4)
        }
        return diagnostic.mdl.select_assignment_by_minimum_description_length(
            current_model_log_losses=(100.0,) * 3,
            held_model_log_losses_by_model_id=(),
            new_model_code_lengths_by_change_interval_sample_count={
                k: v.description_length for k, v in choices.items()
            },
            new_model_learner_names_by_change_interval_sample_count={
                k: v.learner_name for k, v in choices.items()
            },
            switching_code_length=0.0,
            change_interval_grid_sample_count=1,
            change_interval_step_sample_count=1,
        )

    def resolve(**kwargs):
        diagnostic.resolution.select_candidate_initial_parameter_snapshot(
            settings=None,
            available_parameter_snapshots_by_model_id={9: initial},
            current_training_model_id=9,
            evaluated_mean_losses_by_model_id=(),
        )
        return diagnostic.resolution.train_candidate_on_alarm_change_interval(
            initial_candidate_parameter_snapshot=initial,
            parameter_optimizer_settings=OPTIMIZER_SETTINGS,
        )

    monkeypatch.setattr(
        diagnostic.processing, "select_alarm_interval_assignment", select
    )
    monkeypatch.setattr(diagnostic.response, "resolve_alarm_change_interval", resolve)
    monkeypatch.setattr(
        diagnostic.resolution,
        "select_candidate_initial_parameter_snapshot",
        lambda **kw: initial,
    )
    monkeypatch.setattr(
        diagnostic.resolution,
        "train_candidate_on_alarm_change_interval",
        lambda **kw: SimpleNamespace(
            candidate_training_state=SimpleNamespace(
                candidate_classifier=deepcopy(classifier)
            )
        ),
    )
    return select_arguments, dict(
        proposal_sample_index=2,
        change_interval_training_samples=(None,) * 3,
        change_interval_sample_concept_ids=(0,) * 3,
        held_model_training_state_registry=registry,
        parameter_optimizer_settings=OPTIMIZER_SETTINGS,
    )


@pytest.mark.parametrize("failure", ["C", "learner_length", "product", "owner"])
def test_runtime_guards_reject_corrupted_alarm(guarded_alarm, monkeypatch, failure):
    select_arguments, resolve_arguments = guarded_alarm
    original = diagnostic.processing.select_alarm_interval_assignment
    if failure == "C":
        monkeypatch.setattr(
            diagnostic.initialization,
            "reinitialized_candidate_parameter_snapshot",
            lambda **kw: diagnostic.initialization.snapshot_classifier_parameters(
                classifier=kw["classifier"]
            ),
        )
    elif failure == "learner_length":
        accumulate = diagnostic.training._accumulate_prequential_code_length
        monkeypatch.setattr(
            diagnostic.training,
            "_accumulate_prequential_code_length",
            lambda **kw: accumulate(**kw) + 1.0,
        )
    else:

        def fake_train(*args):
            if failure == "owner":
                resolve_arguments[
                    "held_model_training_state_registry"
                ].get_held_model_training_state(
                    model_id=9
                ).classifier.classification_layer.bias.data.add_(1)
            return SimpleNamespace(class_count=4), dict(parameter_digest="corrupted")

        monkeypatch.setattr(diagnostic, "_train_diagnostic", fake_train)
        monkeypatch.setattr(diagnostic, "_evaluate", lambda *args: {})
        monkeypatch.setattr(
            diagnostic, "_generate_samples", lambda **kw: (None, None, 0)
        )
    messages = dict(
        C="Cと学習者",
        learner_length="損失の和",
        product="A・T30と製品",
        owner="借用owner",
    )
    with pytest.raises(ValueError, match=messages[failure]):
        with diagnostic.candidate_start_diagnostics(
            dataset="sine2", run_seed=0, diagnostics={}
        ):
            diagnostic.streams.build_client_observed_streams(
                evaluation_concept_traces=(
                    SimpleNamespace(client_id=1, concept_ids_by_sample_index=(0, 0, 0)),
                ),
                sample_generator=SimpleNamespace(generate_sample=lambda **kw: None),
            )
            chosen = diagnostic.processing.select_alarm_interval_assignment(
                **select_arguments
            )
            assert chosen.change_interval_sample_count == 3
            diagnostic.response.resolve_alarm_change_interval(
                **resolve_arguments, alarm_interval_assignment_selection=chosen
            )
    assert diagnostic.processing.select_alarm_interval_assignment is original


def test_gzip_diagnostics_round_trip(tmp_path):
    import gzip
    import hashlib

    measurements = dict(
        schema_version=1, alarms=[dict(candidates={str(i): {} for i in range(13)})]
    )
    result = dict(candidate_start_diagnostics=deepcopy(measurements))
    research._save_candidate_start_diagnostics(result, tmp_path)
    reference = result["candidate_start_diagnostics"]
    payload = gzip.decompress((tmp_path / reference["path"]).read_bytes())
    assert json.loads(payload) == measurements
    assert hashlib.sha256(payload).hexdigest() == reference["uncompressed_sha256"]
    assert reference["alarm_count"] == 1


def test_training_wrapper_preserves_product_split_and_s4_mixed_concepts(monkeypatch):
    from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
        CandidateEpochTrainingSettings,
    )
    from tests.refactoring.test_alarm_interval_minimum_description_length_selection import (
        CurrentTrainingModelAssignment,
        ModelAndClassLossStatisticsStore,
    )

    registry = _registry(class_count=4, model_ids=(9,))
    classifier = registry.get_held_model_training_state(model_id=9).classifier
    initial = diagnostic.initialization.snapshot_classifier_parameters(
        classifier=classifier
    )
    settings = CandidateEpochTrainingSettings(
        candidate_training_strategy="validation_loss_early_stopping",
        maximum_epoch_count=30,
        maximum_batch_sample_count=4,
        validation_sample_fraction=0.2,
        consecutive_non_improving_epoch_limit=3,
        minimum_validation_loss_decrease=0.0001,
        candidate_shared_feature_update_policy="keep_shared_features_fixed",
    )
    arguments = dict(
        architecture_reference_classifier=classifier,
        initial_candidate_parameter_snapshot=initial,
        parameter_optimizer_settings=OPTIMIZER_SETTINGS,
        candidate_epoch_training_settings=settings,
        held_model_training_state_registry=registry,
        loss_statistics_store=ModelAndClassLossStatisticsStore(),
        current_training_model_assignment=CurrentTrainingModelAssignment(
            initial_model_id=9
        ),
        input_features=torch.randn(10, 3),
        observed_class_labels=torch.tensor(
            [[i % 4] for i in range(10)], dtype=torch.float32
        ),
        pending_assignment_training_samples=(),
        proposal_sample_index=10,
        estimated_change_point_sample_index=1,
        detection_episode_id=None,
        detector_name="diagnostic-test",
    )
    concepts = [0, 1] * 5
    states = diagnostic._random_states()
    original = diagnostic.resolution.train_candidate_on_alarm_change_interval
    expected = original(**deepcopy(arguments))
    expected_states = diagnostic._random_states()
    diagnostic._restore(states)
    actual, base_record = diagnostic._train_diagnostic(
        original, arguments, initial, "T30", {}, concepts
    )
    assert diagnostic._parameter_digest(actual) == diagnostic._parameter_digest(
        expected.candidate_training_state.candidate_classifier
    )
    _assert_random_states_equal(expected_states, diagnostic._random_states())
    generated = []

    def generate(**kwargs):
        generated.extend(kwargs["concepts"])
        return (
            torch.ones(len(kwargs["concepts"]), 3),
            torch.tensor(kwargs["concepts"], dtype=torch.float32).reshape(-1, 1),
            0,
        )

    monkeypatch.setattr(diagnostic, "_generate_samples", generate)
    diagnostic._restore(states)
    _, s4_record = diagnostic._train_diagnostic(
        original, arguments, initial, "S4-T30", {}, concepts
    )
    assert (
        base_record["training_sample_indices"] == s4_record["training_sample_indices"]
    )
    expected_concepts = [
        concepts[i] for i in base_record["training_sample_indices"] for _ in range(3)
    ]
    assert generated == expected_concepts
    assert len(generated) == 24
    assert set(generated) == {0, 1}
    assert s4_record["added_sample_counts_by_concept"] == dict(
        Counter(expected_concepts)
    )
