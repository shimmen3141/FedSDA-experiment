"""一時モデルIDの採番を、実旧clientの初期化と採番処理へ対照する。"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_adopted_candidate_initial_local_registration import build_initial_registration_oracle

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import (
    register_adopted_candidate_as_temporary_held_model,
)


class IntSubclass(int):
    """client_idとして受理しない派生型。"""


def build_legacy_temporary_id_client(*, client_id):
    # 実__init__と実_alloc_temp_idだけを使う。モデル辞書の中身は採番へ影響しない。
    return BaseClient(client_id, {0: object()}, verbose=False)


@pytest.mark.parametrize("client_id", [0, 1, 7, 10**40])
@pytest.mark.parametrize("allocation_count", [0, 1, 2, 5])
def test_allocation_sequence_matches_actual_legacy_client(client_id, allocation_count):
    legacy_client = build_legacy_temporary_id_client(client_id=client_id)
    temporary_model_id_allocator = TemporaryModelIdAllocator(client_id=client_id)
    assert temporary_model_id_allocator.next_temporary_model_id == legacy_client.next_temp_id
    expected_temporary_model_ids = []
    actual_temporary_model_ids = []
    for _allocation_index in range(allocation_count):
        expected_temporary_model_ids.append(legacy_client._alloc_temp_id())
        actual_temporary_model_ids.append(
            temporary_model_id_allocator.allocate_temporary_model_id()
        )
        assert temporary_model_id_allocator.next_temporary_model_id == legacy_client.next_temp_id
    assert actual_temporary_model_ids == expected_temporary_model_ids
    assert all(type(temporary_model_id) is int for temporary_model_id in actual_temporary_model_ids)
    assert all(temporary_model_id < 0 for temporary_model_id in actual_temporary_model_ids)
    assert len(set(actual_temporary_model_ids)) == allocation_count


def test_reading_next_id_does_not_allocate():
    legacy_client = build_legacy_temporary_id_client(client_id=3)
    temporary_model_id_allocator = TemporaryModelIdAllocator(client_id=3)
    for _allocation_index in range(4):
        assert temporary_model_id_allocator.next_temporary_model_id == legacy_client.next_temp_id
    assert (
        temporary_model_id_allocator.allocate_temporary_model_id() == legacy_client._alloc_temp_id()
    )
    with pytest.raises(AttributeError):
        temporary_model_id_allocator.next_temporary_model_id = -1


def test_allocators_are_independent_per_owner():
    first_allocator = TemporaryModelIdAllocator(client_id=0)
    second_allocator = TemporaryModelIdAllocator(client_id=0)
    first_temporary_model_id = first_allocator.allocate_temporary_model_id()
    first_allocator.allocate_temporary_model_id()
    assert second_allocator.next_temporary_model_id == first_temporary_model_id
    assert second_allocator.allocate_temporary_model_id() == first_temporary_model_id


@pytest.mark.parametrize(
    ("invalid_client_id", "expected_exception"),
    [
        (-1, ValueError),
        (-(10**40), ValueError),
        (True, TypeError),
        (False, TypeError),
        (0.0, TypeError),
        ("0", TypeError),
        (None, TypeError),
        (IntSubclass(0), TypeError),
    ],
)
def test_invalid_client_id_is_rejected(invalid_client_id, expected_exception):
    with pytest.raises(expected_exception):
        TemporaryModelIdAllocator(client_id=invalid_client_id)


@pytest.mark.parametrize("class_count", [2, 4])
def test_allocated_ids_satisfy_initial_registration_contract_for_consecutive_candidates(
    class_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(307)
        registration_arguments, _, _, _ = build_initial_registration_oracle(
            class_count=class_count,
            held_model_ids=(4,),
            current_model_is_held=True,
            monkeypatch=monkeypatch,
        )
        legacy_client = build_legacy_temporary_id_client(client_id=2)
        temporary_model_id_allocator = TemporaryModelIdAllocator(client_id=2)
        registry = registration_arguments["held_model_training_state_registry"]
        loss_statistics_store = registration_arguments["loss_statistics_store"]
        pending_upload_state = registration_arguments["pending_model_upload_state"]
        expected_temporary_model_ids = []
        for _allocation_index in range(2):
            expected_temporary_model_ids.append(legacy_client._alloc_temp_id())
            registration_arguments["temporary_model_id"] = (
                temporary_model_id_allocator.allocate_temporary_model_id()
            )
            register_adopted_candidate_as_temporary_held_model(**registration_arguments)
            assert registration_arguments["temporary_model_id"] == expected_temporary_model_ids[-1]
            assert (
                pending_upload_state.get_pending_model_upload().model_id
                == expected_temporary_model_ids[-1]
            )
            # 次の採用候補を、独立した共有部を持つ新しい分類器として用意する。
            classifier = ResidualAdapterClassifier(
                model_architecture_settings=ModelArchitectureSettings(
                    model_architecture_name="shared_backbone_residual_adapter",
                    residual_adapter_requested_rank=2,
                ),
                input_feature_count=2,
                hidden_layer_widths=(5, 4),
                class_count=class_count,
            )
            registration_arguments["adopted_candidate_classifier"] = classifier
            registration_arguments["candidate_concept_specific_parameter_optimizer_state"] = (
                ParameterOptimizerState(
                    parameters=tuple(classifier.residual_adapter.parameters())
                    + tuple(classifier.classification_layer.parameters()),
                    optimizer_settings=AdamParameterOptimizerSettings(
                        learning_rate=0.01, weight_decay=0.001, adam_variant="standard"
                    ),
                )
            )
        assert expected_temporary_model_ids == [-102, -103]
        assert tuple(
            state.model_id for state in registry.snapshot_ordered_held_model_training_states()
        ) == (4, *expected_temporary_model_ids)
        assert tuple(model_id for model_id, _ in loss_statistics_store.get_state_snapshot()) == (
            4,
            *expected_temporary_model_ids,
        )


def test_allocator_module_runs_with_standard_library_only():
    source_directory = Path(__file__).resolve().parents[2] / "src"
    completed_process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            f"sys.path.insert(0, {str(source_directory)!r})\n"
            "from federated_learning_experiments.learning.training.temporary_model_id_allocation"
            " import TemporaryModelIdAllocator\n"
            "allocator = TemporaryModelIdAllocator(client_id=1)\n"
            "assert [allocator.allocate_temporary_model_id() for _ in range(2)] == [-101, -102]\n"
            "assert not {'torch', 'numpy', 'federated_drift_experiment'} & set(sys.modules)\n"
            "print('PASS')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed_process.returncode == 0, completed_process.stderr
    assert completed_process.stdout.strip() == "PASS"
