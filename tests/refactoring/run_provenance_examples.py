"""testで、保存と掃引の関数へ渡すrunの来歴の例。"""

from federated_learning_experiments.core.run_provenance import (
    BEHAVIOR_BASELINE_VERSION,
    RunProvenance,
)

# gitの情報が取れた場合の例（今の基準版）。
TEST_RUN_PROVENANCE = RunProvenance(
    code_commit="0123456789abcdef0123456789abcdef01234567",
    has_uncommitted_changes=False,
    code_revision_unavailable_reason=None,
    behavior_baseline_version=BEHAVIOR_BASELINE_VERSION,
)
