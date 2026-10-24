"""SEAの乱数消費を保ち、ラベル雑音だけを外す研究対照。製品への変更ではない。"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


@contextmanager
def noiseless_sea_policy():
    """雑音の乱数も引くが反転しない。事前学習にも同じ対照を適用する。"""
    import numpy as np
    from federated_learning_experiments.data.observed_streams import ObservedSample
    import federated_learning_experiments.data.sea.sea_sample_generation as sea

    def generate(self, *, concept_id):
        if type(concept_id) is not int or not 0 <= concept_id < self.concept_count:
            raise ValueError("概念IDが範囲外です")
        values = self.numpy_random_generator.uniform(0.0, 10.0, size=3)
        label = int(
            values[0] + values[1] <= sea._CONCEPT_DECISION_THRESHOLDS[concept_id]
        )
        self.numpy_random_generator.rand()
        return ObservedSample(
            feature_values=tuple(float(v) for v in values.astype(np.float32)),
            class_label=label,
        )

    with patch.object(sea.SeaSampleGenerator, "generate_sample", generate):
        yield


def execute_noise_job(job):
    from tools.research_shadow_learner_diagnostic.run_followup import execute_job

    if job["dataset"] != "sea4" or job["task"] != "optimizer":
        raise ValueError("SEA4のoptimizer対照だけを対象にします")
    with noiseless_sea_policy():
        return execute_job(job)


if __name__ == "__main__":
    from tools.research_shadow_learner_diagnostic import run_followup

    if "--help" in sys.argv or "--output" not in sys.argv:
        run_followup.main()
        raise SystemExit()
    scope = argparse.ArgumentParser(add_help=False)
    scope.add_argument("--datasets", nargs="+", choices=["sea4"])
    scope.add_argument("--task", choices=["optimizer"])
    scope.parse_known_args()
    if "--datasets" not in sys.argv:
        sys.argv.extend(["--datasets", "sea4"])
    if "--task" not in sys.argv:
        sys.argv.extend(["--task", "optimizer"])
    # 新規出力の作成・print-planの動作は既存CLIへ任せる。
    index = sys.argv.index("--output") + 1
    output = Path(sys.argv[index])
    planned = "--print-plan" in sys.argv
    source_bytes = Path(__file__).read_bytes()
    source_existed = (output / "research-source.json").exists()
    try:
        with patch.object(run_followup, "execute_job", execute_noise_job):
            run_followup.main()
    finally:
        metadata_path = output / "research-source.json"
        if not planned and not source_existed and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["research_condition"] = {
                "sea_label_noise_probability": 0.0,
                "original_label_noise_probability": 0.10,
                "noise_rng_draws_preserved": True,
                "pretraining_noise_also_disabled": True,
            }
            metadata["sha256"][
                Path(__file__).resolve().relative_to(ROOT).as_posix()
            ] = hashlib.sha256(source_bytes).hexdigest()
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
