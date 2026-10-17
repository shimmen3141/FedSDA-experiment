"""掃引の排他・保存中断復旧・過去掃引選択を検証する。"""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from run_provenance_examples import TEST_RUN_PROVENANCE

from federated_learning_experiments.sweep import experiment_sweep_command as command
from federated_learning_experiments.sweep import experiment_sweep_execution as execution
from federated_learning_experiments.sweep import experiment_sweep_resumption as resumption
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
)


def _runs(seeds=(0,)):
    return build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=("fedavg",),
            dataset_names=("circle2",),
            random_seeds=seeds,
            client_count=2,
            per_client_sample_count=15,
            server_aggregation_intervals_per_client_samples=(5,),
        )
    )


def _npz_path(root, run):
    return root / run.configuration_name / "runs" / f"{run.run_identifier}.npz"


def _orphan(root, run):
    path = _npz_path(root, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"interrupted artifact")
    return path


def test_output_lock_is_cross_process_and_released(tmp_path):
    script = (
        f"import sys; sys.path.insert(0, {str(Path(resumption.__file__).parents[2])!r}); "
        "from pathlib import Path; "
        "from federated_learning_experiments.sweep.experiment_sweep_resumption "
        "import hold_sweep_output_lock; "
        f"\nwith hold_sweep_output_lock(Path({str(tmp_path)!r})): print('acquired')"
    )
    with resumption.hold_sweep_output_lock(tmp_path):
        child = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert child.returncode != 0
        assert "already running" in child.stderr
    child = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr


def test_same_proof_recovers_orphan_without_losing_old_bytes(tmp_path):
    run = _runs()[0]
    resumption.write_run_start_record(tmp_path, run, TEST_RUN_PROVENANCE)
    orphan = _orphan(tmp_path, run)
    outcomes = execution.execute_experiment_sweep(
        planned_runs=(run,),
        output_directory=tmp_path,
        worker_count=1,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert outcomes[0].run_status == "completed"
    assert orphan.read_bytes() != b"interrupted artifact"
    backups = list((tmp_path / ".resume" / "recovered").rglob("*.npz"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"interrupted artifact"
    assert (
        execution.find_existing_run_state(planned_run=run, output_directory=tmp_path).run_state
        == "completed"
    )


@pytest.mark.parametrize(
    "damage", ["missing", "commit", "dirty", "baseline", "environment", "settings", "broken"]
)
def test_unproven_orphan_is_preserved(tmp_path, damage):
    run = _runs()[0]
    if damage != "missing":
        proof = resumption.write_run_start_record(tmp_path, run, TEST_RUN_PROVENANCE)
        data = json.loads(proof.read_text(encoding="utf-8"))
        if damage == "commit":
            data["run_provenance"]["code_commit"] = "a" * 40
        if damage == "dirty":
            data["run_provenance"]["has_uncommitted_changes"] = True
        if damage == "baseline":
            data["run_provenance"]["behavior_baseline_version"] = -1
        if damage == "environment":
            data["environment"]["torch"] = "other"
        if damage == "settings":
            data["run_settings"] = {}
        proof.write_text("broken" if damage == "broken" else json.dumps(data), encoding="utf-8")
    orphan = _orphan(tmp_path, run)
    outcomes = execution.execute_experiment_sweep(
        planned_runs=(run,),
        output_directory=tmp_path,
        worker_count=1,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert outcomes[0].run_status == "failed"
    assert orphan.read_bytes() == b"interrupted artifact"


def test_completed_artifacts_are_not_changed_and_training_interrupt_resumes(tmp_path, monkeypatch):
    run = _runs()[0]
    original = execution.execute_fedavg_experiment_run

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(execution, "execute_fedavg_experiment_run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execution.execute_experiment_sweep(
            planned_runs=(run,),
            output_directory=tmp_path,
            worker_count=1,
            run_provenance=TEST_RUN_PROVENANCE,
        )
    assert not _npz_path(tmp_path, run).exists()
    monkeypatch.setattr(execution, "execute_fedavg_experiment_run", original)

    def execute():
        return execution.execute_experiment_sweep(
            planned_runs=(run,),
            output_directory=tmp_path,
            worker_count=14,
            run_provenance=TEST_RUN_PROVENANCE,
        )

    assert execute()[0].run_status == "completed"
    paths = list((tmp_path / "fedavg" / "runs").iterdir())
    before = {p.name: p.read_bytes() for p in paths}
    assert execute()[0].run_status == "skipped_existing"
    assert before == {p.name: p.read_bytes() for p in paths}


def test_plan_identity_ignores_order_but_not_settings(tmp_path):
    runs = _runs((0, 1))
    assert resumption.describe_planned_run_set(runs) == resumption.describe_planned_run_set(
        runs[::-1]
    )
    resumption.write_sweep_resumption_plan(tmp_path, runs)
    resumption.write_sweep_resumption_plan(tmp_path, runs[::-1])
    with pytest.raises(ValueError):
        resumption.write_sweep_resumption_plan(tmp_path, _runs((0,)))


_ARGS = [
    "--configurations",
    "fedavg",
    "--datasets",
    "circle2",
    "--seeds",
    "0",
    "1",
    "--aggregation-intervals",
    "5",
    "--client-count",
    "2",
    "--per-client-sample-count",
    "15",
    "--purpose",
    "resume tests",
    "--study",
    "study",
    "--label",
    "test",
]


def _tree(root):
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and p.name != ".sweep.lock"
    }


@pytest.fixture
def cli_root(tmp_path, monkeypatch):
    monkeypatch.setattr(command, "_REPOSITORY_DIRECTORY", tmp_path)
    monkeypatch.setattr(command, "build_run_provenance", lambda **kwargs: TEST_RUN_PROVENANCE)
    return tmp_path


def _choose(monkeypatch, value):
    monkeypatch.setattr(command, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: value)


def test_partial_old_readme_can_be_selected_with_new_metadata_and_workers(
    cli_root, monkeypatch, capsys
):
    # 旧実装のように補助計画がなく、最初の1runしか終わっていない掃引。
    output = cli_root / "results" / "old-study" / "old"
    output.mkdir(parents=True)
    from federated_learning_experiments.sweep.experiment_sweep_records import (
        write_experiment_readme,
    )

    write_experiment_readme(
        output_directory=output,
        purpose="original purpose",
        study_name="old-study",
        started_at="2026-10-18T00:00:00+09:00",
        command_arguments=tuple(_ARGS),
        run_provenance=TEST_RUN_PROVENANCE,
    )
    execution.execute_experiment_sweep(
        planned_runs=_runs(),
        output_directory=output,
        worker_count=1,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    readme = (output / "README.md").read_bytes()
    old_npz = _npz_path(output, _runs()[0]).read_bytes()
    _choose(monkeypatch, "1")
    assert command.main([*_ARGS, "--workers", "14", "--purpose", "new purpose"]) == 0
    text = capsys.readouterr().out
    assert "(1/2 run)" in text
    assert "skipped_existing" in text
    assert (output / "README.md").read_bytes() == readme
    assert _npz_path(output, _runs()[0]).read_bytes() == old_npz
    record = json.loads(next(output.glob("sweep_record_*.json")).read_text(encoding="utf-8"))
    assert record["purpose"] == "new purpose"
    assert len(list(output.rglob("runs/*.json"))) == 2


@pytest.mark.parametrize("action", ["completed", "cancel", "noninteractive", "print-plan"])
def test_selection_can_finish_without_any_file_changes(cli_root, monkeypatch, capsys, action):
    assert command.main([*_ARGS, "--new-output"]) == 0
    before = _tree(cli_root)
    _choose(monkeypatch, "0" if action == "cancel" else "1")
    extra = []
    if action == "noninteractive":
        monkeypatch.setattr(command, "_is_interactive", lambda: False)
    if action == "print-plan":
        extra = ["--print-plan"]
    assert command.main([*_ARGS, *extra]) == (2 if action == "noninteractive" else 0)
    assert "(2/2 run)" in capsys.readouterr().out
    assert _tree(cli_root) == before


def test_select_new_and_explicit_new_do_not_overwrite_same_second_output(cli_root, monkeypatch):
    assert command.main([*_ARGS, "--new-output"]) == 0
    _choose(monkeypatch, "2")  # 候補1件の次が新規。
    assert command.main(_ARGS) == 0
    assert command.main([*_ARGS, "--new-output"]) == 0
    assert len(list((cli_root / "results" / "study").glob("*/README.md"))) == 3


def test_different_or_broken_plan_is_not_offered(cli_root, monkeypatch, capsys):
    assert command.main([*_ARGS, "--new-output"]) == 0
    proof = next(cli_root.rglob(".resume/sweep.json"))
    proof.write_text("broken", encoding="utf-8")
    monkeypatch.setattr(command, "_is_interactive", lambda: False)
    assert command.main([*_ARGS, "--print-plan"]) == 0
    assert "(2/2 run)" not in capsys.readouterr().out


def test_active_output_refuses_command_before_readme_or_records_change(cli_root, capsys):
    output = cli_root / "explicit"
    with resumption.hold_sweep_output_lock(output):
        before = _tree(output)
        args = [value for value in _ARGS]
        index = args.index("--study")
        del args[index : index + 2]
        assert command.main([*args, "--output-directory", str(output)]) == 2
        assert "already running" in capsys.readouterr().err
        assert _tree(output) == before


def test_symlink_does_not_allow_recovery_outside_output(tmp_path):
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    try:
        (output / ".resume").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS権限がsymlink作成を許可していない")
    with pytest.raises(ValueError, match="symlink"):
        resumption.write_run_start_record(output, _runs()[0], TEST_RUN_PROVENANCE)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("change", ["version", "unknown", "dirty-current", "missing-commit"])
def test_invalid_recovery_proof_types_and_unknown_current_code_are_refused(tmp_path, change):
    run = _runs()[0]
    proof = resumption.write_run_start_record(tmp_path, run, TEST_RUN_PROVENANCE)
    data = json.loads(proof.read_text(encoding="utf-8"))
    provenance = TEST_RUN_PROVENANCE
    if change == "version":
        data["version"] = True
    if change == "unknown":
        data["unexpected"] = "value"
    if change == "dirty-current":
        provenance = replace(provenance, has_uncommitted_changes=True)
    if change == "missing-commit":
        provenance = replace(
            provenance,
            code_commit=None,
            has_uncommitted_changes=None,
            code_revision_unavailable_reason="unavailable",
        )
    proof.write_text(json.dumps(data), encoding="utf-8")
    orphan = _orphan(tmp_path, run)
    assert not resumption.can_recover_interrupted_run(tmp_path, run, provenance)
    assert orphan.read_bytes() == b"interrupted artifact"


def test_process_termination_releases_output_lock(tmp_path):
    script = (
        f"import sys; sys.path.insert(0, {str(Path(resumption.__file__).parents[2])!r}); "
        "from pathlib import Path; import time; "
        "from federated_learning_experiments.sweep.experiment_sweep_resumption "
        "import hold_sweep_output_lock; "
        f"\nwith hold_sweep_output_lock(Path({str(tmp_path)!r})):\n"
        " print('locked', flush=True)\n time.sleep(120)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(ValueError, match="already running"):
            with resumption.hold_sweep_output_lock(tmp_path):
                pytest.fail("二重に排他を取得した")
    finally:
        child.terminate()
        child.communicate(timeout=30)
    with resumption.hold_sweep_output_lock(tmp_path):
        pass


def test_cli_recovery_counts_and_preserves_saved_npz(cli_root, monkeypatch, capsys):
    output = cli_root / "results" / "interrupted"
    runs = _runs((0, 1))
    resumption.write_sweep_resumption_plan(output, runs)
    resumption.write_run_start_record(output, runs[0], TEST_RUN_PROVENANCE)
    _orphan(output, runs[0])
    _choose(monkeypatch, "1")
    assert command.main(_ARGS) == 0
    printed = capsys.readouterr().out
    assert "(0/2 run)" in printed
    assert "自動復旧可能1件" in printed
    assert len(list((output / ".resume" / "recovered").rglob("*.npz"))) == 1


def test_unproven_interruption_is_shown_and_selection_does_not_modify_it(
    cli_root, monkeypatch, capsys
):
    output = cli_root / "results" / "interrupted"
    runs = _runs((0, 1))
    resumption.write_sweep_resumption_plan(output, runs)
    _orphan(output, runs[0])
    before = _tree(cli_root)
    _choose(monkeypatch, "1")
    assert command.main(_ARGS) == 2
    assert "要確認1件" in capsys.readouterr().out
    assert _tree(cli_root) == before


def test_different_requested_run_set_does_not_resume_existing(cli_root, monkeypatch, capsys):
    assert command.main([*_ARGS, "--new-output"]) == 0
    monkeypatch.setattr(command, "_is_interactive", lambda: False)
    assert command.main([*_ARGS, "--seeds", "2", "--print-plan"]) == 0
    assert "同じ条件" not in capsys.readouterr().out


@pytest.mark.parametrize("change", ["add", "edit", "remove"])
def test_untracked_execution_source_change_disables_recovery(tmp_path, monkeypatch, change):
    source_root = tmp_path / "code"
    (source_root / "src").mkdir(parents=True)
    existing = source_root / "src" / "untracked.py"
    existing.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(resumption, "_REPOSITORY_DIRECTORY", source_root)
    output = tmp_path / "output"
    run = _runs()[0]
    resumption.write_run_start_record(output, run, TEST_RUN_PROVENANCE)
    orphan = _orphan(output, run)
    assert resumption.can_recover_interrupted_run(output, run, TEST_RUN_PROVENANCE)
    if change == "add":
        (source_root / "src" / "extra.py").write_text("value = 2\n", encoding="utf-8")
    elif change == "edit":
        existing.write_text("value = 3\n", encoding="utf-8")
    else:
        existing.unlink()
    assert not resumption.can_recover_interrupted_run(output, run, TEST_RUN_PROVENANCE)
    assert orphan.read_bytes() == b"interrupted artifact"
