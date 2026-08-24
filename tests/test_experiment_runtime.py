from pathlib import Path

from experiment_runtime import configure_native_thread_environment


def test_runtime_environment_defaults_to_single_thread():
    environ = {}

    configure_native_thread_environment(environ)

    assert environ == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
    }


def test_runtime_environment_preserves_external_overrides():
    environ = {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "2",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "0",
    }

    configure_native_thread_environment(environ)

    assert environ["OMP_NUM_THREADS"] == "4"
    assert environ["MKL_NUM_THREADS"] == "2"
    assert environ["PYTORCH_NVML_BASED_CUDA_CHECK"] == "0"


def test_server_sweep_print_plan_exits_before_creating_directories():
    script = Path("tools/run_server_sweep.sh").read_text(encoding="utf-8")

    print_plan_guard = script.index("if [[ $print_plan == 1 ]]")
    print_plan_exit = script.index("exit 0", print_plan_guard)
    create_directories = script.index(
        'mkdir -p "$pareto_dir" "$raw_dir" "$log_dir"'
    )

    assert print_plan_guard < print_plan_exit < create_directories
