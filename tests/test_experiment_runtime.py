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
