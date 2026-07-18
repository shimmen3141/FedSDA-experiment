"""CLI実験を開始する前の実行環境設定。"""

import os


def configure_native_thread_environment(environ=None):
    """OpenMP/MKLとCUDA確認方式の安全な既定値を設定する。

    呼び出し元の環境変数を優先するため、既に指定された値は変更しない。
    PyTorchやNumPyをimportする前に呼び出すこと。
    """
    target = os.environ if environ is None else environ
    target.setdefault("OMP_NUM_THREADS", "1")
    target.setdefault("MKL_NUM_THREADS", "1")
    target.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    return target


_torch_threads_configured = False


def configure_torch_threads():
    """PyTorchのCPU内・演算間スレッド数を実験開始前に設定する。"""
    global _torch_threads_configured
    if _torch_threads_configured:
        return

    import torch

    intraop_threads = int(os.environ["OMP_NUM_THREADS"])
    if intraop_threads < 1:
        raise ValueError("OMP_NUM_THREADS must be a positive integer")
    torch.set_num_threads(intraop_threads)
    torch.set_num_interop_threads(1)
    _torch_threads_configured = True


# このモジュールは各CLIから、NumPy/PyTorchを含む実験パッケージより先にimportされる。
configure_native_thread_environment()
