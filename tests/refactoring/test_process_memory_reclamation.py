"""実行資源としてのメモリ返却。native bindingは代役で検証する。"""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from federated_learning_experiments.runtime import process_memory_reclamation as memory


@pytest.fixture(autouse=True)
def clear_binding_cache():
    memory._get_memory_reclaimer.cache_clear()
    yield
    memory._get_memory_reclaimer.cache_clear()


@pytest.mark.parametrize(
    "platform_name,libc_name", [("win32", ""), ("darwin", ""), ("linux", "musl")]
)
def test_unsupported_environment_does_not_load_native_library(
    monkeypatch, platform_name, libc_name
):
    monkeypatch.setattr(memory.sys, "platform", platform_name)
    monkeypatch.setattr(memory.platform, "libc_ver", lambda: (libc_name, ""))
    loader = Mock(side_effect=AssertionError("未対応環境ではロードしない"))
    monkeypatch.setattr(memory.ctypes, "CDLL", loader)
    memory.reclaim_unused_process_memory()
    memory.reclaim_unused_process_memory()
    loader.assert_not_called()


@pytest.mark.parametrize("return_value", [0, 1])
def test_glibc_binding_is_reused_and_zero_is_not_failure(monkeypatch, return_value):
    monkeypatch.setattr(memory.sys, "platform", "linux")
    libc_ver = Mock(return_value=("glibc", "2.35"))
    monkeypatch.setattr(memory.platform, "libc_ver", libc_ver)
    trim = Mock(return_value=return_value)
    loader = Mock(return_value=SimpleNamespace(malloc_trim=trim))
    monkeypatch.setattr(memory.ctypes, "CDLL", loader)
    assert memory.reclaim_unused_process_memory() is None
    assert memory.reclaim_unused_process_memory() is None
    loader.assert_called_once_with(None)
    libc_ver.assert_called_once_with()
    assert trim.argtypes == [ctypes.c_size_t]
    assert trim.restype is ctypes.c_int
    assert trim.call_args_list == [call(0), call(0)]


@pytest.mark.parametrize("failure", [OSError("load failed"), AttributeError("missing symbol")])
def test_unavailable_binding_is_cached_as_no_op(monkeypatch, failure):
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory.platform, "libc_ver", lambda: ("glibc", "2.35"))
    loader = Mock(side_effect=failure)
    monkeypatch.setattr(memory.ctypes, "CDLL", loader)
    memory.reclaim_unused_process_memory()
    memory.reclaim_unused_process_memory()
    loader.assert_called_once_with(None)


def test_missing_symbol_on_loaded_library_is_no_op(monkeypatch):
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda _: SimpleNamespace())
    assert memory.reclaim_unused_process_memory() is None


def test_unexpected_native_call_failure_is_not_hidden(monkeypatch):
    failure = RuntimeError("native invocation failed")
    monkeypatch.setattr(memory, "_get_memory_reclaimer", lambda: Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as raised:
        memory.reclaim_unused_process_memory()
    assert raised.value is failure
