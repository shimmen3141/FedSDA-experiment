"""モデルや標本の寿命を変えず、解放済み領域のOS返却を要求する。"""

import ctypes
import platform
import sys
from collections.abc import Callable
from functools import cache


@cache
def _get_memory_reclaimer() -> Callable[[], None] | None:
    if sys.platform != "linux" or platform.libc_ver()[0] != "glibc":
        return None
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (OSError, AttributeError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int

    def reclaim() -> None:
        # 戻り0は返却対象がなかったことを示し、実験の失敗ではない。
        trim(0)

    return reclaim


def reclaim_unused_process_memory() -> None:
    """Linux/glibcで解放済みメモリを返却する。未対応環境では何もしない。"""
    reclaim = _get_memory_reclaimer()
    if reclaim is not None:
        reclaim()
