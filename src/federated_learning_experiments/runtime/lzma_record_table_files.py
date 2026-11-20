"""runの記録の表を、各配列をLZMAで圧縮したNPZとして書く。

`numpy.savez_compressed`と同じく、各配列を`<名前>.npy`として辞書の順に書き、zip64を常に許す。違いは、zipの
圧縮方式がLZMAであることと、objectの配列を書かない（`allow_pickle=False`）ことだけ。読込みは`numpy.load`のまま
（zipfileがLZMAを読む。読む側にもPythonの`lzma`が要る）。
`lzma`はmoduleの先頭では読み込まない（`lzma`の無い環境で、deflateのNPZを読むだけの経路まで失敗させないため）。
"""

from importlib import import_module
from typing import BinaryIO
from zipfile import ZIP_LZMA, ZipFile

from numpy import asanyarray, ndarray
from numpy.lib.format import write_array


def ensure_lzma_compression_available() -> None:
    """Pythonの`lzma`が使えることを確かめ、使えなければ`RuntimeError`を出す。掃引がrunの計算の前に呼ぶ。"""
    try:
        import_module("lzma")
    except ImportError as caught_exception:
        raise RuntimeError(
            "Pythonのlzmaが使えないため、runの記録の表（LZMAのNPZ）を書けません。"
        ) from caught_exception


def write_lzma_compressed_record_tables(
    record_table_stream: BinaryIO, record_tables: dict[str, ndarray]
) -> None:
    """記録の表を、各配列を`<名前>.npy`としてLZMAで圧縮したNPZにして、streamへ書く。"""
    with ZipFile(record_table_stream, "w", compression=ZIP_LZMA, allowZip64=True) as archive:
        for table_column_name, values in record_tables.items():
            with archive.open(f"{table_column_name}.npy", "w", force_zip64=True) as member:
                write_array(member, asanyarray(values), allow_pickle=False)
