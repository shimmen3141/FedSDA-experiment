"""サーバ確認前のモデルへ付ける、クライアント内の負の一時IDを採番する。"""


class TemporaryModelIdAllocator:
    """次に採番する一時IDだけを所有する。登録や正式IDへの付替えは行わない。"""

    def __init__(self, *, client_id: int) -> None:
        if type(client_id) is not int:
            raise TypeError("client_idはbool・派生型以外のbuiltin intが必要です。")
        if client_id < 0:
            raise ValueError("client_idは非負が必要です。")
        self._next_temporary_model_id = -100 - client_id

    @property
    def next_temporary_model_id(self) -> int:
        return self._next_temporary_model_id

    def allocate_temporary_model_id(self) -> int:
        """次の一時IDを返し、以後の採番値を1減らす。"""
        temporary_model_id = self._next_temporary_model_id
        self._next_temporary_model_id -= 1
        return temporary_model_id
