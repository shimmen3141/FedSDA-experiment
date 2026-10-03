"""失敗した実行段階・位置・理由を保持する例外。"""

from federated_learning_experiments.execution.run_execution_records import (
    validate_run_execution_stage_and_positions,
)


class RunExecutionError(RuntimeError):
    """接続先の元例外はraise-fromによる標準causeとして保持する。"""

    def __init__(
        self,
        *,
        stage_name: str,
        failure_reason: str,
        client_id: int | None = None,
        sample_index: int | None = None,
        round_index: int | None = None,
    ) -> None:
        validate_run_execution_stage_and_positions(
            stage_name=stage_name, client_id=client_id,
            sample_index=sample_index, round_index=round_index,
        )
        if type(failure_reason) is not str:
            raise TypeError("failure_reasonには失敗理由の文字列を指定してください。")
        self.stage_name = stage_name
        self.failure_reason = failure_reason
        self.client_id = client_id
        self.sample_index = sample_index
        self.round_index = round_index
        super().__init__(
            f"runの実行段階 {stage_name!r} で失敗しました "
            f"(client_id={client_id}, sample_index={sample_index}, round_index={round_index})。"
            f"{failure_reason}"
        )
