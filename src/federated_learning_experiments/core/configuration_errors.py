"""実験設定の入力エラーを表す基礎例外。"""


class RunSettingsValidationError(ValueError):
    """不正な設定項目・指定値・日本語の理由と許容条件を保持する。"""

    def __init__(
        self,
        configuration_parameter_name: str,
        specified_parameter_value: object,
        validation_failure_reason: str,
    ) -> None:
        self.configuration_parameter_name = configuration_parameter_name
        self.specified_parameter_value = specified_parameter_value
        self.validation_failure_reason = validation_failure_reason
        super().__init__(
            f"実験設定の項目 {configuration_parameter_name!r} に指定された値 "
            f"{specified_parameter_value!r} は不正です。{validation_failure_reason}"
        )
