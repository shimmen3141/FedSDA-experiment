"""seedのCLI表記だけを正規化し、実験設定には整数列を渡す。"""

import re
from argparse import Action, ArgumentError, ArgumentParser, Namespace


def _expand_seed_tokens(tokens: list[object]) -> list[int]:
    expanded: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if not isinstance(token, str):
            raise ValueError("seedには整数または開始-終了を指定してください。")
        interval = re.fullmatch(r"([0-9]+)-([0-9]+)", token)
        if interval is None:
            try:
                start = end = int(token)
            except ValueError as error:
                raise ValueError("seedには整数または開始-終了を指定してください。") from error
        else:
            start, end = (int(value) for value in interval.groups())
        if not (0 <= start < 2**32 and 0 <= end < 2**32):
            raise ValueError("seedは0以上2**32未満で指定してください。")
        if start > end:
            raise ValueError("seedの範囲は開始値<=終了値で指定してください。")
        if len(expanded) + end - start + 1 > 100000:
            raise ValueError("--seedsの展開結果は100000個以内にしてください。")
        for seed in range(start, end + 1):
            if seed in seen:
                raise ValueError(f"seedが重複しています: {seed}")
            seen.add(seed)
            expanded.append(seed)
    return expanded


class SeedListArgumentAction(Action):
    """列挙と終端込みの範囲を、順序を維持したseedのlistへ変換する。"""

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        try:
            if not isinstance(values, list):
                raise ValueError("--seedsには1個以上のseedを指定してください。")
            expanded = _expand_seed_tokens(values)
        except ValueError as error:
            raise ArgumentError(self, str(error)) from error
        setattr(namespace, self.dest, expanded)
