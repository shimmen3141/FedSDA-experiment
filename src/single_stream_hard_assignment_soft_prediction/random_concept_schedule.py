"""1本の系列の概念列を、製品のランダムな概念列（1クライアント）と同じ規則・乱数の消費の順で作る。

製品の`generate_random_client_concept_traces`は概念の数をデータセットの定義から取るので、製品に無いデータ
（例: STAGGERの3概念）には使えない。その1クライアント分の写しで、概念の数を引数で受け取る。再来は、
ランダムな変化の結果として生じる。riverとtorchをimportしない。
"""

from random import Random


def _require_int(name: str, value: object, minimum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name}はbool以外のbuiltin intが必要です。")
    if value < minimum:
        raise ValueError(f"{name}は{minimum}以上が必要です。")


def generate_random_concept_schedule(
    *,
    sample_count: int,
    concept_count: int,
    python_random_generator: Random,
    minimum_sample_index_gap_before_change_trial: int = 300,
    per_eligible_sample_concept_change_probability: float = 0.0015,
) -> tuple[int, ...]:
    """概念0・直前の変化の番号0から始め、間隔を超えた標本ごとに確率で変化を試し、今以外の概念から選ぶ。"""
    _require_int("sample_count", sample_count, 0)
    _require_int("concept_count", concept_count, 2)
    _require_int(
        "minimum_sample_index_gap_before_change_trial",
        minimum_sample_index_gap_before_change_trial,
        0,
    )
    if type(python_random_generator) is not Random:
        raise TypeError("python_random_generatorはrandom.Randomが必要です。")
    probability = per_eligible_sample_concept_change_probability
    if type(probability) not in (int, float):
        raise TypeError("per_eligible_sample_concept_change_probabilityは数が必要です。")
    if not 0 <= probability <= 1:
        raise ValueError("per_eligible_sample_concept_change_probabilityは0以上1以下が必要です。")
    concept_ids: list[int] = []
    current_concept_id = 0
    last_concept_change_sample_index = 0
    for sample_index in range(sample_count):
        if (
            sample_index - last_concept_change_sample_index
            > minimum_sample_index_gap_before_change_trial
            and python_random_generator.random() < probability
        ):
            alternative_concept_ids = [
                concept_id
                for concept_id in range(concept_count)
                if concept_id != current_concept_id
            ]
            # 製品と同じく、候補が一つでもchoiceの乱数消費を省略しない。
            current_concept_id = python_random_generator.choice(alternative_concept_ids)
            last_concept_change_sample_index = sample_index
        concept_ids.append(current_concept_id)
    return tuple(concept_ids)
