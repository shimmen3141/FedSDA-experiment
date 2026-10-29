"""警報の後の変化区間の長さと帰属先を、最小記述長の原理で選ぶ純粋な計算（torchもownerも使わない）。

警報の時点で、帰属の保留にある標本と警報の標本を古い順に並べた列について、次の仮説の符号長（nats）を比べ、
最短の仮説を選ぶ。

- 現在のモデルのまま: 列の全部を、現在のモデルの対数損失で符号化する。
- 変化区間の長さkで、ほかの保有モデルmへ切り替わる: 前のn−k件を現在のモデル、後のk件をmの対数損失で
  符号化し、切り替わりの符号長を足す。
- 変化区間の長さkで、新しいモデルへ切り替わる: 前のn−k件を現在のモデル、後のk件を新しいモデルの予測的
  符号長（学びながら予測する学習者の対数損失の和）で符号化し、切り替わりの符号長を足す。学習者が複数あれば、
  最短の学習者の予測的符号長に、どの学習者かを表す符号長ln(学習者の数)を足す（`select_new_model_learner`）。

kは、最小の変化区間の件数から指定された刻みで、列の長さn以下。新しいモデルの学び始めの誤りの分が、新しいモデルを
作ることの罰として働くので、固定の閾値を使わない。
"""

from dataclasses import dataclass
from math import isfinite, log

# 選んだ仮説: 現在のモデルのまま、保有モデルへ切り替わる、新しいモデルへ切り替わる。
SELECTED_HYPOTHESES: tuple[str, ...] = ("current_model", "held_model", "new_model")

# 「新しいモデル」の仮説の学習者の名前: 概念ごとの部分を初期化し直した学習者、現在のモデルの写しを初期化せずに
# 学ばせるwarm startの学習者。「新しいモデル」以外の仮説を選んだときの、該当なし。
REINITIALIZED_LEARNER_NAME = "reinitialized_learner"
WARM_STARTED_LEARNER_NAME = "warm_started_learner"
NO_NEW_MODEL_LEARNER_NAME = ""
_NEW_MODEL_LEARNER_NAMES: tuple[str, ...] = (REINITIALIZED_LEARNER_NAME, WARM_STARTED_LEARNER_NAME)


def _validate_count(*, count: int, count_name: str, minimum_count: int) -> None:
    if type(count) is not int:
        raise TypeError(f"{count_name} must be builtin int")
    if count < minimum_count:
        raise ValueError(f"{count_name} must be at least {minimum_count}")


def _validate_code_length(*, code_length: float, code_length_name: str) -> None:
    if type(code_length) not in (int, float):
        raise TypeError(f"{code_length_name} must be builtin int or float")
    if not isfinite(code_length) or code_length < 0:
        raise ValueError(f"{code_length_name} must be finite and nonnegative")


def compute_switching_code_length(*, switching_probability: float, destination_count: int) -> float:
    """帰属先が1回切り替わることの符号長: −ln(切り替わりの確率)＋ln(切り替わり先の数)。"""
    if type(switching_probability) is not float:
        raise TypeError("switching_probability must be builtin float")
    if not 0.0 < switching_probability < 1.0:
        raise ValueError("switching_probability must be greater than 0 and less than 1")
    _validate_count(count=destination_count, count_name="destination_count", minimum_count=1)
    return -log(switching_probability) + log(destination_count)


def _validate_new_model_learner_name(*, learner_name: str) -> None:
    if type(learner_name) is not str:
        raise TypeError("new model learner name must be builtin str")
    if learner_name not in _NEW_MODEL_LEARNER_NAMES:
        raise ValueError("new model learner name must be a declared learner")


@dataclass(frozen=True, kw_only=True)
class NewModelLearnerSelection:
    """「新しいモデル」の仮説で、予測的符号長が最短だった学習者と、その符号長＋学習者の選択の符号長（nats）。"""

    learner_name: str
    description_length: float

    def __post_init__(self) -> None:
        _validate_new_model_learner_name(learner_name=self.learner_name)
        _validate_code_length(
            code_length=self.description_length, code_length_name="description_length"
        )


def select_new_model_learner(
    *, prequential_code_lengths_by_learner_name: tuple[tuple[str, float], ...]
) -> NewModelLearnerSelection:
    """学習者ごとの予測的符号長から、最短の学習者を選び、学習者の選択の符号長ln(学習者の数)を足す。

    学習者に一様な事前分布を置いた、どの学習者で符号化したかを表す長さ。同じ値なら、先の学習者を選ぶ。学習者が
    1つなら、足す値は0で、予測的符号長そのもの。
    """
    if type(prequential_code_lengths_by_learner_name) is not tuple:
        raise TypeError("prequential_code_lengths_by_learner_name must be builtin tuple")
    if not prequential_code_lengths_by_learner_name:
        raise ValueError("prequential_code_lengths_by_learner_name must not be empty")
    seen_learner_names: set[str] = set()
    for learner_entry in prequential_code_lengths_by_learner_name:
        if type(learner_entry) is not tuple or len(learner_entry) != 2:
            raise TypeError("learner entries must be builtin (learner_name, code_length) tuples")
        learner_name, code_length = learner_entry
        _validate_new_model_learner_name(learner_name=learner_name)
        if learner_name in seen_learner_names:
            raise ValueError("learner names must be unique")
        seen_learner_names.add(learner_name)
        _validate_code_length(code_length=code_length, code_length_name="learner code length")
    best_learner_name, best_code_length = prequential_code_lengths_by_learner_name[0]
    for learner_name, code_length in prequential_code_lengths_by_learner_name[1:]:
        if code_length < best_code_length:
            best_learner_name, best_code_length = learner_name, code_length
    learner_count = len(prequential_code_lengths_by_learner_name)
    return NewModelLearnerSelection(
        learner_name=best_learner_name,
        description_length=best_code_length
        if learner_count == 1
        else best_code_length + log(float(learner_count)),
    )


@dataclass(frozen=True, kw_only=True)
class MinimumDescriptionLengthAssignmentSelection:
    """最小記述長の原理で選んだ仮説、変化区間の長さ、行き先の保有モデルのID、その仮説の符号長（nats）。"""

    selected_hypothesis: str
    change_interval_sample_count: int
    # 行き先の保有モデルのID。現在のモデルのまま、または新しいモデルなら、None。
    destination_model_id: int | None
    description_length: float
    # 「新しいモデル」を選んだときの、その変化区間の長さで最短だった学習者。ほかの仮説では`NO_NEW_MODEL_LEARNER_NAME`。
    new_model_learner_name: str

    def __post_init__(self) -> None:
        if (
            type(self.selected_hypothesis) is not str
            or self.selected_hypothesis not in SELECTED_HYPOTHESES
        ):
            raise ValueError("selected_hypothesis must be a declared value")
        _validate_count(
            count=self.change_interval_sample_count,
            count_name="change_interval_sample_count",
            minimum_count=1,
        )
        if self.selected_hypothesis == "held_model":
            if type(self.destination_model_id) is not int:
                raise ValueError("held_model selection needs a builtin int destination_model_id")
        elif self.destination_model_id is not None:
            raise ValueError("only held_model selection has destination_model_id")
        _validate_code_length(
            code_length=self.description_length, code_length_name="description_length"
        )
        if type(self.new_model_learner_name) is not str:
            raise TypeError("new_model_learner_name must be builtin str")
        if self.selected_hypothesis == "new_model":
            _validate_new_model_learner_name(learner_name=self.new_model_learner_name)
        elif self.new_model_learner_name != NO_NEW_MODEL_LEARNER_NAME:
            raise ValueError("only new_model selection has a new model learner name")


def select_assignment_by_minimum_description_length(
    *,
    current_model_log_losses: tuple[float, ...],
    held_model_log_losses_by_model_id: tuple[tuple[int, tuple[float, ...]], ...],
    new_model_code_lengths_by_change_interval_sample_count: dict[int, float],
    new_model_learner_names_by_change_interval_sample_count: dict[int, str],
    switching_code_length: float,
    change_interval_grid_sample_count: int,
    change_interval_step_sample_count: int,
) -> MinimumDescriptionLengthAssignmentSelection:
    """符号長が最短の仮説を選ぶ。

    `current_model_log_losses`は、列（古い順）の各標本の、現在のモデルの対数損失。
    `held_model_log_losses_by_model_id`は、行き先の保有モデル（現在のモデル以外。この順に調べる）ごとの、列の
    対数損失。`new_model_code_lengths_by_change_interval_sample_count`は、変化区間の長さkごとの、
    列の末尾k件での新しいモデルの予測的符号長。kは`change_interval_grid_sample_count`から
    `change_interval_step_sample_count`刻みでn以下の全部（どちらも正のbuiltin int）。
    `new_model_learner_names_by_change_interval_sample_count`は、同じkごとの、その符号長の学習者の名前で、
    「新しいモデル」を選んだときに結果へ入れる（選択の規則には使わない）。

    「現在のモデルのまま」を最初の最良とし、kを短い順に、各kで行き先の保有モデル（順に）、新しいモデルの順に
    調べ、最良より厳密に短いときだけ置き換える（同じ長さなら、先に調べた仮説を選ぶ）。切り替わる仮説の符号長は、
    （前の部分の和＋切り替わりの符号長）＋後の部分の和の順に足す。
    """
    if type(current_model_log_losses) is not tuple:
        raise TypeError("current_model_log_losses must be builtin tuple")
    sequence_length = len(current_model_log_losses)
    if sequence_length < 1:
        raise ValueError("current_model_log_losses must not be empty")
    for loss in current_model_log_losses:
        _validate_code_length(code_length=loss, code_length_name="current model log loss")
    if type(held_model_log_losses_by_model_id) is not tuple:
        raise TypeError("held_model_log_losses_by_model_id must be builtin tuple")
    seen_model_ids: set[int] = set()
    for held_model_entry in held_model_log_losses_by_model_id:
        if type(held_model_entry) is not tuple or len(held_model_entry) != 2:
            raise TypeError("held model entries must be builtin (model_id, losses) tuples")
        model_id, losses = held_model_entry
        if type(model_id) is not int:
            raise TypeError("held model ID must be builtin int")
        if model_id in seen_model_ids:
            raise ValueError("held model IDs must be unique")
        seen_model_ids.add(model_id)
        if type(losses) is not tuple:
            raise TypeError("held model log losses must be builtin tuple")
        if len(losses) != sequence_length:
            raise ValueError("held model log losses must have the sequence length")
        for loss in losses:
            _validate_code_length(code_length=loss, code_length_name="held model log loss")
    _validate_code_length(
        code_length=switching_code_length, code_length_name="switching_code_length"
    )
    _validate_count(
        count=change_interval_grid_sample_count,
        count_name="change_interval_grid_sample_count",
        minimum_count=1,
    )
    if type(new_model_code_lengths_by_change_interval_sample_count) is not dict:
        raise TypeError(
            "new_model_code_lengths_by_change_interval_sample_count must be builtin dict"
        )
    _validate_count(
        count=change_interval_step_sample_count,
        count_name="change_interval_step_sample_count",
        minimum_count=1,
    )
    change_interval_sample_counts = tuple(
        range(
            change_interval_grid_sample_count,
            sequence_length + 1,
            change_interval_step_sample_count,
        )
    )
    if set(new_model_code_lengths_by_change_interval_sample_count) != set(
        change_interval_sample_counts
    ):
        raise ValueError(
            "new model code lengths must cover exactly the grid change interval lengths"
        )
    for code_length in new_model_code_lengths_by_change_interval_sample_count.values():
        _validate_code_length(code_length=code_length, code_length_name="new model code length")
    if type(new_model_learner_names_by_change_interval_sample_count) is not dict:
        raise TypeError(
            "new_model_learner_names_by_change_interval_sample_count must be builtin dict"
        )
    if set(new_model_learner_names_by_change_interval_sample_count) != set(
        change_interval_sample_counts
    ):
        raise ValueError(
            "new model learner names must cover exactly the grid change interval lengths"
        )
    for learner_name in new_model_learner_names_by_change_interval_sample_count.values():
        _validate_new_model_learner_name(learner_name=learner_name)

    best = MinimumDescriptionLengthAssignmentSelection(
        selected_hypothesis="current_model",
        change_interval_sample_count=sequence_length,
        destination_model_id=None,
        description_length=sum(current_model_log_losses),
        new_model_learner_name=NO_NEW_MODEL_LEARNER_NAME,
    )
    for change_interval_sample_count in change_interval_sample_counts:
        earlier_sample_count = sequence_length - change_interval_sample_count
        earlier_code_length = (
            sum(current_model_log_losses[:earlier_sample_count]) + switching_code_length
        )
        options: list[tuple[float, str, int | None]] = [
            (sum(losses[earlier_sample_count:]), "held_model", model_id)
            for model_id, losses in held_model_log_losses_by_model_id
        ]
        options.append(
            (
                new_model_code_lengths_by_change_interval_sample_count[
                    change_interval_sample_count
                ],
                "new_model",
                None,
            )
        )
        for later_code_length, hypothesis, model_id in options:
            description_length = earlier_code_length + later_code_length
            if description_length < best.description_length:
                best = MinimumDescriptionLengthAssignmentSelection(
                    selected_hypothesis=hypothesis,
                    change_interval_sample_count=change_interval_sample_count,
                    destination_model_id=model_id,
                    description_length=description_length,
                    new_model_learner_name=new_model_learner_names_by_change_interval_sample_count[
                        change_interval_sample_count
                    ]
                    if hypothesis == "new_model"
                    else NO_NEW_MODEL_LEARNER_NAME,
                )
    return best
