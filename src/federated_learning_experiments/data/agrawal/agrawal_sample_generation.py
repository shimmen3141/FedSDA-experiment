"""借りたNumPy乱数で、Agrawal（agrawal4）の観測標本を供給する。

概念の定義はRiver 0.27.0の`river.datasets.synth.Agrawal`（`balance_classes=False`、`perturbation=0.05`）と同じ。
概念cは分類の関数c（0〜3）。特徴の生成・ラベル・摂動の順序と式はRiverの`__iter__`と同じにし、
Pythonの`random.Random`の`random()`をNumPyの`random_sample()`へ、`randint(a, b)`を`randint(a, b + 1)`へ置き換える。
hvalueの摂動は、Riverの`_perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)`の引数の並びのまま再現する
（結果は`(9 - zipcode) * 100000`か0になる。分類の関数0〜3はhvalueを使わない）。
"""

import numpy as np

from federated_learning_experiments.data.observed_streams import ObservedSample

_MAXIMUM_CONCEPT_COUNT = 4
_PERTURBATION = 0.05
# 特徴の名前と、0〜1へ写す固定の範囲（下端、幅）。摂動0.05の出力が取りうる値を覆う。
_FEATURE_SCALES = (
    ("salary", 20000.0, 130000.0),
    ("commission", 0.0, 75000.0),
    ("age", 20.0, 60.0),
    ("elevel", 0.0, 4.0),
    ("car", 1.0, 19.0),
    ("zipcode", 0.0, 8.0),
    ("hvalue", 0.0, 900000.0),
    ("hyears", 1.0, 29.0),
    ("loan", 0.0, 500000.0),
)


def compute_agrawal_class_label(
    *,
    salary: float,
    commission: float,
    age: int,
    elevel: int,
    car: int,
    zipcode: int,
    hvalue: float,
    hyears: int,
    loan: float,
    concept_id: int,
) -> int:
    """尺度を写す前の9特徴と概念（0〜3）から、Riverの`Agrawal`の分類の関数と同じラベルを返す。"""
    if type(concept_id) is not int:
        raise TypeError("concept_idには整数を指定してください。boolは受理しません。")
    if concept_id == 0:
        return int((age < 40) or (60 <= age))
    if concept_id == 1:
        if age < 40:
            return int((50000 <= salary) and (salary <= 100000))
        if age < 60:
            return int((75000 <= salary) and (salary <= 125000))
        return int((25000 <= salary) and (salary <= 75000))
    if concept_id == 2:
        if age < 40:
            return int((elevel == 0) or (elevel == 1))
        if age < 60:
            return int((elevel == 1) or (elevel == 2) or (elevel == 3))
        return int((elevel == 2) or (elevel == 3) or (elevel == 4))
    if concept_id == 3:
        if age < 40:
            if (elevel == 0) or (elevel == 1):
                return int(25000 <= salary <= 75000)
            return int(50000 <= salary <= 100000)
        if age < 60:
            if (elevel == 1) or (elevel == 2) or (elevel == 3):
                return int(50000 <= salary <= 100000)
            return int(75000 <= salary <= 125000)
        if (elevel == 2) or (elevel == 3) or (elevel == 4):
            return int(50000 <= salary <= 100000)
        return int(25000 <= salary <= 75000)
    raise ValueError(f"concept_idには0以上{_MAXIMUM_CONCEPT_COUNT}未満を指定してください。")


class AgrawalSampleGenerator:
    """runのRandomStateを保持し、特徴を固定の範囲で0〜1へ写して返す。"""

    def __init__(
        self, *, numpy_random_generator: np.random.RandomState, concept_count: int
    ) -> None:
        if type(concept_count) is not int:
            raise TypeError("concept_countには整数を指定してください。boolは受理しません。")
        if not 1 <= concept_count <= _MAXIMUM_CONCEPT_COUNT:
            raise ValueError(
                f"concept_countには1以上{_MAXIMUM_CONCEPT_COUNT}以下を指定してください。"
            )
        self.numpy_random_generator = numpy_random_generator
        self.concept_count = concept_count

    def _random(self) -> float:
        return float(self.numpy_random_generator.random_sample())

    def _randint(self, lowest: int, highest: int) -> int:
        """Riverの`randint(lowest, highest)`（両端を含む）。"""
        return int(self.numpy_random_generator.randint(lowest, highest + 1))

    def _perturb_value(self, value, minimum, maximum, value_range=None) -> float:
        """Riverの`Agrawal._perturb_value`と同じ式と切り詰め。"""
        if value_range is None:
            value_range = maximum - minimum
        value += value_range * (2 * (self._random() - 0.5)) * _PERTURBATION
        if value < minimum:
            value = minimum
        elif value > maximum:
            value = maximum
        return value

    def generate_sample(self, *, concept_id: int) -> ObservedSample:
        """指定した概念から、観測標本を1つ生成する。概念を検査してから乱数を使う。"""
        if type(concept_id) is not int:
            raise TypeError("concept_idには整数を指定してください。boolは受理しません。")
        if not 0 <= concept_id < self.concept_count:
            raise ValueError(f"concept_idには0以上{self.concept_count}未満を指定してください。")
        salary = 20000 + 130000 * self._random()
        commission = 0 if (salary >= 75000) else (10000 + 75000 * self._random())
        age = self._randint(20, 80)
        elevel = self._randint(0, 4)
        car = self._randint(1, 20)
        zipcode = self._randint(0, 8)
        hvalue = (8 - zipcode) * 100000 * (0.5 + self._random())
        hyears = self._randint(1, 30)
        loan = self._random() * 500000
        class_label = compute_agrawal_class_label(
            salary=salary,
            commission=commission,
            age=age,
            elevel=elevel,
            car=car,
            zipcode=zipcode,
            hvalue=hvalue,
            hyears=hyears,
            loan=loan,
            concept_id=concept_id,
        )
        # 摂動（ラベルを決めた後。Riverと同じ順序）。
        salary = self._perturb_value(salary, 20000, 150000)
        if commission > 0:
            commission = self._perturb_value(commission, 10000, 75000)
        age = round(self._perturb_value(age, 20, 80))
        hvalue = self._perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)
        hyears = round(self._perturb_value(hyears, 1, 30))
        loan = self._perturb_value(loan, 0, 500000)
        raw_values = dict(
            salary=salary,
            commission=commission,
            age=age,
            elevel=elevel,
            car=car,
            zipcode=zipcode,
            hvalue=hvalue,
            hyears=hyears,
            loan=loan,
        )
        float64_feature_values = np.array(
            [(float(raw_values[name]) - lowest) / width for name, lowest, width in _FEATURE_SCALES],
            dtype=np.float64,
        )
        float32_feature_values = float64_feature_values.astype(np.float32)
        return ObservedSample(
            feature_values=tuple(float(feature_value) for feature_value in float32_feature_values),
            class_label=class_label,
        )
