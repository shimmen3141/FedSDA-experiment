# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `capture_case_reusing_runs` | `tests/refactoring/behavior_golden_cases.py`の関数 | ケースと版から、goldenと比べる形の結果と版の差の事象の数を返す。同じprocessの中で、同じ完全なrun設定のrunは1回だけ実行し、使い回す（経路の記録と必須の経路の確認は版ごと）。testの側だけが使う。`capture_case_and_count_divergence_events`（毎回実行する。生成が使う）とは、使い回すかどうかだけが違う |

新しいmodule・classは無い（testのhelperと、`tests/conftest.py`の定数・関数の追加）。
