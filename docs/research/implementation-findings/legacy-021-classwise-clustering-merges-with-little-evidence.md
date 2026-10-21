# LEGACY-021: クラス別の正誤の判定が、評価の標本が少ないと、違う概念のモデルを統合する

- 発見日: 2026-10-22。対象: `src/federated_learning_experiments/methods/fedsda/consolidation/model_clustering_calculations.py`の`compute_classwise_unique_correctness_decision_score`（旧の`class_functional_confidence`を移植したもの）と、判定の値の閾値0.1、対の評価の標本の最小数`_MINIMUM_PAIR_EVALUATION_SAMPLE_COUNT = 5`（`runtime/fedsda_final_configuration_run_settings.py`）。発見の経緯: 初期モデルの統計を交差検証で取る変種（[LEGACY-020](legacy-020-optimistic-initial-model-loss-statistics.md)）で、MNIST4・間隔100・seed 9の精度が下がった原因の調査。
- 種別: 旧から引き継いだ判定の弱点。状態: 未対応（2026-10-22に記録）。

## 内容

判定の値は、クラス別の「片方のモデルだけが正解した割合」の、同時信頼下限（Wilson、信頼水準0.95、Bonferroni補正）の最大。0.1以下なら同じクラスタにする。信頼下限は、片方だけが正解した件数が0なら、評価の標本数に関わらず0になる。標本数が下限（5件）以上のクラスが無ければ、全体の件数を使う。

この判定は「違いの証拠が十分にあるときだけ、別のモデルとして残す」形なので、評価の標本が少ないと、違う概念のモデルでも、違いの証拠が足りずに統合される。

## 起きた場面（手元、Windows、基準版10の既定＋初期モデルの統計の交差検証）

MNIST4・間隔100・seed 9（各5000標本・10 client、精度91.97%。同じ条件の既定は94.34%）の40回目の同期で、違う概念のモデル0と1を、判定の値0.0で統合した。そのときの入力（Git管理外の一時scriptで記録、`results/fedsda-provisional-candidate-expert/20261022_mnist4-cross-validated-initial-statistics-diagnosis/merge_inputs_seed9.json`）は、評価した標本が全体で28件、片方だけが正解したのは1件（小さいIDのモデルだけ）。クラス別は10クラスで1〜5件ずつ、5件以上のクラス（2つ）では、片方だけが正解した件数が0。統合の後、モデルは3つ（MNIST4の概念は4つ）のまま、後半の精度が既定より4〜6pp低いままだった。

既定の同じseedでは、この統合は起きない（序盤の軌跡が違い、40回目の同期の場面に入らない）。交差検証が原因ではなく、軌跡が変わったことで、この弱点が表に出たとみる。

## 影響と直す方法の案（未検討）

- 評価の標本が少ない対では、統合しない（違いの証拠の不足を、同じ概念の証拠とみなさない）。例えば「同じ」とみなす側も検定する（同等性の検定）、または、評価の標本の最小数を、判定の値の信頼区間の幅が閾値を下回る数にする。
- 同じ弱点は、既定のrunでも起こり得る（頻度は未確認）。本規模のrawの`model_pair_clustering`で、違う概念の対の誤った統合の件数と、その評価の標本数を数えると確かめられる（記録の表に評価の標本数は無いので、クロス評価の記録の表と突き合わせる必要がある）。
