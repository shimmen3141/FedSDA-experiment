# LEGACY-017: クロス評価の読取りで、現行モデルの空の学習データの列が作られる

- 発見日: 2026-10-10（独立レビューの指摘による）。対象: 旧`federated_drift_experiment/clients/base.py::_cross_evaluation_data`（529〜543行）。固定基準748c3aa。
- 発見spec: model-cross-evaluation。種別: 読取りのつもりの処理が、状態を変える。状態: 再現済み・未修正。正常な経路（ラウンドの中のクロス評価）で起きるかどうかは未確認。

## 再現と観測

2026-10-10、Windows基準環境で、`tests/refactoring/test_model_cross_evaluation.py`の`test_cross_evaluation_of_current_model_does_not_create_empty_training_collection`が再現する。実旧の事前学習の結果から、実`__init__`で作った`ResidualAdapterRestartingSoftRoutingFedSDAClient`（標本を1件も処理していない。現行モデルはID 0、`train_data_store`は空）へ、`evaluate_model(params, 0)`と`evaluate_model_diagnostics(params, target_model_id=0, include_class_correctness=True)`を呼ぶ。

- どちらも、件数0を返す（評価標本も学習データもない）。
- 呼出しの後、`train_data_store`は`{0: []}`になる（呼出しの前は`{}`）。

## 原因

`train_data_store`は`defaultdict(list)`である。`_cross_evaluation_data`は、対象のモデルの評価標本が6件未満で、対象が現行モデルのとき、`len(self.train_data_store[target_model_id]) > 10`を評価する。キーがなければ、この読取りで、空の列が作られる。

## 影響

- 評価の結果は変わらない。`get_held_model_ids`も変わらない（現行モデルのIDは、学習データの有無によらず含まれる）。
- `train_data_store`のキーの並びが変わる。集約は、保有モデルごとに学習データの件数を読むので、空の列があっても、ない場合と同じ件数（0）になるはずである（コードの読みによる。実行では確かめていない）。
- 現行モデルが学習データを1件も持たない状態で、クロス評価の対象になる必要がある。ラウンドの中のクロス評価は、グローバルモデルが2つ以上あるラウンドで行われる。新実装の対照test（毎ラウンドのクロス評価、8条件×15ラウンド、3 client）では、この状態は起きなかった（評価の後の学習データの並びが、実旧と新で一致した）。最終3goldenの条件で起きているかどうかは、調べていない。

## 新実装の扱い

- `evaluate_candidate_model_on_target_model_samples`は、学習データを、状態を変えずに読む（空の列を作らない）。上の再現のtestが、この違いを固定している。
- 全体runのgoldenとの照合で、学習データの並びに由来する差が出た場合は、ここを最初に疑う。

関連: [model-cross-evaluationのdesign.md](../../../.kiro/specs/model-cross-evaluation/design.md)の「旧と違う点」。
