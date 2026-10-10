# Research & Design Decisions

## Summary

- **Feature**: `model-computation-measurement`
- **Discovery Scope**: Extension（計測を足す）＋ 完了済みの3つのmoduleの、結果を変えない変更
- **Key Findings**:
  - 旧の計数は、外側からの数え直しと、goldenの3ケースで完全に一致した（漏れ・重複なし）。
  - 新実装は、3箇所で、旧より多く順伝播を行っている（結果は同じ）。
  - goldenの7項目は、すべて、「共有部・概念固有部を通った標本数（学習と推論の別）」「概念固有部のoptimizerの更新回数」「検出器の更新回数と候補の数」から作れる。用途別の内訳は要らない。

## Research Log

### 旧の計数の定義

- **Sources Consulted**: federated_drift_experiment/clients/base.py（`_record_model_compute`、`compute_counters`を足す箇所）、clients/fedsda.py・clients/shared_backbone.py（同）、experiment.py（`_COMPUTE_COUNTER_KEYS`、`_add_telemetry_results`）、models.py（`SharedFeatureBackbone`、`ResidualConceptAdapter`、`SharedBackboneMLP`の`extract_features`・`forward_from_features`・`update`、`per_sample_error_and_prediction`）、drift_detectors/e_detector.py（`active_hypothesis_count`）、tests/test_proposed_regression.py。
- **Findings**:
  - `_record_model_compute(phase, examples, calls, backbone_examples, head_examples)`: 用途別の`{phase}_examples`と`{phase}_forward_calls`、`backbone_examples`、`head_examples`を足す。共有部の特徴を使い回す箇所（予測、再較正）では、`backbone_examples`が標本数、`head_examples`が標本数×モデル数。
  - `optimizer_steps`: 概念固有部のoptimizerの更新回数（モデルごとに1）。`backbone_optimizer_steps`: 共有部のoptimizerの更新回数。
  - `drift_detector_updates`: 検出器の部品（全体と、正解クラス）の更新ごとに1。`drift_detector_hypotheses`: 更新の後の「保持している候補の変化点の数×賭け率の数」。
  - 集計（`_add_telemetry_results`）: `compute_inference_examples_total`＝学習以外の6用途の標本数の合計。`compute_training_examples_total`＝学習の標本数。`compute_optimizer_steps_total`＝`optimizer_steps`。`compute_backbone_examples_total`・`compute_head_examples_total`。`compute_drift_detector_updates_total`・`compute_drift_detector_hypotheses_total`。全clientの合計。
  - 旧のモデルは、共有部を`self.backbone(x)`、概念固有部を`self.adapter(features)`で呼ぶ（`__call__`を通る）ので、moduleの順伝播のhookで、すべての計算を数えられる。
- **Implications**: 概念固有部の順伝播の標本数が、旧の「モデルへ入力した標本数」（用途別の合計）に当たる。勾配の有無が、学習と推論の別に当たる。

### 外側からの数え直し（実測）

- 方法: `torch.nn.modules.module.register_module_forward_hook`と`torch.optim.optimizer.register_optimizer_step_post_hook`を、旧の参加者の準備（`_setup_server_and_clients`）の後から、終端までの間、登録する。順伝播は、moduleの型（共有部、概念固有部）と、勾配が有効かで分ける。optimizerは、最初のパラメータの形（共有部の最初の層か）で分ける。
- 結果（sine2 / sea2 / mnist2。旧の計数＝hook）:
  - 共有部を通った標本数: 341786 / 58139 / 3054。概念固有部: 352366 / 58139 / 3121。
  - 勾配つき（学習）: 322048 / 52736 / 2224（共有部・概念固有部とも。旧の`training_examples`）。
  - 勾配なしの概念固有部: 30318 / 5403 / 897（旧の、学習以外の用途の合計）。
  - optimizerの更新: 概念固有部 10165 / 1648 / 73（旧の`optimizer_steps`・`head_optimizer_steps`）、共有部 4412 / 1648 / 73（旧の`backbone_optimizer_steps`）。
  - 参加者の準備の間、clientの計数は0（事前学習は、clientの計数に入らない）。
- 新の全体run（goldenの条件、sine2。新のmoduleの型`SharedFeatureExtractor`・`NonlinearResidualAdapter`で数えた。準備の後から）: 学習 322048、optimizer 10165 / 4412 は一致。概念固有部の推論 32375（旧30318。+2057）、共有部の推論 25514（旧19738。+5776）。
- 呼出し元ごとの内訳の比較から、差は3箇所（requirements.mdの冒頭）。それ以外の箇所の件数は、旧と一致した（予測 11361／共有部4500、標本ごとの損失 4500、取込みの統計 3332＋1404、再利用の評価 836、候補の学習の評価 298、候補検証の標本 210、採用の登録 53、クロス評価の損失だけ 437）。

### 新の3箇所

- クロス評価（runtime/client_model_cross_evaluation.py）: `evaluate_classifier_per_sample_bounded_losses`（順伝播1回）の後、正誤を比べるときに、`candidate_classifier(input_features)`を、もう一度呼ぶ。
- 警報時の区間の準備（runtime/alarm_training_interval_preparation.py）: 変更区間より前の観測のそれぞれに、`evaluate_classifier_per_sample_bounded_losses`を呼んで、結果の件数（1件）だけを確かめる（状態の更新より前の検査）。その後、`absorb_assigned_training_samples_into_held_model`が、同じ標本の損失を計算する。
- 集約後の再較正（runtime/post_aggregation_prediction_recalibration.py）: `_compute_pending_sample_loss_sequence`が、保有モデルごとに、`evaluate_classifier_per_sample_bounded_losses`（共有部から）を呼ぶ。予測（runtime/observed_sample_prediction.py）は、共有部の特徴を1回だけ計算する形になっている。

## Design Decisions

### Decision: 計算量は、PyTorchのhookで、実行中に外側から数える

- **Alternatives Considered**: (1)旧と同じく、各部品が、計数のownerへ足す——順伝播を行う全部品（標本の処理、警報、候補、クロス評価、再較正、学習）へ、計数の受渡しが入る。書き忘れ（漏れ）と、内側と外側の両方で足す（重複）が、起こりうる。部品を足すたびに、数え方を書く必要がある。(2)分類器の中に計数を持たせる——分類器の写し・作り直し（候補、配布、クロス評価）のたびに、計数の受渡しが要る。(3)moduleごとにhookを登録する——同じく、作る箇所のすべてで登録が要る。
- **Selected Approach**: 計測の区間（context manager）の間だけ、PyTorchの全体のhook（moduleの順伝播、optimizerの更新）を登録し、moduleの型と、勾配の有無で、数える。区間を出るときに、必ず外す。
- **Rationale**: 実際に行われた計算を数えるので、漏れ・重複が、原理的に起きない。部品を変えない。新しい手法（FedDriftほか）にも、同じ計測が使える。旧の計数と、この方法が一致することは、検査で確かめた。
- **Trade-offs**: hookは、processの全体に効く（区間の中の、別のモデルの計算も数える）。実験は、1 processで1 runを順に実行するので、問題にならない。入れ子の区間は、それぞれが数える。用途別の内訳と、clientごとの内訳は、得られない（goldenの7項目には要らない）。順伝播のたびに、Pythonの関数が1回呼ばれる（下書きの実測では、goldenの条件の全体runの時間は、ほぼ変わらなかった）。
- **Follow-up**: 用途別・clientごとの内訳が要るときは、区間を細かく区切る（計測の区間を、処理の単位で開閉する）か、別のspecで扱う。

### Decision: 手法の比較のために、全結合層の積和演算の数を、同じ計測で測る（2026-10-10、ユーザー決定）

- **Context**: 旧の計数（モデルへ入力した標本の数）は、モデルの大きさを反映しない。FedSDA（共有部1つ＋小さいアダプタ）とFedDrift（概念ごとに完全なモデル）、SINE（2次元、隠れ層32）とMNIST（784次元、隠れ層1568）で、同じ1件の重さが違う。共有部の件数、概念固有部の件数、学習の件数は、足し合わせる根拠がない。
- **Selected Approach**: 全結合層（`torch.nn.Linear`）の順伝播ごとに、積和演算の数（標本数×入力の次元×出力の次元）を足す。順伝播のhookが、層の形と入力を受け取るので、モデルの構造ごとの式を書かずに測れる。逆伝播は、勾配つきの順伝播ごとに、層の形から見積って、別の項目で持つ（重みの勾配と、入力へ戻す勾配。最初の層の入力へは戻さない）。
- **Rationale**: 単位が1つ（積和演算）になり、共有部と概念固有部、推論と学習、手法の間で、足し合わせ・比較ができる。逆伝播を別の項目にするので、「順伝播だけ」「逆伝播を含む」のどちらでも報告できる。
- **Trade-offs**: バイアスの加算、活性化関数、損失、optimizerの更新（パラメータ数に比例）は、数えない（積和演算に比べて小さい。慣例）。逆伝播は、実測ではなく、見積り。検出器の計算（候補×賭け率の数）は、単位が違うので、別枠のままにする。
- **Follow-up**: サーバの演算数（集約・統合・パラメータ距離で扱うパラメータ値の数）、ラウンドごとの系列、保有モデル数での正規化（計算量の合計÷client・標本ごとの保有モデル数の合計）は、次のspec。

### Decision: 学習と推論は、順伝播のときの、勾配の有無で分ける

- **Rationale**: 旧の「学習の標本数」は、勾配つきの順伝播の標本数と、「学習以外の用途の合計」は、勾配なしの順伝播の標本数と、goldenの3ケースで一致した。新実装でも、勾配つきの順伝播は、学習（共同更新、候補の学習、事前学習）だけである（呼出し元の内訳で確かめた）。

### Decision: 重複の解消は、有界損失の評価のmoduleへ、3つの入口を足して行う

- **Selected Approach**: `learning/prediction/classifier_bounded_loss_evaluation.py`へ、(a)1回の順伝播から、損失と、分類器の出力を返す関数、(b)共有部の特徴を1回だけ計算して、複数の分類器の損失を返す関数、(c)順伝播を行わない、入力の検査の関数、を足す。既存の関数は、(a)を使う形にして、挙動を変えない。
- **Rationale**: 損失の式と検査を、1箇所に保つ。3つの利用箇所は、呼ぶ関数を変えるだけ。

### Decision: 事前学習の計算は、計測に含めない

- **Rationale**: 旧の定義（clientの計算だけ）と、goldenに合わせる。計測つきの全体runは、参加者の準備が終わった時点の計数を控え、終わりの計数との差を返す。事前学習の計算を知りたいときは、控えた計数を読めばよい（結果に持たせる）。

### Synthesis

- **Build vs. Adopt**: PyTorchのhookを使う（自前の計数の配線を作らない）。
- **Simplification**: 用途別の内訳、時系列、実行時間は、作らない。

## Risks & Mitigations

- hookの外し忘れで、以後の計算に影響する — context managerの`finally`で外す。testで、区間の後と、例外の後に、登録が残らないことを確かめる。
- 重複の解消で、結果や、拒否の順が変わる — 既存の、実旧との対照と、拒否のtestを、そのまま通す。数値は、同じ演算（共有部→概念固有部）なので、変わらない見込み。実旧との対照で確かめる。
- 勾配の有無による分類が、将来の部品で崩れる（勾配つきの推論） — 計測の説明に、分類の規則を書く。goldenの照合が検出する。

## References

- [fedsda-run-metric-derivation](../fedsda-run-metric-derivation/)（指標の導出と、goldenの照合のtest）、[model-cross-evaluation](../model-cross-evaluation/)、[post-aggregation-prediction-recalibration](../post-aggregation-prediction-recalibration/)。
