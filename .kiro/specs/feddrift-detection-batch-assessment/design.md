# Design Document: feddrift-detection-batch-assessment

## Overview

旧`BaseFedDriftClient._detect_and_assign`の「全モデルの平均損失→最小の選択→閾値比較」を、状態を持たない1つの関数へ移す。標本別有界損失は、既存の分類器の評価を再利用する。平均は旧と同じTensorのmeanからPythonのfloatへ変換する。

これは数値の一致が繊細な部分だけを切り出すspecである。clientの全体や実験の入口へ接続したという完了主張はしない。新実装だけの共用smokeから実際の分類器で呼び出す。

## Boundary Commitments

### This Spec Owns
- 検出バッチに対する保有順の損失評価、最小の選択、閾値比較、不変な結果。
- 受け取るモデル列・バッチ・前回値・閾値の検査。全入力を順伝播前に検査する。

### Out of Boundary
- clientの状態、判定後の処理、モデルの構築とoptimizer、サーバ、run設定と選択肢、全体runの照合。
- FedSDAの判定と旧実装・goldenの変更。LEGACY-018の修正。

### Allowed Dependencies
- 標準のdataclass・有限値検査、PyTorch Tensor。
- `learning/models/residual_adapter_classifier.py`の分類器と独立モデル構造の定数。
- `learning/prediction/classifier_bounded_loss_evaluation.py`の入力検査と標本別有界損失の評価。
- `methods/feddrift`からruntime・evaluation・旧実装へ依存しない。optimizerを使わないため学習bindingは受け取らない。

### Revalidation Triggers
- 有界損失の計算、平均の計算、入力契約またはモデル保有順が変わったら、本specの旧対照を再実行する。
- 後続clientの接続時は、前回値の更新・切替・乱数の消費・計数と全体runを追加で旧対照する。

## Architecture

`methods/feddrift`の関数が、既存learningの評価を呼ぶ。返り値はモデルの参照を保持しない。
新規ライブラリは追加しない。バッチの順・保有順を並べ替えず、全モデルを同じバッチで個別に評価する。共有部の評価をまとめる方式は独立モデルでは使わない。

## File Structure Plan

| ファイル | 責務 |
|---|---|
| `src/federated_learning_experiments/methods/feddrift/__init__.py` | 新手法のpackage境界（再exportなし） |
| `src/federated_learning_experiments/methods/feddrift/detection_batch_assessment.py` | 結果型と検出バッチの判定 |
| `tests/refactoring/test_feddrift_detection_batch_assessment.py` | 実旧の判定との照合、入力拒否と状態保持 |
| `tests/refactoring/test_single_run_dependency_boundaries.py` | 新moduleの許可依存の登録と上位依存の拒否 |
| `tests/refactoring/fresh_process_smoke.py` | 新分類器からの独立モデル評価と判定の呼出し |

## Components and Interfaces

`assess_feddrift_detection_batch`は、保有順の`tuple[(model_id, classifier), ...]`、shape[N,F]の入力、shape[N,1]の観測ラベル、前回の最小平均損失（floatまたはNone）、非負有限floatの閾値を受け取り、`FeddriftDetectionBatchAssessment`を返す。

- モデルIDはbuiltin int（負の一時IDも可）、重複不可。分類器は既存のexact型で、独立モデル構造だけを受理する。全モデルの入力幅とクラス数を一致させる。
- 入力とラベルは既存のCPU float32・非空・有限値・クラス範囲の契約を、全モデルについて先に検査する。前回値は有限な0〜1のfloat、閾値は有限な0以上のfloat（boolは拒否）。
- 平均は`float(per_sample_bounded_losses.mean().item())`。最小更新には厳密な`<`、ドリフトには厳密な`>`を使う。初回は偽。
- 結果型は保有順のモデル別平均損失、最小モデルID、最小平均損失、標本数、検出のboolを持つ。constructorも型・範囲・重複と「最初の最小」との整合を検査する。
- パラメータ、勾配、学習モードと乱数を保存・復元して隠す処理は追加しない。既存のno_grad評価はそれらを変更しない。

## Requirements Traceability

| 要求 | 実現と検証 |
|---|---|
| 1.1, 1.2, 1.3 | 全モデル評価と結果型。二値・多クラス、同率、非連続・負のIDを実旧と照合 |
| 2.1, 2.2, 2.3 | 比較判断。初回、未超過、等号、超過、1標本の部分バッチを実旧と照合 |
| 3.1, 3.2 | 入力検査とno_grad評価。後ろの不正モデルでもforwardなし、既存勾配とモード・乱数の保持 |
| 3.3 | 依存境界・共用fresh smoke・全pytest（旧goldenを含む） |

## Testing Strategy

旧`FedDriftClient`を`__new__`で作り、モデル・前回値・閾値を与えて実際の`_detect_and_assign`を呼ぶ。判定後のモデル生成・割当・評価標本保持だけを代役にして、その分岐と評価された平均損失を観測する。新モデルには実旧`SimpleMLP`と同じパラメータを与える。旧の式をtestへ複製して正解にしない。

全pytestは固定Windows環境で実行し、旧11ケース・最終3ケースのgoldenと既存構成の照合を維持する。独立レビューは全taskの後に1回。数値比較と更新境界が対象なのでHaiku 5.5を優先し、利用不能ならLuna mediumへ代替する。WSLと変異テストは既定では実行しない。
