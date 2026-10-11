# Design Document: feddrift-detection-batch-processing

## Overview
移植済み数値判定から、旧FedDriftの検出バッチ1回の処理を完結させるruntimeのprocessorを作る。前回の最小平均損失だけを新ownerが保持し、一覧・統計・帰属・標本・採番・送信保留は既存ownerを再利用する。

## Boundary Commitments
### This Spec Owns
- バッチの入力検査、数値判定への接続、非検出の既存モデルへの割当、新規モデル生成と登録、前回値更新、評価標本保持をこの順で行う処理。
- 即時送信可能なsnapshotを、既存の一保留枠へ登録する操作。
### Out of Boundary
- バッファ蓄積・標本予測・学習要求、計測、適応記録保存、サーバ同期、設定preset・全体run・goldenの移植。旧early_stoppingによるrandperm。
### Allowed Dependencies
- runtimeからmethods/feddriftの数値判定、既存learningの分類器・snapshot・optimizer・状態owner・統計、evaluationの評価標本owner、既存の送信保留ownerへ依存する。
- dataclass、math有限検査、random.Random、torch.cat。旧コード・test・CLI・保存・描画へ依存しない。methodsからruntimeへ逆依存を足さない。
### Revalidation Triggers
- モデル生成/RNG・初期統計・抽出順・ID更新順・owner契約の変更時は本specの実旧対照を再実行する。
- 後続のclient組立とサーバ同期では、バッファ・位置記録・一時ID確認・局所学習・全体runを別に照合する。

## File Structure Plan
| ファイル | 責務 |
|---|---|
| `runtime/feddrift_detection_batch_processing.py`（src package内） | processorと不変の分岐結果 |
| `methods/fedsda/model_registration/pending_model_upload.py` | 即時送信操作の追加 |
| `tests/refactoring/test_feddrift_detection_batch_processing.py` | 実旧との多バッチ・分岐・統計・RNG・ownerの照合、異常系 |
| `tests/refactoring/test_single_run_dependency_boundaries.py` | 許可依存への登録と逆方向の拒否 |
| `tests/refactoring/fresh_process_smoke.py` | 新分類器から実processorの生成・割当を実行 |

## Components and Interfaces
- `PendingModelUploadState.queue_ready_model_upload(model_id, parameter_snapshot)`は既存のPendingModelUpload検査後、残境界数0で一枠を置換する。遅延操作の契約を変えない。
- `FeddriftDetectionBatchProcessor`のconstructorはモデル一覧、統計、現在帰属、学習標本、評価標本、採番、送信保留、Python Random、新規optimizerのAdam/SGD設定、非負有限float閾値と初回Noneまたは有限[0,1]floatの前回値を受け取る。
- `process_detection_batch(training_samples, processed_sample_count)`はexact tupleのexact ObservedTrainingSample列を受け取る。各payloadはCPU float32 shape[1,F]/[1,1]。processed_sample_countはbuiltin intで標本件数以上。範囲は`count-len(samples)`〜`count-1`。
- 検査はowner exact型、全HeldModelTrainingStateのID/optimizerとparameterの対応、全classifierの構造/寸法/有限値、現在IDの存在、各標本を先に行う。前回値と閾値はconstructorで検査する。新規IDは一覧・統計・標本・送信保留との衝突を生成前に拒否する。
- 数値判定は既存assessをそのまま呼ぶ。非検出は選んだモデルの標本別再評価とWelford統計を別storeで準備し、現在帰属と標本・統計を更新する。検出は現在モデルと同寸法の独立分類器を構築してパラメータを複写し、空の新optimizerとバッチ初期統計、送信snapshotを準備してから採番と一覧・統計・保留・帰属・標本を更新する。初期学習しない。
- 評価保持は既存storeへ同じRandomで渡す。一時IDの無視、抽出・容量制限・空抽出の構造を変えない。
- `FeddriftDetectionBatchTransition`は分岐名（current_model_retained/held_model_selected/new_model_created）、割当ID、バッチ先頭/末尾、次の前回値を持ち、constructorで型・位置・有限範囲・分岐とID符号を検査する。clientが分岐と位置から旧drift_type/位置列を導く。processorは結果を先に構築してからownerを更新する。
- classifier・optimizerは借用し、Tensorは標本参照のまま保持する。新モデルとsnapshotは独立。既存parameter・grad・mode・optimizerを変更しない。

## 旧と違う点
- 不正入力の事前拒否、Tensor参照を型付きrecordで扱うこと、グローバルrandomでなく明示Randomを使うこと。testで同じ乱数状態を渡す。
- 検出/切替の位置列そのものは所有せず、バッチ結果として返す。標準の正常入力での値・処理順・乱数消費は変えない。

## Requirements Traceability
| 要求 | 実現と検証 |
|---|---|
| 1.1, 1.2, 1.3 | 実旧の非検出・現在維持/他モデル切替、一標本再評価、統計と標本順 |
| 2.1, 2.2, 2.3 | 実旧の生成・parameter/RNG・初期統計・ID・即時保留・連続生成 |
| 3.1 | 正式ID/一時ID・抽出0/容量超過・Random状態の実旧照合 |
| 3.2 | 後ろの不正標本/モデル・手組record・bool/欠落ID・衝突を更新とRNG前に拒否 |
| 3.3 | 依存検査、共用fresh smoke、全pytestと旧/提案golden |

## Testing Strategy
実旧FedDriftClientを実__init__と実モデルで作り、_detect_and_assignを代役なしで実行する。新に同じ初期parameter、統計、Random/Torch状態を渡し、分岐・位置・全owner・新optimizer初期状態・parameter identity/RNGをバッチごとに照合する。二値/多クラス、1件部分バッチ、複数モデル、現在≠最小の生成、連続生成を含む。検証順はRED→GREEN→全task後の独立Luna medium（Haiku利用不能）→有用な指摘の反映→固定Windows全pytest・Ruff・Pyright・pip check。WSLと変異は既定では実行しない。
