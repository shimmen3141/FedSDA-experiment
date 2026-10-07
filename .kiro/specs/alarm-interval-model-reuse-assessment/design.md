# 設計 revision2

## Overview

警報区間の損失を算出するruntimeと、有限の平均/履歴基準から適合を判定するmethods部品を分ける。既存の区間損失評価・履歴基準選択・初期値選択を再利用する。

## Boundary Commitments

### This Spec Owns

- 順序付き評価平均と履歴平均から適合列を生成する純粋判定、不変の結果record。
- 公開保有状態/統計の読取り→各分類器の区間損失平均→警報区間基準選択→純粋判定という組立。
- 成功/入力拒否時の状態とRNG保持、同率は入力順という区間独自の規則。

### Out of Boundary

区間切出し、最小区間件数、前区間処理、学習/吸収/帰属変更、候補初期値の生成やsession開始、通知/FIFO/episode/active state、I/O、計算量診断、新client/全run、旧/config/private import、alias。

### Allowed Dependencies

- methods/alarm_interval_model_reuse_assessment.py: exact `dataclasses.dataclass`と`math.isfinite`のみ。runtime/Tensor/config/将来検証評価を参照しない。
- runtime/alarm_interval_model_reuse_assessment.py: exact `math.isfinite`、`torch.Tensor`、`torch.mean`、公開`HeldModelTrainingStateRegistry`、`ModelAndClassLossStatisticsStore`、`evaluate_classifier_per_sample_bounded_losses`、`select_alarm_interval_reuse_baseline_mean_loss`、`AlarmIntervalModelReuseAssessment`、`assess_alarm_interval_model_reuse`のみ。定義moduleからの直接import。module丸ごと/再export/子module/privateは不可。
- testでだけ旧oracleを参照。新しい設定型/RunSettings項目は追加せず明示閾値を渡す。

### Revalidation Triggers

損失精度/平均演算、保有順、履歴の2件/零判定、初期値選択の候補範囲、選択同率規則、record fields、数値閾値を変えたら実旧対照/接続を再検証。命名/責務変更は承認を解除。前区間吸収後の評価順は後続警報組立で検証する。

## Architecture / System Flow

```mermaid
flowchart LR
 R[保有状態と統計] --> E[各モデルの区間平均]
 E --> B[使用可能な履歴基準だけ保持]
 B --> A[純粋な差分判定]
 A --> O[不変評価情報]
 O --> C[既存の初期値選択または呼出側の再利用]
```

## File Structure Plan

| 操作 | ファイル | 責務 |
| --- | --- | --- |
| 作成 | src/federated_learning_experiments/methods/fedsda/candidate_model_selection/alarm_interval_model_reuse_assessment.py | 不変record・純粋な適合判定 |
| 作成 | src/federated_learning_experiments/runtime/alarm_interval_model_reuse_assessment.py | 既存公開評価/履歴基準と判定の組立 |
| 作成 | tests/refactoring/test_alarm_interval_model_reuse_assessment.py | 純粋判定・実旧区間評価・拒否・初期値選択/学習接続 |
| 変更 | tests/refactoring/test_single_run_dependency_boundaries.py | exact2module/両resolver/注入 |
| 更新 | 本specの正本/証拠、steering resume/roadmap | 承認/完了/再開 |

## Components & Interfaces / Contract

`AlarmIntervalModelReuseAssessment`: frozen/kw_only、全必須`baseline_supported_interval_mean_losses_by_model_id:tuple[tuple[int,float],...]`、`reusable_mean_losses_by_model_id:tuple[tuple[int,float],...]`。property`selected_reuse_model_id:int|None`は適合列のmean最小ID、空ならNone。同率はtuple先着。record constructorは信頼した内部producer用で再検査せず、入力検査は以下公開判定へ置く。

`assess_alarm_interval_model_reuse`: 全keyword必須`baseline_supported_interval_mean_losses_by_model_id`（上記tuple）、`reuse_baseline_mean_losses_by_model_id:dict[int,float]`（評価済みIDと完全に同じkey集合）、`maximum_alarm_interval_mean_loss_increase:float`→record。exact tuple/pair/dict、IDはbool以外builtin int・一意（負可）、mean/baselineはbool以外builtin int/floatの有限[0,1]でbaselineは非零、閾値は有限floatへ表現できる非負builtin int/float（bool/巨大int overflow拒否）。型はTypeError、形状/値/重複/key不一致はValueError。全検査後にmean−baseline<=thresholdで同順適合列を返す。空評価＋空基準は正常。平均のfloat32値をPython sumで再計算しない。

`evaluate_held_models_for_alarm_interval_reuse`: 全keyword必須`input_features:Tensor`、`observed_class_labels:Tensor`、`held_model_training_state_registry:HeldModelTrainingStateRegistry`、`loss_statistics_store:ModelAndClassLossStatisticsStore`、`maximum_alarm_interval_mean_loss_increase:float`→record。

1. runtime helperでexact owner型と閾値を分類器評価前に検査する。閾値はbuiltin int/float（bool除外）、float変換時のoverflowをValueErrorへ変換、isfiniteと非負を確認する。保有snapshot非空、現行ID引数を設けない。純粋判定を空入力で事前呼出しする方式は採用しない。
2. 保有順に公開損失評価を呼び、`torch.mean(per_sample_bounded_losses).item()`を使う。特徴/ラベルのCPU float32 strided・非空shape[N,F]/[N,1]、有限値・整数ラベル範囲は既存公開損失APIへ委譲する。モデルの契約外改変は対象外。
3. 各評価後に公開統計を取得し、overall momentsを警報区間baseline選択へ渡す。Noneなら評価候補へ含めず、非零baselineならordered tupleと基準dictへ含める。欠損/不足でもforwardは実行する。バッチ検査は各公開損失APIのforward前に行う。異なる特徴数の保有モデルがある場合、先行モデルのforward後に後続モデルでshape拒否されることを許す。先行forwardをなかったことにしないが、既存ResidualAdapterClassifierのforwardはparameter/grad/optimizer/owner/RNGを変更しないので、拒否時の状態保持は満たす。全モデルを事前検査する保証や評価回数zeroの保証は設けない。非公開forward差替え/モデル改変は契約外。
4. 完成した平均/基準と閾値を公開純粋判定へ一度渡す。純粋判定は単独でも呼べる公開APIなので自己の入力として閾値を再検査する。runtime側の早期検査とpure側の境界検査は意図した重複で、共有private importや余分な空判定を設けない。不要な型/ownerやバッチコピーは作らない。
5. parameter/grad/optimizer・owner・全3RNGを保持し、診断記録追加や学習を行わない。OOM/overflow/非公開改変/同時更新は保証外。

初期値選択への明示test-only接続は、baseline_supported_interval_mean_losses_by_model_idを既存selectorのevaluated_mean_losses_by_model_id引数へ渡す。呼出側で対応を明示し、上流APIを改名しない。3方式の値・順序・fallback/独立snapshotを旧と照合し、そのsnapshotで既存session開始→実観測を行う。全警報処理への接続を実装済みと主張しない。

## Testing Strategy / Traceability

| 要求 | 部品/flow | 証拠 |
| --- | --- | --- |
| 1.1 | runtime flow2〜3、既存損失/基準API | 実旧_resolve_driftの評価呼出し/候補列を観測、欠損/0/1/2+/零基準もforward、順序一致 |
| 1.2 | methods公開判定、runtime flow4 | 制御meanと実NN、float境界直前/同値/直後、差分の符号 |
| 2.1 | methods recordのselected property | 負ID/逆ID順/同率/現行が後ろ、適合なし/複数 |
| 2.2 | methods frozen record、runtime flow2〜5 | frozen/kw_only、全owner/parameter/grad/optimizer/3RNG保持 |
| 3.1 | methods入力検査、runtime flow1〜4/既存損失検査 | 純粋値/owner/閾値/バッチ拒否、後半shape拒否でも全状態保持 |
| 3.2 | methods判定＋runtime全flowとtest-only初期値/開始接続 | 実旧CPU2/4class×複数モデル、3初期値方式と実開始/観測、source代表変異とbyte復元 |

全体完了ゲートはbriefに従う。exact AST注入RED→guard/両resolver→GREEN、新CPU2/4class、全pytest/JUnit/旧11最終3golden、Ruff/Pyright/pip、固定旧748c3aa差分、承認/source hashとtested commit、全task独立レビュー後の別fresh feature GO。全suiteは主担当実測/JUnit照合を用い独立全再実行は必須にしない。
