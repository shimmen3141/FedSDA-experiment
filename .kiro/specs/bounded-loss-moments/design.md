# 設計: 有界損失の逐次集計

## Overview
旧Welfordの値と更新・平均/標本分散を、ID管理なしの再利用できる数値部品としてlearningへ置く。

## Boundary Commitments
### This Spec Owns
一系列の不変件数/平均/偏差平方和、既存seedの明示受取、1損失追加、2件以上の平均・不偏標本分散。
### Out of Boundary
モデル/クラスmap、seedのbatch算出、監視clip/1件条件、候補履歴、ID再採番/統計merge、モデル操作/学習。
### Allowed Dependencies
stdlibのみ。torch/NumPy/config/他の新部品・旧importなし。
### Revalidation Triggers
Pythonfloat演算順・seed保持・n<2のNone・値域の変更では上位monitor/候補履歴/登録seed・統合の接続を再検証する。

## Architecture
stdlibのfrozen値型とpure関数。汎用registry・継承・model Protocolは導入しない。上位がモデル全体/クラスごとに別値を保持し、帰属した損失だけを追加する。

## File Structure Plan
- 新規src/federated_learning_experiments/learning/loss_statistics/__init__.py: package責務のみ、API再exportなし。
- 新規同ディレクトリbounded_loss_moments.py: 不変moments・追加・推定・検査。
- 新規tests/refactoring/test_bounded_loss_moments.py: 旧直接oracle、seed/更新/拒否・副作用/上位接続。
- 変更tests/refactoring/test_single_run_dependency_boundaries.py: stdlib許可と禁止注入。
- 変更.kiro/steering/roadmap.mdとspec証拠: 完成範囲/後続。

## Requirements Traceability
| 条件 | 実装/検証 |
|---|---|
| 1.1, 1.2, 1.3 | frozen BoundedLossMoments、seed/空・型・有限・契約検査 |
| 2.1, 2.2, 2.3 | pure追加・旧Welford全値完全一致・更新前全検査/入力不変 |
| 3.1, 3.2, 3.3 | Noneとfrozen推定値・n>=2保存mean/M2除算・非零1件seed保持 |
| 4.1, 4.2, 4.3 | 旧全体/classとgetstats・上位test接続、immutable/RNG/既定型/device、AST/全golden/smoke/roadmap |

## Components and Interfaces
BoundedLossMoments(*, observed_loss_count:int, mean_loss:float, sum_squared_loss_deviations:float): frozen kw_only。__post_init__で契約検査・float正規化。n1/M2非零も受ける。seedの作り方は所有しない。
LossMeanAndSampleVariance(*, observed_loss_count:int, mean_loss:float, sample_variance:float): frozen kw_onlyな生成結果。n>=2だけでpure推定から生成。
accumulate_bounded_loss_observation(*, loss_moments:BoundedLossMoments, observed_loss:float)->BoundedLossMoments: exact型と全field・lossを再検査し、旧n→delta→mean→delta2→M2順を保持する。不正または出力契約外では入力を変えず拒否する。
estimate_loss_mean_and_sample_variance(*, loss_moments:BoundedLossMoments)->LossMeanAndSampleVariance|None: exact型/field再検査。n<2はNone、それ以外は保存mean/M2/(n-1)。旧(0,0)への対応はtestで明示し新互換読込みを作らない。
_validate_finite_nonnegative_number(*, specified_value, parameter_name, maximum_value=None)->float: builtin型/有限/非負と任意上限の検査。
_validate_loss_moment_fields(*, observed_loss_count, mean_loss, sum_squared_loss_deviations)->tuple[float,float]: count型/非負/有限float表現、平均[0,1]、M2有限非負、n0整合。検査後float値を返すが既存inputは更新しない。
_validate_loss_moments(*, loss_moments)->None: exact型・fieldを再検査する。既存instanceの__post_init__を再呼出して入力を正規化しない。

## Error Handling
TypeError/ValueErrorに項目名と理由。巨大整数count/M2のfloat Overflowも項目名付きValueErrorへ変換。bool/配列scalarの暗黙受理なし。不正inputで変更しない。

## Testing Strategy
task1: 旧_update_running_statsと_get_model_stats、空/1/2件/非零1件seed/サーバ由来M2=0・一定/交互/微小損失の全観測を直接比較、入力拒否・frozen/Noneと真の0区別。
task2: 旧_update_model_stats全体/classを直接実行し別値を同じ帰属順で更新、初期Welford状態と明示seedを比較。監視へ保存meanと旧baseline明示入力、候補へn>=2の平均を明示入力し、class統計と全体baselineを混同しない。RNG/default型/device/keywordも確認。
task3: stdlib only AST禁止注入・全tests旧11/最終3golden・旧/NumPy/torchなしsmoke、12条件・部分完成とroadmapをLuna最終GO前に更新。

