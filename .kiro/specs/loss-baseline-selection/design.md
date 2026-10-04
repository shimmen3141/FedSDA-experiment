# 設計: 用途別損失基準値
## Boundary Commitments
### This Spec Owns
一系列の任意集計から監視/警報区間再利用/警報後履歴の平均を選ぶpure関数。
### Out of Boundary
モデル/class map、統計更新/seed/merge、参照選択・候補採否・学習・snapshot取得時機、baselineの設定追加。
### Allowed Dependencies
stdlibとexact module federated_learning_experiments.learning.loss_statistics.bounded_loss_moments、および宣言型BoundedLossMomentsだけ。private validatorや推定関数、旧/NumPy/torch/上位importなし。
### Revalidation Triggers
n閾値・clip順/値・ゼロ扱い変更時は旧監視、警報区間再利用と警報後参照選択を再検証。アルゴリズム変更は別spec。

## Architecture
手法固有の方針をmethods/fedsda/loss_statisticsへ配置。learningの集計は方針を所有しない。model IDも引数へ含めない。既存stdlib frozen値のpublic constructorによる独立コピーで入力を検査・float化し、入力instanceは変更しない。

## File Structure Plan
- src/federated_learning_experiments/methods/fedsda/loss_statistics/__init__.py: packageの日本語責務のみ。再exportなし。
- 同loss_baseline_selection.py: 3pure選択とoptional入力の検査。
- tests/refactoring/test_loss_baseline_selection.py: 旧直接oracle、拒否/非変更/共有状態、監視/参照選択への明示test接続。
- tests/refactoring/test_single_run_dependency_boundaries.py: exact module+宣言型だけ許可する例外と禁止/許可注入。
- roadmap/spec証拠: 完成境界と後続。

## Components and Interfaces
全公開関数はkeyword-only loss_moments:BoundedLossMoments|Noneを受ける。
select_loss_monitoring_baseline_mean_loss ->float: 欠落/n0は.01、n1以上はmin(1-1e-6,max(.01,保存mean))。
select_alarm_interval_reuse_baseline_mean_loss ->float|None: 欠落/n<2/保存mean0はNone、それ以外は保存mean。
select_post_alarm_reference_historical_mean_loss ->float|None: 欠落/n<2はNone、それ以外は保存mean（0も保持）。
_validate_optional_loss_moments ->BoundedLossMoments|None: NoneならNone、exact型を検査してpublic constructorへ全fieldを渡す。検証済みコピーを返し、既存inputを正規化/変更しない。エラーはloss_momentsまたはfield名/理由付きTypeError/ValueError。

## Requirements Traceability
| 条件 | 検証 |
|---|---|
| 1.1,1.2,1.3 | 旧監視直接oracleと明示全体平均、公的監視接続、n0/1/2、clip両境界 |
| 2.1,2.2,2.3 | 旧getstats/resolve drift/_begin_forward_validation直接oracle、不足と0の差、候補への明示履歴入力 |
| 3.1 | 型/forgedfield拒否、入力不変、共有RNG/default型/device |
| 3.2 | 上記旧oracle+新monitor/reference選択test |
| 3.3 | AST exact依存・全golden・独立stdlib smoke、部分完成記録 |

## Testing Strategy
task1 TDDで3関数とpublic constructor検査を実装し、n0/1/2、平均0/.005/.01/.2/1-1e-6/1、非零1件M2・サーバM2=0、旧3用途直接oracleで選択を照合。
task2 test-only上位monitor/参照選択、入力不変/keyword/RNG/device/forged値を検証。
task3 AST許可/禁止注入・全tests旧11/最終3golden・独立smoke・9条件/roadmap/最終GO。既存venv/pytest基盤を使用、環境新設なし。
