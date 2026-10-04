# 設計: サーバ向け損失平均集約
## Boundary Commitments
### This Spec Owns
一モデルの参加済みクライアント全体損失集計の順序付き入力から、件数加重平均・合計件数・M2=0を返す純関数。
### Out of Boundary
参加判定/モデル・学習データ/パラメータFedAvg/通信/ID/サーバ状態、クラス集約、分散のpooling、監視・候補進行・新全体run。
### Allowed Dependencies
stdlib、exact federated_learning_experiments.learning.loss_statistics.bounded_loss_moments moduleと公開型BoundedLossMomentsだけ。
同moduleの別関数/private/子module、store/別module/methods/runtime/旧/torch/NumPyは禁止。
### Revalidation Triggers
演算順・重み・M2・None条件・入力順/型の変更は旧base/shared oracleとサーバ→ID補完→store/baseline接続を再検証。

## Architecture
learning/loss_statisticsの中立数値部品。既存BoundedLossMomentsを再利用し、新集計型やサーバclassを追加しない。
上位が参加判定して各modelのoverallを順序付きtupleにする。統計のない参加clientは渡さない。
Noneを返した場合、上位は既存サーバrecordをそのまま保持する。更新時は明示ModelAndClassLossStatistics(overall=結果,class=())でwhole置換する。
これらの上位所有操作は今回testで明示接続し、productionには先取りしない。
損失観測件数と学習データ件数を区別し、ここに学習量引数を追加しない。

## File Structure Plan
- src/federated_learning_experiments/learning/loss_statistics/server_loss_mean_aggregation.py: 全入力検査と順序付き件数加重平均。
- tests/refactoring/test_server_loss_mean_aggregation.py: base/shared実旧oracle、拒否/独立/環境、既存保持/wholeclass置換/ID補完/store更新/baseline接続。
- tests/refactoring/test_single_run_dependency_boundaries.py: exact module/型許可と禁止注入。
- 対象spec/roadmap、実測した発見がある場合のみ既存台帳。

## Components and Interfaces
```python
aggregate_participating_client_loss_means(
    *, participating_client_loss_moments: tuple[BoundedLossMoments, ...],
) -> BoundedLossMoments | None
```
入力exact tuple、要素exact BoundedLossMoments。publicconstructorへ全3fieldを渡し、全入力を検査・独立コピーしてから加算する。
入力を再__post_init__しない。件数合計とweighted_mean_sumは0/0.0から入力順の通常加算。
weighted_mean_sum += loss_moments.mean_loss * loss_moments.observed_loss_count、
total_observed_loss_count += loss_moments.observed_loss_count。
n==0入力も同じ演算で通す。total==0はNone。正ならPythonfloat除算し、BoundedLossMoments(n,mean,M2=0.0)のpublicconstructorで結果を検査して返す。
math.fsum/NumPy/pooling/分散加算/逐次Welfordへ置換せず、旧順を保持する。
算術OverflowErrorやconstructorのTypeError/ValueErrorに項目名を添え、範囲外/非有限結果をclipしない。

## Requirements Traceability
| 条件 | 実装/検証 |
|---|---|
| 1.1 | 登録順のmean*n +=/n +=/除算、異なるndata gate、順序/丸めoracle |
| 1.2 | 正n/合計n/mean/M2=0、n1M2非零入力も出力0、classなし |
| 1.3 | 空/全zero Noneと既存serverrecord保持 |
| 2.1 | exact型/後段forgedfield/算術・結果表現不能拒否/入力不変 |
| 2.2 | 既存frozen値の独立コピー/別呼出非共有 |
| 2.3 | RNG/grad/default dtype/device保持 |
| 3.1 | モデルを生成しないstubからBaseServer/SharedBackboneServer両updateを直接実行 |
| 3.2 | oldgate選別→pure→wholeglobal置換またはNone保持→ID補完→新store→旧帰属更新/baseline |
| 3.3 | exactAST/全golden/stdlibfreshsmoke/roadmap/発見・承認記録 |

## Error Handling
型/値違反はTypeError/ValueError、入力項目名と理由を示す。OverflowErrorはValueErrorへ変換し算出結果の問題と明示する。
入力全検査後に加算し、例外でも入力/外部状態を変更しない。Noneは異常ではなく更新不要を示す。

## Testing Strategy
task1 TDDで空/zero/単一/複数/順序・丸め/不参加・欠落/学習件数の違いを両旧サーバupdateの軽いget_params stubへ直接照合。
task2 test-onlyでexact型/tuple/forged値・後段invalid/入力非変更/算出結果overflow・範囲外、frozen/deepcopy/共有環境を確認。旧参加gateを明示し、None保持とclasswhole置換、サーバID補完→store更新/baselineを検証。
task3 exactAST、stdlib -S smoke、全tests旧11/最終3golden、9条件/配置/台帳有無、Luna最終GO。
