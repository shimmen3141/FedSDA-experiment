# Requirements Document

## Project Description (Input)

ablationの構成の2つ目。提案構成から、「警報の後の標本での、候補モデルの検証」を外した構成を、新実装で実行できるようにする。

旧実装での指定（`tools/run_main_ablation_suite.sh`の`immediate-creation`）: 提案構成と同じ構成名で、`--new-model-creation-policy immediate`。旧のclient（`clients/fedsda.py`の`_resolve_drift`）は、この値のとき、警報の区間に合う保有モデルが無ければ、区間で学習した新しいモデルを、その場で登録し、学習帰属を切り替える（`action`は`create`）。候補の検証（`create_pending`、判定の記録、棄却）は、起きない。

提案構成では、区間で学習した候補を保持し、警報の後の標本で、警報の時点の保有モデルと比べてから、採否を決める。

## Requirements

### Requirement 1: 候補検証をしない構成

#### Acceptance Criteria

1. Where 候補モデルの採否の方針が「検証せずに採用する」のとき, when 警報の区間に合う保有モデルが無い, the FedSDA Run Client shall 区間で学習した候補を、警報の位置で、新しいモデルとして採用し、学習帰属を切り替える。候補検証は、始めない。
2. The 採用 shall 候補検証での採用と同じ手順で行う（一時IDの採番、初期の損失統計、送信の保留、学習の計数、区間の標本の追加、学習帰属の切替え）。
3. The 適応の記録 shall この採用を、候補検証の結果とは別の結果種別で持つ。学習帰属の切替位置と、適応の粗い区分（候補の採用）には、数える。
4. The 構成 shall 掃引のコマンドで、`fedsda-without-candidate-validation`という名前で選べる。この構成のrun設定は、提案構成と、候補の採否の方針だけが違う。
5. The 構成の全体run shall 小さい条件（sine2）で、実旧の、同じ指定の全体runと、指標と離散列が一致する。
6. The 照合 shall この構成に固有の経路を通る: 警報の位置での採用が1回以上あり、候補検証の開始・検証中の警報・検証の確定・判定の記録が、新旧とも無い。結果が、提案構成と違う条件がある。
7. The 提案構成 shall 結果が変わらない。
