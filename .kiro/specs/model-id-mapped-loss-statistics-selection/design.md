# 設計: モデルID対応後の損失統計選択
## Boundary Commitments
### This Spec Owns
全体・クラス別統計の順序付きsnapshot、モデルID対応、サーバsnapshotを受け、ID対応と候補選択だけを行う純関数。
### Out of Boundary
モデル実体・重み・学習データ・現在の帰属ID・登録/送信、ID対応生成、store更新/削除、サーバ集計、監視・警報後進行。
### Allowed Dependencies
stdlib、exact module federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics と、その公開型 ModelAndClassLossStatistics のみ。
store/同moduleの別symbol/private helper/別module/methods/runtime/旧/torch/NumPyは参照しない。
### Revalidation Triggers
対応回数、最大件数/tie、server補完条件、モデル/クラス順・入力形式変更は旧oracleとstore/update/baseline接続を再検証する。

## Architecture
中立learning/loss_statisticsで選択のみを行う。storeのget_state_snapshot出力を直接受け、同じtuple形式を返す。
新しいsnapshot型や汎用merge枠組みを追加しない。既存公開constructorで全統計を再検査・深い数値コピーする。
元IDの残留を避けるため、呼出側が返却tupleをdictにして新storeを構築する。既存storeへupsertを連続適用する方法は全置換ではない。
今回は接続をtestで明示し、productionへ上位調整を先取りしない。

## File Structure Plan
- src/federated_learning_experiments/learning/loss_statistics/model_id_mapped_loss_statistics_selection.py: snapshot/対応表の検査・コピー、ID対応と候補選択。
- tests/refactoring/test_model_id_mapped_loss_statistics_selection.py: 旧mapping oracle、拒否/独立/共有環境、store/update/baseline接続。
- tests/refactoring/test_single_run_dependency_boundaries.py: exact module/型許可と禁止注入。
- 対象spec文書と.kiro/steering/roadmap.md: 承認・検証・完成範囲。実測した新発見のみ既存台帳へ記録。

## Components and Interfaces
```python
select_loss_statistics_after_model_id_mapping(
    *,
    local_model_loss_statistics: tuple[tuple[int, ModelAndClassLossStatistics], ...],
    model_id_mapping: dict[int, int],
    server_model_loss_statistics: tuple[tuple[int, ModelAndClassLossStatistics], ...] | None = None,
) -> tuple[tuple[int, ModelAndClassLossStatistics], ...]
```
両snapshotはexact tuple、各pairもexact tupleで長さ2、IDはsigned exact builtin int、同snapshot内のID重複を拒否。
recordはexact ModelAndClassLossStatistics。public constructorに両fieldを渡して入れ子のmoment/classも検査・コピー。壊されたfrozenrecordも再検査するが入力の__post_init__を呼び直さない。
対応表はexact dict、全key/valueはsigned exactbuiltinint。未使用の項目も検査し、範囲・連鎖・循環による追加制約は設けない。
省略/Noneサーバは空tupleへ正規化する。ローカルNoneは許さない。全snapshotと対応表を選択前に検査する。
検査済みローカルを入力順に走査し、対応を一回だけ適用。出力dictが未登録または新候補のoverall observed_loss_countが厳密に大きい場合だけ全recordを置換。同nは置換しない。
検査済みserverを入力順に走査し、出力未登録または既存overall件数0のとき全recordで置換。serverIDは再対応しない。
dictの既存位置保持を用い、tuple(items())を返す。算術を行わず数値を保つ。
永続状態は持たない。入力/別結果/入れ子momentとの共有を作らない。

## Requirements Traceability
| 条件 | 実装/検証 |
|---|---|
| 1.1 | 一回対応、欠落/負ID/identity/chain/cycle/unused、旧oracle |
| 1.2 | overall最大n/firsttie・wholeclass保持、旧oracle |
| 1.3 | 初出target順・勝者置換位置、旧oracle |
| 2.1 | server既存IDを再対応せず欠落/zeroだけ補完 |
| 2.2 | 正local優先、server大件数対照 |
| 2.3 | zero server/位置維持/末尾順、旧oracle |
| 3.1 | 全入力type/ID/重複/field検査、無変更拒否 |
| 3.2 | 深い独立コピー/frozen/共有環境保持 |
| 3.3 | モデル実体なし旧oracle、結果→新store→帰属追加→baseline |
| 3.4 | exact AST/stdlib fresh smoke/全golden/正本 |

## Error Handling
TypeError/ValueErrorで入力項目名と理由を示す。constructorの検査失敗にもlocal/server項目名を添える。
入力を変更しない。正常なemptyは空tuple。moment/classの値は既存型契約のままで、全体とクラス合計の一致やclass上限は要求しない。

## Testing Strategy
task1: TDDで旧BaseClient.apply_server_mappingを空モデル/空学習データのstubへ直接呼び、通常/欠落/identity/負ID/chain/cycle/unused/collision/tie反転/サーバ補完/順序と全fieldを完全一致照合。
task2: 後段不正入力・bool/型/重複・壊されたrecord/入れ子・keyword/入力無変更/結果frozenと独立/共有環境保持、get_state_snapshot→選択→新store→record→旧更新/用途別baselineを検証。
task3: ASTのexact型許可とstore/private/他module/旧/torch/NumPy/上位禁止注入、stdlibのみ独立smoke、全tests旧11/最終3golden不変、10条件証拠、roadmap、Luna最終GO。
