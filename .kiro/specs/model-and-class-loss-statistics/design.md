# 設計: モデル全体・クラス別損失統計
## Boundary Commitments
### This Spec Owns
一所有者内のmodel ID→全体/class集計、明示初期seed/一括set、帰属損失の原子的更新、独立immutable参照。
### Out of Boundary
model実体、class_count検査、batch seed/統計merge/ID変更/削除/全reset、baseline方針、学習/監視自動更新/候補進行/サーバ同期。
### Allowed Dependencies
stdlibとexact module federated_learning_experiments.learning.loss_statistics.bounded_loss_moments、およびBoundedLossMoments/accumulate_bounded_loss_observationだけ。
### Revalidation Triggers
seed/set意味・ID制約・tuple順・更新順/atomic性変更時はbaseline/monitor/参照接続を再検証。merge/ID変更は別spec。

## Architecture
旧BaseClient共通の統計所有なのでlearning/loss_statisticsへ中立部品を置く。上位がmodel/classの帰属・seed生成を決める。内部辞書の値はimmutable snapshot、public入出力で全momentsをコピー検査して可変/改変参照を共有しない。数値は既存public APIへ委譲し、registry/継承/Protocolは新設しない。

## File Structure Plan
- 新src/federated_learning_experiments/learning/loss_statistics/model_and_class_loss_statistics.py: frozen一モデル集計とstore、検査/コピー。
- 新tests/refactoring/test_model_and_class_loss_statistics.py: 旧全体/class oracle、seed/upsert/参照/拒否/副作用/上位接続。
- 変更tests/refactoring/test_single_run_dependency_boundaries.py: exact上流module+2symbol許可と禁止注入。
- roadmap/spec証拠、LEGACY-006移植結果の追記。

## Components and Interfaces
ModelAndClassLossStatistics(*, overall_loss_moments:BoundedLossMoments,class_loss_moments_by_class_id:tuple[tuple[int,BoundedLossMoments],...]=()): frozen kw_only。全fieldを検査/独立コピーしてからobject.__setattr__。class tupleはexacttupleの2要素tuple、非負exact ID/重複禁止。各momentsのみ検査、全体/class数の対応条件なし。
ModelAndClassLossStatisticsStore(*,initial_loss_statistics_by_model_id:dict[int,ModelAndClassLossStatistics]|None=None): Noneなら空、exactdictの全ID/値を先にコピー検査して保持。model順を維持。
set_model_loss_statistics(*,model_id:int,loss_statistics:ModelAndClassLossStatistics)->None: ID/全値検査後に一モデルの全体/classを置換。新IDは末尾、既存ID位置保持。
record_assigned_loss(*,model_id:int,observed_loss:float,observed_class_id:int|None=None)->None: ID/class事前検査。既存状態から全体更新、指定classだけ更新（欠落はemptyMoments）、新snapshot全検査後dictへ一回commit。失敗でemptyentryも残さない。
get_model_loss_statistics(*,model_id:int)->ModelAndClassLossStatistics|None: ID検査、欠落None、独立コピー。
get_state_snapshot()->tuple[tuple[int,ModelAndClassLossStatistics],...]: dict順・class順を保持し全値独立コピー。mutable dictや内部値を返さない。
_validate_identifier(*,identifier,parameter_name,minimum_value=None)->None: exactint/bool除外、classのみminimum0、モデルsigned無制限。
_copy_loss_moments(*,loss_moments)->BoundedLossMoments: exact型とpublicconstructorでfield再検査・独立コピー。input__post_init__再呼出/privateimportなし。
_copy_model_and_class_loss_statistics(*,loss_statistics)->ModelAndClassLossStatistics: exact型とpublicconstructorで全fieldコピー検査。
内部_model_loss_statistics_by_model_id:dict。状態更新はconstructor/set/recordだけ。

## Requirements Traceability
| 条件 | 実装/検証 |
|---|---|
| 1.1,1.2,1.3 | frozenモデル集計、store初期/set/get/snapshot、seed/置換/順序/欠落と0件/独立copy |
| 2.1,2.2,2.3 | publicWelford帰属更新、既存/未登録/クラスなしと別系列不変、旧oracle各観測 |
| 3.1,3.2,3.3 | 全検査→一回commit、invalidmodel/class/loss/seedとclass後段overflow、参照独立/frozen/RNG/defaults |
| 4.1,4.2,4.3 | directoldoracle、全体→baseline→monitor/history参照test、AST/fullgolden/smoke/台帳/roadmap |

## Error Handling
TypeError/ValueErrorは項目名と理由。constructor/setの全要素・recordの両更新結果を検査してから状態変更する。
内部属性を直接改変する操作はpublic APIではないが、public seed/値型fieldの改変を再検査し、返却snapshotの改変をstoreへ波及させない。

## Testing Strategy
task1: TDDで値型/storeの初期/set/帰属/参照を実装。正負ID/class0/1/None、複数model/クラス、非零n1M2/class欠落/サーバM2zero、旧_update_model_stats各更新完全一致、全体のみの件数差、upsertclass置換/order/missing。
task2: 拒否atomic（不正class旧部分更新対照、全体/class countnextoverflow、seed全要素）、forged拒否/inputdict/getsnapshot独立、frozen/RNG/dtype/device/keyword、全体momentsをbaseline選択からmonitor/参照選択へ明示接続。
task3: AST exactmodule/2symbol、private/別symbol/別module/methods/torch/NumPy/旧を拒否。全tests旧11/最終3golden、stdlib smoke、12条件/roadmapとLEGACY006追記・最終GO。既存venv/pytest基盤を利用。
