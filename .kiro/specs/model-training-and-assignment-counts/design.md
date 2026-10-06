# 設計: モデル別学習・割当件数

revision: 1

## 所有と配置

`learning/training/model_training_and_assignment_counts.py`に一storeとsnapshot宣言を置く。モデルID移管でまとめて変更する診断量であり、学習/帰属の決定とは分離する。storeの3 dictは独立の項目有無/順序を持つ。conceptの真値は予測や割当を決定する入力ではない。
snapshotはfrozen kw-only recordだが中のdictは呼出し側が変更できる独立copyである。ownerには戻さない。getterもcopy、欠落読み取りにsetdefaultを使わない。
依存はdataclasses.dataclassだけ（__future__.annotationsは許可）。exact AST symbol guardをgeneric stdlib許可より前に置く。Tensor/NumPy/Counter/Random/学習実行/設定/他ownerを依存にしない。

## API

- `ModelTrainingAndAssignmentCountsSnapshot`のfieldは `trained_sample_counts_by_model_id:dict[int,int]`、`parameter_update_step_counts_by_model_id:dict[int,int]`、`assigned_sample_counts_by_model_and_concept_id:dict[int,dict[int,int]]`。
- `ModelTrainingAndAssignmentCountsStore()`は同じ意味のprivate3辞書を空で生成。
- `record_completed_model_training(*,model_id:int,trained_sample_count:int,parameter_update_step_count:int)->None`。全検査→両増分加算、0でも項目作成。共有optimizer step数ではなく各モデルの個別更新回数。
- `record_assigned_sample_concept(*,model_id:int,observed_concept_id:int|None)->None`。全検査→None戻る、既知conceptへ1加算。
- `get_model_assigned_sample_concept_counts(*,model_id:int)->dict[int,int]`。ID検査→copy。
- `snapshot_model_training_and_assignment_counts()->ModelTrainingAndAssignmentCountsSnapshot`。3辞書と概念内辞書をcopy。
- `transfer_model_training_and_assignment_counts(*,original_model_id:int,receiving_model_id:int)->None`。両ID検査・同IDValueErrorを先に実施。各元の存在に応じpop→先加算。概念は先setdefault、元順get加算。存在しない元は先を生成しない。
- `remap_model_training_and_assignment_counts(*,model_id_mapping:dict[int,int])->None`。全mapping検査、3一時dictへ各元順の一回get加算、概念は先初出/元concept順。完成後に3dictを交換。
- `_validate_integer(*,parameter_value:int,parameter_name:str,minimum_value:int|None=None)->None`。exactでない型はTypeError、下限違反はValueError、項目名と理由を表示。IDには下限なし、件数は0。

負ID/負conceptは許す。非正常なMemoryError/private改変/並行更新のrollbackは通常契約外。元先の同一IDは明示拒否しLEGACY011へ対照を残す。

## ファイル計画

|パス|役割|
|---|---|
|learning/training/model_training_and_assignment_counts.py|3計数owner/独立snapshot|
|tests/refactoring/test_model_training_and_assignment_counts.py|実旧計数/移管/再編/拒否/学習接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|exact AST guard RED/GREEN|
|docs/research/implementation-findings/legacy-011-*とREADME|実旧不正通知の修正候補|
|対象spec・steering resume/roadmap|正本・実測・現在地|

## 検証trace

- 1.1–1.3: 旧_attribute_model_trainingで0/反復/任意大int増分、_record/get_model_concept_countsのNone/既知/負concept対照。getter欠落/取得結果変更/後続更新の分離。独立辞書のkey順をlist比較。
- 2.1–2.2: 実confirm（負元→異先）、apply_server_mappingの先既存/新先/元欠落・計数部分存在、concept共通/新concept順、0項目/空概念辞書、連鎖/循環/多元先。新APIのsigned異IDは直接期待値でも確認。
- 2.3: 実旧同一負ID破損の再現と、新同ID拒否/全snapshot不変を同入力で比較。
- 3.1: 全引数・末尾mapping不正、None/元欠落でも型検査、不正型bool/int派生/NumPy int・負増分、操作前後のsnapshot不変。
- 3.2–3.3: exact AST RED/GREEN、fresh stdlib単独起動/torch非import、新CPU学習接続のfresh起動。class2/4×Adam標準/AMSGrad/SGD×共有更新有無の12条件各3学習、初回後に旧confirmと新count transfer、上位がbinding/標本IDを対応。各stepの全計数/損失/NN値/grad/optimizer/RNGと3共有RNG/defaultsを照合する。
- 全pytest旧11/最終3golden、Ruff/format/Pyright/pip、固定旧差分空、hash/JUnit/実測。全taskの独立承認後に別feature最終GO。
