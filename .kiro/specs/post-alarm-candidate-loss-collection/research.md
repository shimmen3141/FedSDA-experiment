# 調査・設計判断

## Summary

旧748c3aaのForwardValidationSessionと_begin/_observe/_finalize_forward_validationを直接調査。既存採否評価へ渡すlossだけを所有する部品として抽出する。モデル・held_data・履歴snapshot・episode・終端回収は上位へ残す。

## 旧コードと直接観測

- federated_drift_experiment/provisional_model.py のsessionは開始時reference_models順でlossdictを作り、float化して順次追加する。収集時にfloat32再丸めしない。
- clientの観測は今回警報解決より先。proposal当日を新sessionに含めず、最初はproposal+1。target到達した呼出の中でfinalizeしsessionを破棄する。
- 参照モデルは開始時snapshotの固定実体。liveモデルからIDが消えてもその参照lossを観測し、現存IDを使う再適合評価だけ後で絞る。
- 旧未完了終端は実際のvalidation_countを診断し、NaN平均・held_data回収・session消去。新収集の部分snapshotは保持できるが終端挙動を所有しない。
- 調査agentがunbound observeを直接実行しproposal100、target2で101は未到達、102は即finalize、103はsessionなしno-opを確認。candidate→reference2→reference1順、live models={}でも参照を観測する。
- 旧裸appendは不足参照で部分更新、余剰参照を無視、target超過可能。[共通LEGACY-004記録](../../../docs/research/implementation-findings/legacy-004-partial-loss-collection-on-invalid-input.md)を正本として追跡する。通常clientへの影響は未確認、今回旧productionは変更しない。

## Synthesis・保存前設計ゲート

stdlib list/dict/tupleを採用し、model Protocol/汎用session/新数値依存を省く。snapshotのtuplepairをtest/上位がdictへ明示変換して既存採否へ接続する。主担当は15条件traceabilityと所有/対象外/依存/再検証境界、具体ファイル、全検査後更新、3単位の実行可能性を保存前draftで確認した。既存処理の抽出で外部API変更はないためWeb調査不要。kiro-spec-requirements/design/tasks/impl/validate-implとfable-methodを利用する。

