# 設計: 新規モデルの送信保留

revision: 1

## Overview / Boundary Commitments
FedSDAの一保留枠へ生成済みparameter snapshotと対応model IDを借用保持し、正の待機ラウンド数を通知ごとに減算する。入力検証/一枠置換/非消費取得/送信可能判定/残回数/明示解除を所有する。
NN取得・学習・snapshot生成/clone・統計・候補採否・登録確認/ID変更・実通信・ラウンド実行位置/診断・runtime/I/Oは上位。外部による借用dict/Tensor破壊・private改ざん・並行変更は対象外。
通常の保留中同ID統計更新は上位がstoreから現在値を取得する。旧統計辞書の全置換/ID変更/統合時の対応は後続の登録/同期specで定める。

## Architecture / Allowed Dependencies
上位→methods/fedsda/model_registration/pending_model_upload。一保留所有者はPendingModelUploadState。
exact from importだけ: dataclasses.dataclass、torch.Tensor/float32/isfinite/strided。bare import・その他stdlib・learning/models/training/loss_statistics・既存snapshot/initializer・他手法・runtime・旧package・I/Oを拒否する。Tensor型検証に必要なPyTorch公開型/演算のみ依存し、モデル実体を持たない。新ライブラリ/Protocol/serializerなし。

## Components & Interfaces
`PendingModelUpload(*, model_id:int, parameter_snapshot:dict[str,Tensor])`: frozen kw_only dataclass。__post_init__でID/snapshot契約を検査する。recordのfieldは不変だがsnapshot辞書と各Tensorは借用し、深い不変値とは呼ばない。
`_validate_pending_model_upload_inputs(*, model_id:int, parameter_snapshot:dict[str,Tensor])->None`。
IDはexact builtin int（bool/派生型拒否、signed）。snapshotはexact plain dict非空、keyはexact str非空、値はTensor（派生を含む）、CPU float32 strided/notnested/有限、requires_grad=False/grad_fnなし。shape/モデル構造は生成側の責務。型不正TypeError、環境/値不正ValueError、項目名と理由の日本語message。入力補正/コピーなし。

`PendingModelUploadState()`初期 `_pending_model_upload=None`、`_remaining_upload_delay_round_count=0`。
`queue_model_upload(*, model_id:int, parameter_snapshot:dict[str,Tensor], upload_delay_round_count:int)->None`: delayはexact builtin int>=1。全検証/record生成後だけ一保留枠とcounterを置換する。過去のrecordやsnapshotは変更しない。
`get_pending_model_upload()->PendingModelUpload|None`: waiting/readyとも同じrecordを借用返却、消費/コピーなし。
`has_ready_model_upload()->bool`: 保留ありかつ残回数0のみTrue。
`remaining_upload_delay_round_count`読取専用property→int、空なら0。
`advance_upload_readiness_at_round_boundary()->None`: waitingならcounter−1、空/readyならno-op。自動クリア/統計更新なし。
`clear_pending_model_upload()->None`: payload None/counter0。繰り返し可能、ID付替え/登録確認を実行しない。

## File Structure Plan
|パス|変更|責務|
|---|---|---|
|src/federated_learning_experiments/methods/fedsda/model_registration/__init__.py|新規|空のpackage境界、re-exportなし|
|同/pending_model_upload.py|新規|record/入力検証/一枠待機状態|
|tests/refactoring/test_pending_model_upload.py|新規|状態列/拒否/旧対照/統計上位接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|変更|exact依存と注入検証|
|対象spec・steering resume/roadmap|新規/更新|承認・証拠・再開位置|

## Requirements Traceability / Testing Strategy
|要件|証拠|
|---|---|
|1.1,1.2,1.3,2.1,2.2,2.3,3.1|実ClassConditionalESRFedSDAClientの空/queue/境界/取得/置換/clear列、delay1/2/4、負/0/正ID、ready後/空のno-op、record identityとcounter|
|1.4|借用dict/Tensor参照と順序/全値不変、producer生成後のモデル更新がsnapshotへ非波及|
|2.4|ID/待機回数/辞書/key/Tensor環境/有限性/勾配の不正。空/待機/readyの保持と先行検証|
|3.2|class2/4×delay1/2/4の6条件で実旧register→保留→同ID統計Welford更新→境界→取得。新snapshot/損失/初期統計/storeの上位接続で登録時モデル値と現在統計の全fieldをexact照合|
|3.3|exact AST RED/GREEN、fresh新CPU producer→loss→store→pending接続（旧非import）、全pytest旧11/最終3golden、Ruff/format/Pyright/pip/diff/hash/JUnit|

## Risks / Revalidation
借用Tensor/辞書は読み手が値を書き換えない契約。cloneが必要なら上位が別snapshotを作る。統計をfixed初期値として保留しない。新設定は今は機能入力として渡し、run/CLI構成は後続。遅延・即時ready・ID対応・統計寿命を変更したら隣接登録/同期を再検証する。
