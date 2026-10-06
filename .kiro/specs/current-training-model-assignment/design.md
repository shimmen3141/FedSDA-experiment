# 設計 revision 1

## 所有と依存

learning/training/current_training_model_assignment.pyに単一ID ownerとfrozen kw-only変更recordを置く。FedDrift/FedSDAで共有できる学習帰属状態であり、混合予測の重みownerとは区別する。初期値はruntimeから明示する。モデルの存在確認と登録済み/一時IDの判定をここへ入れない。

依存はdataclasses.dataclassのみ（__future__.annotationsも許可）。他owner・Tensor・設定・乱数・診断への依存はexact AST symbol guardで拒否する。上位が変更recordを使って理由別のhookや位置記録を行う。ownerはcallbackを保持しない。

## API

- CurrentTrainingModelAssignment(*, initial_model_id:int): 全検証後に_current_training_model_idを保持。
- current_training_model_id:int: setterのないproperty。
- assign_model_for_training(*, model_id:int)->TrainingModelAssignmentChange|None: exact int検証。同値ならNone。異なる場合、record構築後にID交換。上位が選択したIDを保持するだけで選択計算をしない。
- remap_current_training_model_id(*, model_id_mapping:dict[int,int])->TrainingModelAssignmentChange|None: dict本体と全key/valueを検証後、get(current,current)をassignへ渡す。一段適用。dict保存なし。
- TrainingModelAssignmentChange(previous_model_id:int,current_model_id:int): frozen kw-only。ownerが返す記録は検証済みint。recordの手動生成時の型検証は提供しない。
- _validate_model_id(*,model_id:int)->None: builtin int以外でTypeError。signed/任意大int可。

同値設定はno-opであり計数の加算移管とは異なる。例外時の不変は通常の入力拒否について保証し、private直接変更/MemoryError/並行更新は対象外。

## ファイル計画と検証

|ファイル|責務|
|---|---|
|learning/training/current_training_model_assignment.py|単一ID状態と変更record|
|tests/refactoring/test_current_training_model_assignment.py|実旧切替/通知/一段map、拒否/record、登録確認と既存計数storeの接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|exact依存の許可/禁止注入|
|本spec各正本、steering resume/roadmap|承認・実測・再開情報|

Task1でRED→GREEN、Task2で依存注入RED→GREENと上位接続を検証する。旧BaseClientの実methodを既存test fixtureへ適用し、ID/通知を比較。ローカル切替は実FedSDAClient methodをSimpleNamespaceへ適用し実hook引数を比較。サーバmapは実BaseClientの通知位置・理由とrecordの前後IDを対応付ける。登録確認は負IDから異なる正式IDへ実旧confirmを実行し、新count移管＋current変更をtest-onlyで組み立てる。正式登録transactionの実装完了は主張しない。

全suiteは固定CPU環境、旧11/最終3goldenを含む。fresh python -SでID状態のみ起動しtorch/NumPy/旧importなしを検証。承認文書LF hash・source hash・JUnitと品質検査を記録し、独立Lunaのfeature最終GO後に完了する。
