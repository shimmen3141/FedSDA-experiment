# 設計 revision 1

## 配置と責務

runtime/held_model_registration_confirmation.pyへ状態を持たない組立関数を置く。学習層・評価標本・FedSDA送信保留を横断するため、個別の計算層ではなく外側runtimeに配置する。各ownerからruntimeへの逆依存は作らない。設定・モデル生成・採番・通知・数値計算を関数へ追加しない。

## APIと順序

`confirm_held_model_registration(*, registered_global_model_id:int, held_model_training_state_registry:HeldModelTrainingStateRegistry, loss_statistics_store:ModelAndClassLossStatisticsStore, training_sample_store:ModelTrainingSampleStore, evaluation_sample_store:ModelEvaluationSampleStore, model_training_and_assignment_counts_store:ModelTrainingAndAssignmentCountsStore, current_training_model_assignment:CurrentTrainingModelAssignment, pending_model_upload_state:PendingModelUploadState)->TrainingModelAssignmentChange|None`

1. 正式IDをexact intかつ非負と検証する。全7ownerをexact具体型で検証する。ID型はTypeError、負値はValueError、owner型はTypeError。
2. 現在IDが非負ならpending解除のみ、Noneを返す。
3. 負の場合、registry.get_held_model_training_stateで保有済みを確認。欠落ならKeyError、更新開始前に終了。
4. registry.reassign → statistics.reassign → training samples.reassign → evaluation samples.reassign → counts.transfer の順に既存APIを呼ぶ。
5. current.assignで返る変更recordを保持し、pending.clearの後に返す。負元/非負先なので必ず実変更となる。

入力拒否とモデル欠落は変更開始前の不変を保証する。正常なowner APIの順次更新であり、private改変・並行更新・MemoryError・monkeypatchによる任意の途中例外へのrollbackは提供しない。rollback用コピーを作ってモデル参照/optimizerを壊さない。ownerは関数が保持せず、current/pending整合は呼出し側で保証する。上位が欠落モデルを既存initializerで再構築・optimizerを用意してregistryへ登録してから呼ぶ経路は後続specで組み立てる。

## 依存とファイル計画

許可importは上記7owner＋TrainingModelAssignmentChangeの8symbolのみ（__future__.annotations許可）。exact AST guardをgeneric runtime許可より前に適用。module丸ごと/再export/private/torch/NumPy/乱数/設定/旧実装/候補/学習実行は禁止。

|ファイル|役割|
|---|---|
|runtime/held_model_registration_confirmation.py|7ownerの登録確認組立のみ|
|tests/refactoring/test_held_model_registration_confirmation.py|実旧confirm対照・拒否/順序・後続学習接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|exact依存許可/禁止注入|
|本spec正本・steering resume/roadmap|承認/証拠/再開|

## 検証

- Task1: RED module未実装→GREEN。旧実confirmで保有済み一時model/統計/標本/計数を構成し、先既存/新先・補助owner元欠落・順序・非負current分岐・正ID拒否/7owner拒否/負model欠落と全state不変を対照。保持record/parameter/grad/optimizer/payload identityも観測する。
- Task2: AST注入RED→exact guard GREEN。12条件（class2/4×Adam標準/AMSGrad/SGD×共有更新有無）3共同更新の初回後に新confirmと実旧confirmを呼び、registryの現在bindingで後続学習しloss/全値/grad/optimizer/count/RNGを照合。fresh新CPU起動でconfirm→後続学習、旧importなし。
- Task3: 固定環境全pytest旧11/最終3golden、Ruff全適用対象/Pyright/pip/diff、固定旧差分空。LF承認hash/source hash/JUnitを記録。全tasks独立承認後に別feature最終GO。初期登録/new client/runはこのgateに含めない。
