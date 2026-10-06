# 設計 revision 2

## 配置と責務

runtime/post_alarm_candidate_validation_sample_observation.pyへ状態を持たない組立関数を置く。損失評価（learning/prediction）とFedSDAの損失収集（methods/fedsda）を横断する。確定と同じ外側runtimeに配置する。下位層からruntimeへの依存は作らない。

## APIと順序

`observe_post_alarm_candidate_validation_sample(*, sample_index:int, input_features:Tensor, observed_class_labels:Tensor, candidate_classifier:ResidualAdapterClassifier, reference_classifiers_by_model_id:dict[int, ResidualAdapterClassifier], post_alarm_candidate_loss_collection:PostAlarmCandidateLossCollection)->bool`

1. 損失収集がexact PostAlarmCandidateLossCollectionでなければTypeError。参照分類器の対応がexact dictでなければTypeError。特徴がTensorで、2次元かつ行数1でなければ、どの分類器のforwardよりも前にValueError（1標本契約）。特徴がTensorでない場合は2の損失評価がforward前にTypeErrorで拒否する。ラベルの形状[1,1]は損失評価が特徴の行数との一致として検査する。
2. 候補について`evaluate_classifier_per_sample_bounded_losses(classifier=candidate_classifier, input_features=..., observed_class_labels=...)`を呼ぶ（分類器・特徴・ラベル・出力を検証し、no_gradで1forward）。1で行数1を確認済みなので戻り値は1要素で、損失を`losses[0].item()`のfloatにする。
3. 対応の各(model_id, 参照分類器)について、渡された順に同じ評価を行い、`{model_id: float}`を作る。
4. `collection.observe_losses_after_label_observation(sample_index=..., candidate_loss=..., reference_losses_by_model_id=...)`を1回呼ぶ。収集は、到達済み（RuntimeError）、標本位置の型と連続性、損失の型と範囲、参照IDの型と集合の一致を全て検査してから全系列へ追加する（完了spec）。拒否時は収集も未変更。
5. `collection.ready_for_acceptance_evaluation`を返す。

2〜3が状態を変えない根拠: 既存損失評価はno_gradの1forwardだけを行い、分類器のparameter・grad・学習modeを変更しない（classifier-per-sample-bounded-loss-evaluationの契約と検証）。新モデル（共有特徴抽出部・非線形残差アダプタ・分類層）はdropout・batch正規化・bufferを持たず、forwardは乱数を消費せず内部状態を更新しない（residual-adapter-model-architectureとadopted-candidate-initial-local-registrationで確認済み）。本specのtestでも、観測の前後で全分類器のparameter/grad/学習modeと乱数状態が不変であることを確認する。この前提のもとで、候補の評価後に参照の評価が失敗しても、実行済みのforwardが残す状態はなく、4が拒否した場合を含め失敗時に変わる状態はない。評価順（候補→対応の順）が結果へ影響しないことも同じ前提による。前提が崩れるモデル構造（forwardで状態が変わる層）を追加する場合は、本specの順序と不変の主張を見直す。sample_indexとmodel_idの検査は収集へ委ね、本関数で重複させない。

旧はfloat(per_sample_error(x,y).mean().item())。1標本では平均が1要素の値と同じfloat32値で、item()の結果は一致する。

## 依存とファイル計画

許可importは次の4symbolと`__future__.annotations`（数に含めない）だけ: (1)torch.Tensor（注釈用）、(2)ResidualAdapterClassifier、(3)evaluate_classifier_per_sample_bounded_losses、(4)PostAlarmCandidateLossCollection。exact AST guardをgeneric runtime許可より前に適用。採否評価・確定・採用・吸収・設定・owner群・Tensor以外のtorch・乱数・旧実装・private・module import・再exportは禁止。

|ファイル|役割|
|---|---|
|runtime/post_alarm_candidate_validation_sample_observation.py|検証標本1件の損失評価と収集への追加のみ|
|tests/refactoring/test_post_alarm_candidate_validation_sample_observation.py|実旧観測処理の対照・拒否/不変・順序・評価と確定への接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|exact依存許可/禁止注入|
|本spec正本・steering resume/roadmap|承認/証拠/再開|

## 検証

- oracle: 上流の確定oracle（build_local_adoption_oracle）の実旧clientと実ForwardValidationSessionを使う。sessionの損失列を空にし、参照モデルを実旧_snapshot_reference_models（実旧_new_modelで独立モデルを作り警報時点の値を複製）で作る。新側は同じ値の独立した参照分類器を作る。検証標本を1件ずつ実旧_observe_forward_validationと新関数へ渡す。規定件数目で実旧は確定処理まで進むので、新側は到達を見て既存の評価関数と確定をtest-only接続する。手順をtestへ再構成しない。実行可能性はREDより前に確認する。
- Task1: testを書いたら実装より前にREDを実行して記録する。GREEN後、実装をstubと誤実装（参照損失の取り違え、候補損失を参照に使う、収集へ追加しない、到達を返さない、複数行を平均で受理、評価前に収集へ追加する等）へ一時的に差し替えてtestの失敗を確認し、元へ戻してhashを照合する。class2/4×規定件数2/4で、各観測回後の候補損失列・参照損失列（値・順序・ID順）と到達の戻り値を実旧sessionと照合。分類器のparameter/grad/学習mode・乱数の不変。2.1〜2.3の拒否で収集snapshotの不変、呼出順（全評価→収集1回）。AST注入RED→4symbol exact guard GREEN。品質/型/独立レビュー/主担当gate。
- Task2: 実旧の観測処理を規定件数まで実行したときの確定後の状態と、新の観測→評価→確定の状態を照合する（履歴平均の与え方で旧の分岐を変える: なし/現行モデル/別モデル。class2/4）。12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）で、共同更新→観測と確定→各自の標本storeを使う共同更新を実旧と照合。fresh新CPUで旧importなしの収集開始→観測→評価→確定→学習。独立レビュー/主担当gate。
- Task3: 固定環境全pytest（旧11/最終3golden、主担当実測＋JUnit、基準はsteering/agent-handoff.md）、Ruff/Pyright/pip/diff、固定旧差分空、承認/source hash。独立レビュー/主担当gate後、別feature最終GO。
