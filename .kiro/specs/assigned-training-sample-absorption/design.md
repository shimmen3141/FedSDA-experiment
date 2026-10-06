# 設計 revision 1

## 配置と責務

runtime/assigned_training_sample_absorption.pyへ状態を持たない組立関数を置く。学習状態一覧・学習標本・計数（learning/training）、損失統計（learning/loss_statistics）、損失評価（learning/prediction）を横断する。手法固有の判断を含まず、FedSDAの複数経路と後のFedDrift移植から呼ばれる。下位層からruntimeへの依存は作らない。

## APIと順序

`absorb_assigned_training_samples_into_held_model(*, model_id:int, assigned_training_samples:tuple[ObservedTrainingSample, ...], assigned_sample_concept_ids:tuple[int | None, ...], held_model_training_state_registry:HeldModelTrainingStateRegistry, training_sample_store:ModelTrainingSampleStore, model_training_and_assignment_counts_store:ModelTrainingAndAssignmentCountsStore, loss_statistics_store:ModelAndClassLossStatisticsStore)->None`

状態を変更しない段階:

1. model_idがexact intでなければTypeError。4ownerをexact具体型と検証（TypeError）。標本列がexact tupleでない/要素がexact ObservedTrainingSampleでないならTypeError。概念ID列がexact tupleでないならTypeError、長さが標本列と異なればValueError、要素がNoneでもexact intでもなければTypeError。
2. `classifier = registry.get_held_model_training_state(model_id=model_id).classifier`。未保有はKeyError（空列でも）。
3. 各標本について`evaluate_classifier_per_sample_bounded_losses(classifier=..., input_features=..., observed_class_labels=...)`を1回ずつ呼ぶ（特徴・ラベルの検証と1forward）。戻り値の要素数が1でなければValueError。損失を`losses[0].item()`のfloat、観測クラスを`int(observed_class_labels.reshape(-1)[0].item())`として保持する。損失評価の検証により、ラベルは有限な整数値で0以上class_count未満である。

状態を変更する段階（標本順に、標本ごとに次の順）:

4. `training_sample_store.append_model_training_samples(model_id=..., training_samples=(標本,))`。
5. `counts_store.record_assigned_sample_concept(model_id=..., observed_concept_id=概念IDまたはNone)`。
6. `loss_statistics_store.record_assigned_loss(model_id=..., observed_loss=損失, observed_class_id=観測クラス)`。

旧は標本ごとに追加→概念→損失評価→統計の順。吸収中にモデルは変わらないので、3を先に行っても各標本の損失は同じ値になる。4〜6の標本ごとの順は旧と同じ。4〜6の入力は1〜3で検証・生成済みで、既存APIの拒否条件（model_idの型、tupleと要素型、概念IDの型、損失とクラスの型・範囲）を満たす。record_assigned_lossが受ける損失は損失評価が返す[0,1]の有限値、クラスは非負int。private改変・並行更新・MemoryError等の途中例外へのrollbackは提供しない。

空列では2だけを行い、4〜6を呼ばない。標本1件ごとに1-tupleで追加するため、空列でモデルの標本列を作らない。

戻り値はNone。

## 依存とファイル計画

許可importは次の6symbolと`__future__.annotations`（数に含めない）だけ: (1)HeldModelTrainingStateRegistry、(2)ModelTrainingSampleStore、(3)ObservedTrainingSample、(4)ModelTrainingAndAssignmentCountsStore、(5)ModelAndClassLossStatisticsStore、(6)evaluate_classifier_per_sample_bounded_losses。torchはimportしない（Tensorのメソッド呼出しだけを使う）。exact AST guardをgeneric runtime許可より前に適用。module丸ごと/再export/private/torch/NumPy/乱数/設定/旧実装/現在ID owner/採番/登録・採用・登録確認/学習実行/評価標本storeは禁止。

|ファイル|役割|
|---|---|
|runtime/assigned_training_sample_absorption.py|帰属確定標本の吸収の組立のみ|
|tests/refactoring/test_assigned_training_sample_absorption.py|実旧吸収対照・拒否/不変・順序・確定処理の非採用分岐と後続学習への接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|exact依存許可/禁止注入|
|本spec正本・steering resume/roadmap・implementation-findings|承認/証拠/再開/旧挙動の記録|

## 検証

- oracle: 上流の採用oracle（build_local_adoption_oracle）が作る実旧`SharedBackboneClassConditionalESRFedSDAClient`の実_absorb_into_storeを、同じ初期値の新owner群と同じ標本列で呼ぶ。旧標本は(特徴, ラベル, 概念)または(特徴, ラベル)のtuple。式や手順をtestへ複製しない。
- Task1: RED module未実装→GREEN。class2/4、吸収先（現在のモデル/別の保有モデル/統計未登録のモデル）、標本0/1/複数件、概念IDあり/None混在/概念要素なしで、標本列と順序・Tensor identity、割当概念計数、統計全field（クラス初出順を含む）を実旧と照合。他モデル・学習計数・parameter/grad/optimizer・乱数の不変。2.1〜2.4の拒否（途中の標本が不正な場合を含む）で全状態不変、呼出順（全損失評価→標本ごとに追加→概念→統計）。旧の途中失敗時の部分更新を実旧で再現し、implementation-findingsへ記録。AST注入RED→6symbol exact guard GREEN。品質/型/独立Luna/主担当gate。
- Task2: 実旧_finalize_forward_validationの棄却分岐（候補損失が参照より大きい条件。現在のモデルへ保留標本を吸収）を実行し、新の吸収と標本・計数・統計・現在ID不変を照合する。12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）の実NNで、共同更新→吸収→各自の標本storeを使う共同更新を実旧と照合。fresh新CPUで旧importなしの吸収→学習。独立Luna/主担当gate。
- Task3: 固定環境全pytest（旧11/最終3golden、主担当実測＋JUnit、基準はsteering/agent-handoff.md）、Ruff/Pyright/pip/diff、固定旧差分空、承認/source hash。独立Luna/主担当gate後、別feature最終GO。
