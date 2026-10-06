# 設計 revision 1

## 配置と責務

runtime/post_alarm_reference_model_fixation.pyへ、結果recordと状態を持たない組立関数を置く。学習状態一覧・分類器・parameter snapshot（学習層）と、FedSDAの用途別基準値の選択（methods/fedsda/loss_statistics）を横断する。観測・確定と同じ外側runtimeに配置する。下位層からruntimeへの依存は作らない。

## 型

`FixedPostAlarmReferenceModels`（frozen, kw_only dataclass）:
- `reference_classifiers_by_model_id: dict[int, ResidualAdapterClassifier]`（保有一覧の順。観測APIの同名引数へそのまま渡せる）
- `reference_historical_mean_losses_by_model_id: dict[int, float]`（保有一覧の順で、条件を満たすモデルだけ。評価APIの同名引数へそのまま渡せる）

recordはfieldの付替えを禁じるだけで、対応と分類器は呼出し側へ渡す通常のobjectである。

## APIと順序

`fix_reference_models_at_alarm(*, held_model_training_state_registry:HeldModelTrainingStateRegistry, loss_statistics_store:ModelAndClassLossStatisticsStore)->FixedPostAlarmReferenceModels`

乱数を消費しない段階:

1. 一覧と統計storeをexact具体型と検証（TypeError）。
2. `held_model_training_states = registry.snapshot_ordered_held_model_training_states()`。空ならLookupError。
3. 各保有モデルについて`snapshot_classifier_parameters(classifier=...)`で全parameterの独立した複製を得る（device/dtype/有限性を検証）。全モデルぶんを先に揃える。
4. 各保有モデルについて`loss_statistics = store.get_model_loss_statistics(model_id=...)`。Noneでなければ`select_post_alarm_reference_historical_mean_loss(loss_moments=loss_statistics.overall_loss_moments)`を呼び、戻り値がNoneでなければ対応へ入れる。

乱数を消費する段階（一覧の順）:

5. 各保有モデルについて、`ResidualAdapterClassifier(model_architecture_settings=held.model_architecture_settings, input_feature_count=held.feature_extractor.input_feature_count, hidden_layer_widths=held.feature_extractor.hidden_layer_widths, class_count=held.class_count)`を生成し（共有部を渡さないので独立した共有特徴抽出部を持つ）、3の複製を`load_state_dict`で読み込む。保有モデルと同じ構造から作るのでkeyと形状は一致する。
6. recordを返す。

旧はモデルごとに生成→値の複製と読込みを繰り返す。新は値の複製（3）を全モデルぶん先に行うが、複製は乱数を消費せず保有モデルも変えないので、生成の順序と乱数の消費（5）は旧と同じである。3・4で拒否された場合は分類器を1つも生成しておらず、乱数は未消費。5の生成は保有モデルの検証済みの構造値を使うため拒否しない。メモリ不足等の途中例外で一部だけ生成された場合のrollback（乱数の巻戻し）は提供しない。

参照分類器の学習modeは生成直後の既定（学習mode）のままで、旧も同じ。損失評価は学習modeを変えず、モデルにmode依存の層はない。参照には個別optimizer管理器を作らない。

## 依存とファイル計画

許可importは次の6symbolと`__future__.annotations`（数に含めない）だけ: (1)dataclasses.dataclass、(2)ResidualAdapterClassifier、(3)snapshot_classifier_parameters、(4)HeldModelTrainingStateRegistry、(5)ModelAndClassLossStatisticsStore、(6)select_post_alarm_reference_historical_mean_loss。exact AST guardをgeneric runtime許可より前に適用。torch・乱数・設定・損失収集・観測・評価・確定・採用・吸収・optimizer・共有反映・候補初期化・旧実装・private・module import・再exportは禁止。

|ファイル|役割|
|---|---|
|runtime/post_alarm_reference_model_fixation.py|参照分類器の生成と履歴平均の取得、結果record|
|tests/refactoring/test_post_alarm_reference_model_fixation.py|実旧の参照複製・session開始の対照、拒否/乱数不変、観測〜確定への接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|exact依存許可/禁止注入|
|本spec正本・steering resume/roadmap|承認/証拠/再開|

## 検証

- oracle（実行可能性は設計前に確認済み、research.md）: 上流の採用oracleの実旧clientで、実旧_snapshot_reference_models（実旧_new_modelでモデルを生成）を実行する。同じ乱数状態から新関数を実行し、処理後のtorch乱数状態・参照IDの順序・各参照の全parameter値と出力を照合する。履歴平均は、実旧_begin_forward_validationを候補の学習（_train_new_model）だけ無効化して実行し、sessionが持つ対応と照合する。式や手順をtestへ複製しない。
- Task1: testを書いたら実装より前にREDを実行して記録する。GREEN後、実装をstubと誤実装（保有モデル自体を返す、共有部を共有する、値を複製しない、順序の逆転、履歴平均の件数条件や0の扱いの誤り、統計なしモデルを含める等）へ一時的に差し替えてtestの失敗を確認し、元へ戻してhashを照合する。class2/4×保有一覧(4,)/(4,9)/(9,-3,4)で実旧の参照複製と照合。独立性（共有部と全parameterの実体が保有モデル・他の参照と別、保有モデルを更新しても参照が不変）。履歴平均は統計の件数0/1/2/多数・平均0・統計未登録の組合せで実旧のsession開始と照合。保有モデル・統計・Python/NumPy乱数の不変。2.1〜2.3の拒否でtorch乱数が未消費。AST注入RED→6symbol exact guard GREEN。品質/型/独立レビュー/主担当gate。
- Task2: 実旧のsession開始（候補の学習だけ無効化）→観測→確定までと、新の固定→損失収集の開始→観測→評価→確定を照合する（統計の与え方で現行維持/別モデル再利用/それ以外、class2/4）。新の候補は、実旧のsession開始が作る候補と同じ値の分類器をtest側で用意する（候補の生成は後続spec）。12条件（class2/4×Adam標準・AMSGrad・SGD×共有部更新有無）で、共同更新→固定→保有モデルをさらに共同更新（参照が固定されたままであること）→観測と確定を実旧と照合。fresh新CPUで旧importなしの固定→観測→評価→確定。独立レビュー/主担当gate。
- Task3: 固定環境全pytest（旧11/最終3golden、主担当実測＋JUnit、基準はsteering/agent-handoff.md）、Ruff/Pyright/pip/diff、固定旧差分空、承認/source hash。独立レビュー/主担当gate後、別feature最終GO。
