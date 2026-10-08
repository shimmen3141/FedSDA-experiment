# UNPORTED-001: 検出episodeの制御を新実装へ移植していない

- 記録日: 2026-10-09。対象: 旧`federated_drift_experiment/detection_episode.py::DetectionEpisodeController`と、`clients/fedsda.py`の`observe_detection`・`mark_operation`の呼出し（493〜507行、314・325・336行）、`_resolve_episode_duplicate`（718〜735行）。固定基準748c3aa。
- 判断したspec: alarm-occurrence-handling（基点commit `d764474`）。種別: 移植範囲の判断。状態: 当面移植しない（主担当Claude Codeの判断。ユーザーは2026-10-09に、後から追えるよう記録することを指示した。移植の要否そのものをユーザーが決めたわけではない）。

## 旧の機能

近接した複数の検出を、最初の検出からFIFO長（`N_FIFO`）の区間の1つの「episode」として扱い、1つのepisodeでモデル操作（他モデルの再利用、新規モデルの採用）を最大1回に制限する。

- 警報ごとに`observe_detection(位置)`が（操作を許可するか、episode ID）を返す。
- 許可されたら通常の警報解決を行い、モデルが切り替わったら`mark_operation()`で「操作済み」にする。候補検証の確定で切り替わったときも同じ。
- 同じepisodeの中で操作済みなら、警報解決の代わりに`_resolve_episode_duplicate`を行う: FIFO全件を現行モデルへ吸収し、action `episode_suppressed`のイベントを記録し、検出器をresetしてFIFOをclearする。
- 無効（`FEDSDA_DETECTION_EPISODES_ENABLED = False`）のときは、常に（許可、IDなし）を返し、`mark_operation`は何もしない。

## 移植しない根拠（確認した事実）

- 既定値はFalse（`config.py` 94行）。最終構成の回帰（`tests/proposed_regression_golden.json`と`tests/test_proposed_regression.py`）はfalse。旧11ケースの回帰test、`tools/baselines/studies/`の比較条件にも、有効にした条件はない（2026-10-09にリポジトリ全体を`detection_episodes`・`DETECTION_EPISODES`で検索して確認）。
- したがって、最終構成と現在のgoldenでは、episode IDは常にNoneで、`episode_suppressed`の経路へは入らない。有効にした場合の新旧一致を確かめる基準（golden）がない。

確認していないこと: 過去の実験成果（`results/`）に、この機能を有効にしたrunがあるかどうか。CLI（`--detection-episodes`）と掃引軸には残っているので、過去に試した可能性はある。

## 新実装の現状

- 警報応答・完了処理・適応記録・候補検証session・`handle_alarm_occurrence`は、episode IDを`int | None`で受け取って記録まで運ぶ。完了処理と候補検証の確定の結果は、「episodeの操作が必要か」（学習帰属が変わったか）を返すpropertyを持つ（`detection_episode_operation_required`）。
- 新実装にないもの: episodeの状態（開始位置、操作済みか、次のID）を持つowner、警報の前に許可を求める処理、`episode_suppressed`に当たる適応結果の値と、その経路（全件吸収→記録→監視の再開→保留位置の消費）。
- 新の設定（`src/federated_learning_experiments/configuration/`）に、この機能の有効・無効の項目はない。

## 後で移植する場合

- 置く場所: `handle_alarm_occurrence`の呼出し側（標本1件の処理）。警報の前に許可とIDを求め、許可されなければ抑制の経路を実行し、許可されたら`handle_alarm_occurrence`へIDを渡して、戻り値と候補検証の確定の結果から操作を記録する。既存の部品と`handle_alarm_occurrence`の変更は要らない見込み（未検証）。
- 必要な検証: 実旧`DetectionEpisodeController`と`_resolve_episode_duplicate`を使う対照test、適応結果の値の追加、設定項目の追加、有効にした条件の基準（golden）の作成。
- 旧の掃引軸・CLI・manifestの`detection_episodes`を新の実行scriptへ引き継ぐかどうかは、掃引・CLIの切替えのspecで決める。それまで、新実装でこの機能を有効にする指定は受け付けない形にする（黙って無視しない）。

関連: [alarm-occurrence-handlingのresearch.md](../../../.kiro/specs/alarm-occurrence-handling/research.md)、[再開案内](../../../.kiro/steering/resume.md)の「未移植として残している細目」。
