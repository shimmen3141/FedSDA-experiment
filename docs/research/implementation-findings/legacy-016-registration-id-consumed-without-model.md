# LEGACY-016: 採用の後に現行モデルが戻ると、正式IDだけが採番され、採用したモデルは登録されない

- 発見日: 2026-10-10。対象: 旧`federated_drift_experiment/servers/fedsda.py::_register_new_models`（72〜87行）と、`clients/base.py::confirm_model_registration`（427〜462行）、`has_pending_model`（414〜415行）。固定基準748c3aa。
- 発見spec: server-model-registration-and-aggregation。種別: 正常な経路で起きる、状態の食い違い。状態: 再現済み・未修正。意図した仕様かどうかは未確認。

## 再現と観測

2026-10-10、Windows基準環境で、下書きのtest（commitしていない。同じ構成は`tests/refactoring/test_server_model_registration_and_aggregation.py`の対照testにある）を実行した。実旧の事前学習の結果を`SharedBackboneFedSDANoCachedServer`へ登録し、実`__init__`で作った`ResidualAdapterRestartingSoftRoutingFedSDAClient`を3つ登録して、ラウンドごとに、標本処理→保留中の学習→`_register_new_models(t)`→`update_global_models(…)`→`promote_pending_to_ready()`を実行した（配布は行わない）。

- あるclientが、候補を採用して一時ID（-100）のモデルを現行にした。送信までの待ち（2ラウンド）の間に、警報で、現行モデルが元のモデル（ID 0）へ戻った（保有モデルの再利用）。
- 待ちが済んだラウンドで、`has_pending_model()`が真になり、サーバは正式ID（2）を採番して、来歴（モデル2、そのラウンド、そのclient）を記録し、`confirm_model_registration(2)`を呼んだ。
- `confirm_model_registration`は、現行モデルのIDを「付け替える一時ID」として使う。現行が非負（0）なので、送信保留を外しただけで戻った。
- その後、サーバの`next_model_id`は3へ進み、来歴にはモデル2が残るが、グローバルモデルにID 2は現れない。clientの中には、一時ID（-100）のモデルが、学習データと統計とともに残り続ける。集約の対象は非負のIDだけなので、このモデルは集約されない。clientの予測には、保有モデルとして参加し続ける。

## 原因

clientは、送信保留のモデルのIDを持たない（`pending_model_params`・`pending_model_stats`・`pending_model_ready`だけ）。正式IDの確認は「現行モデルが、送信保留のモデルである」と仮定している。採用から送信までの間に現行モデルが変わると、この仮定が崩れる。

## 影響

- 正式IDに欠番ができる。来歴に、存在しないモデルの登録が残る。
- 採用したモデルは、そのclientの中だけで使われ、他のclientへ共有されない。再びそのモデルが現行になっても、送信保留は外れているので、登録されない。
- 過去の実験成果・goldenへの影響は未確認（最終3goldenの条件で起きているかどうかは調べていない）。

## 新実装の扱い

- 正式IDの確認（`confirm_held_model_registration`、held-model-registration-confirmation）は、旧と同じく、現在の学習帰属が非負なら送信保留を外すだけにしている。登録（`register_ready_client_models`）も、旧と同じく、採番と来歴の記録を行う。挙動を維持し、実旧との対照testが、この経路を通る。
- 新の送信保留（`PendingModelUploadState`）は、モデルIDを持つ。直すなら、正式IDの確認が、現在の学習帰属ではなく、送信保留のモデルIDを付け替えればよい。挙動が変わる（goldenが変わりうる）ので、移植とは分けて判断する。

関連: [fedsda-run-client-assemblyのtasks.md](../../../.kiro/specs/fedsda-run-client-assembly/tasks.md)の「Implementation Notes」（送信保留のモデルIDの照合）。
