# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `PENDING_MODEL_REGISTRATION_POLICIES` | `methods/fedsda/model_registration/pending_model_registration_settings.py`（設定のmodule。設定のmoduleは、`core`のほかに依存しない規則なので、値の定数は、ownerのmoduleでなく、こちらに置く） | 登録の扱いの、宣言した値の一覧 | 個々の値の定数（下の2つ）を並べたもの |
| `REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME` | 同上 | 値`register_each_pending_model_by_its_own_id`。送信保留を全部保持し、それぞれを、自身のIDで登録する | もう一方は、1つだけ保持して、現行を付け替える |
| `KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME` | 同上 | 値`keep_latest_pending_model_and_reassign_current_model_id`。送信保留を、最後に採用した1つだけ保持し、確認のとき、現行モデルのIDを正式IDへ付け替える（旧FedSDAとFedDriftの扱い。現行が非負なら、送信保留を外すだけ。現行が負なら、送信保留のIDと照らさずに、現行を付け替える） | 「legacy」としない（FedDriftは、手順に合うので使う） |
| `PendingModelUploadState.pending_model_registration_policy` | `methods/fedsda/model_registration/pending_model_upload.py` | このownerの方針（読取り） | 設定の項目（下）と同じ値 |
| `PendingModelUploadState.get_all_pending_model_uploads` | 同上 | 送信保留の全部を、採用の順で返す | `get_pending_model_upload`（単数）は、最も古い1つ |
| `PendingModelUploadState.get_ready_pending_model_upload` | 同上 | 待ちの済んだ送信保留のうち、最も古い1つ | `has_ready_model_upload`は、有無だけ |
| `PendingModelUploadState.remove_pending_model_upload` | 同上 | 指定したIDの送信保留を、1つ外す | `clear_pending_model_upload`は、全部を外す |
| `PendingModelUploadState.count_ready_model_uploads` | 同上 | 待ちの済んだ送信保留の数 | `has_ready_model_upload`は、有無だけ |
| `FedsdaRunClient.count_models_ready_for_server_registration` | `runtime/fedsda_run_client.py` | サーバが、1回の同期で、このclientから登録する件数（待ちの済んだ送信保留の数） | `has_model_ready_for_server_registration`は、有無だけ |
| `PendingModelRegistrationSettings` | `methods/fedsda/model_registration/pending_model_registration_settings.py`（module名も公開名） | 登録の扱いの設定型。項目は`pending_model_registration_policy`の1つ | ownerのpropertyは、この値を受け取ったもの。送信までの待ちのラウンド数（`new_model_upload_delay_round_count`）は、これまでの場所のまま |
| `FedsdaRunClientSettings.pending_model_registration_settings` | `runtime/fedsda_run_client_settings.py` | clientの設定の束の項目（保存するrun設定に入る） | — |

## 既存の公開名で、役割が広がるもの

| 公開名 | これまで | これから |
|---|---|---|
| `PendingModelUploadState` | 送信保留を1つ持つ | 方針に応じて、1つ、または、採用の順の列を持つ。生成に方針が要る |
| `PendingModelUploadState.get_pending_model_upload` | ただ1つの送信保留 | 最も古い送信保留（待ちが済んでいるとは限らない）。1つだけの方針では、これまでと同じ |
| `PendingModelUploadState.remaining_upload_delay_round_count` | ただ1つの送信保留の、残りの待ち | 最も古い送信保留の、残りの待ち。1つだけの方針では、これまでと同じ |
| `PendingModelUploadState.has_ready_model_upload` | ただ1つの送信保留の、待ちが済んだか | 待ちの済んだ送信保留が、1つでもあるか |
| `PendingModelUploadState.advance_upload_readiness_at_round_boundary` | ただ1つの待ちを進める | 待ちの残る全部を、1ずつ進める |
| `confirm_held_model_registration` | 現行モデルを付け替える | ownerの方針に応じて、現行モデル、または、待ちの済んだ最も古い送信保留のモデルを付け替える |
| `register_ready_client_models` | clientごとに、高々1件を登録 | clientごとに、待ちの済んだ送信保留の全部を登録 |

`PendingModelUpload`・`RegisteredClientModel`と、送信保留を足す操作（`queue_model_upload`・`queue_ready_model_upload`）、全部を外す操作（`clear_pending_model_upload`）は、名前・引数を変えない（足す操作は、方針に応じて、置き換えるか、末尾へ足す）。
