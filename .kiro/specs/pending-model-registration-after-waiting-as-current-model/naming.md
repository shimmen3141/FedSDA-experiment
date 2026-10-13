# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME` | `methods/fedsda/model_registration/pending_model_registration_settings.py` | 値`register_pending_model_after_waiting_as_current_model`。送信保留を全部保持し、現行である間だけ待ちを進め、待ちを終えて登録の時点で現行のモデルだけを、自身のIDで登録する（FedSDAの既定） | `REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME`は、現行かどうかに依らず、待ちの済んだ全部を登録する |
| `PendingModelUploadState.count_model_uploads_to_register` | `methods/fedsda/model_registration/pending_model_upload.py` | 現行のモデルIDを受け取り、同期で登録する送信保留の数を返す（新しい方針では、現行の送信保留で待ちが0なら1。`FedsdaRunClient.count_models_ready_for_server_registration`は、clientの層の同じ役割の操作で、これを呼ぶ） | `count_ready_model_uploads`は、現行を知らず、待ちの済んだ数だけを返す（新しい方針では拒否） |
| `PendingModelUploadState.get_next_pending_model_upload_to_register` | 同上 | 現行のモデルIDを受け取り、この同期で次に登録する送信保留を返す（無ければNone）。新しい方針では、現行の送信保留で待ちが0のもの。「それ自身のID」では、待ちの済んだ最も古いもの | `get_ready_pending_model_upload`は、現行を知らず、待ちの済んだ最も古いもの（新しい方針では拒否）。`get_pending_model_upload`は、待ちに依らず、最も古いもの |
| `PendingModelUploadState.advance_upload_wait_after_synchronization` | 同上 | 現行のモデルIDを受け取り、同期の後に待ちを進める（新しい方針では、現行の送信保留だけ） | `advance_upload_readiness_at_round_boundary`は、現行を知らず、全部を進める（新しい方針では拒否）。`FedsdaRunClient.advance_new_model_upload_wait_after_synchronization`は、clientの層の同じ役割の操作で、ownerのこの操作を、現行のIDを渡して呼ぶ。どの送信保留を進めるかは、ownerの方針が決める（新しい方針だけ、現行の送信保留に限る） |

既存の、現行を知らない操作（`count_ready_model_uploads`・`get_ready_pending_model_upload`・`advance_upload_readiness_at_round_boundary`）の名前は変えない（FedDriftと既存のtestが使う）。新しい3つの操作は、「登録する（to register）」「同期の後（after synchronization）」の語で、現行のモデルIDを受け取って登録の判断まで含むことを示し、既存の「待ちが済んだ（ready）」の操作と区別する（仕様レビュー2回目の指摘で、`registrable`・`at_round_boundary`から改めた）。

比較scriptの登録の水準`waited-as-current`（`run_behavior_fix_comparison.py`）は、新しい方針（現行として待った後に登録する）を指す。「現行」の概念そのものと紛れないよう、`current`だけにしない。

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `PENDING_MODEL_REGISTRATION_POLICIES` | 2つの値 | 3つの値（新しい値を末尾へ） |
| `PendingModelUploadState.has_ready_model_upload`・`count_ready_model_uploads`・`get_ready_pending_model_upload`・`advance_upload_readiness_at_round_boundary`・`remaining_upload_delay_round_count` | どの方針でも使える | 新しい方針では、状態を変えずに拒否する（ほかの2つの方針では、これまでどおり） |
| `confirm_held_model_registration` | 「それ自身のID」の方針では、待ちの済んだ最も古い送信保留を付け替える | 全部を保持する2つの方針で、`get_next_pending_model_upload_to_register`の送信保留を付け替える（「それ自身のID」では、これまでと同じ対象） |
| `FedsdaRunClient.has_model_ready_for_server_registration`・`count_models_ready_for_server_registration`・`advance_new_model_upload_wait_after_synchronization` | ownerの、現行を知らない操作を呼ぶ | ownerの、現行のモデルIDを受け取る操作を呼ぶ（「それ自身のID」と「現行を付け替える」では、これまでと同じ結果）。名前は変えない（実行の枠の契約と、実行の記録の段の名前に及ぶため）。名前の「ready」は、新しい方針では「待ちが済み、かつ現行」を含む |

比較scriptと計測script（直下の`run_behavior_fix_comparison.py`・`analyze_behavior_fix_comparison.py`、specのdirectoryの計測script）の中の名前（標準偏差の関数、水準の定数、水準の値`waited-as-current`）は、sourceの公開する名前ではないので、この表の対象外（命名の規約の範囲は、sourceの公開する名前）。標準偏差の関数は、既存の`mean_and_standard_error`（標準誤差）と紛れない名前にする。
