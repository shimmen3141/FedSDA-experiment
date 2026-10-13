# Design Document

## Overview
送信保留のowner（`PendingModelUploadState`）の登録の扱い（方針）に、3つ目の値を足す。

| 方針の値 | 送信保留の保持 | 待ちを進めるもの（ラウンド境界） | 登録するもの（同期） | 使う所 |
|---|---|---|---|---|
| `register_pending_model_after_waiting_as_current_model`（新規） | 採用の順に、全部 | 現行モデルの送信保留だけ | 待ちが0で、現行の送信保留（高々1件） | FedSDAの既定 |
| `register_each_pending_model_by_its_own_id` | 採用の順に、全部 | 全部 | 待ちが0の全部（採用の順） | 基準版2・3、比較 |
| `keep_latest_pending_model_and_reassign_current_model_id` | 1つだけ（次の採用で置き換える） | 全部 | 待ちが0なら、現行モデル（現行が非負なら、送信保留を外すだけ） | 旧との照合、基準版1、FedDrift |

新しい方針は、判断の基準（「現行モデルとして、待ちの期間の学習を終えたモデルだけを共有する」）を、ラウンドを単位に近似して実装したもの。現行かどうかは、待ちの進行の時点（ラウンド境界）と登録の同期の時点の2つで判定する（ラウンドを単位にした近似。区間の途中で現行を外れ、次の時点までに戻ったモデルは、外れなかったものと同じに扱う。ユーザーが選んだ、ラウンド境界を単位にする進め方による）。

- 待ちは、ラウンド境界で、そのモデルが現行であるときだけ進める。外れている間は止める（やり直さない。ユーザーの選択）。
- 登録の時点でも、現行であることを条件にする。理由は2つの組合せ: (1) 判断の基準（LEGACY-016の記録。「clientの現行モデルとして、待ちの期間の学習を終えたときに限る」。「現行であること」は、ここから来る）。(2) 待ちの期間の定義（`docs/overview/sequence-diagrams.md`、`FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS=1`の説明。「作成の次ラウンドで学習された後、そのラウンド末にアップロード」。この文書は、現行であることには触れない）。待ちの数は、作成のラウンドの後の境界で0になるが、(2)の待ちの期間は、その次のラウンドの終わり（登録の同期）まで続く。その間に現行から外れたモデルは、(1)の「現行モデルとして、待ちの期間を終えた」に当たらない。待ちを進める条件だけでは、このモデルが登録される。これは、技術的な必要（欠番を避けるなど）ではなく、判断の基準の解釈である（新しい確認は、送信保留自身のIDで付け替えるので、この条件が無くても欠番は起きない）。登録の時点は、旧の確認が見ていた時点（同期の登録）と同じで、通常の場合（現行のまま）の結果を変えない（要求1.8）。主担当が設計の時点で足し、ユーザーへ伝えた（2026-10-14。仕様レビュー4回目の指摘で、理由の書き方を改めた）。
- 待ちが0のまま現行でない送信保留は、捨てずに保持する。後に現行へ戻れば、その同期で登録する（計数では、標本5000で、登録されなかった32件のうち6件が、後に長く現行だった）。現行へ戻らなければ、runの終わりまで送信保留のまま残る（共有しない。一時IDの採番は、採用ごとに新しいIDを使う——`temporary_model_id_allocation.py`——ので、残る送信保留とIDは衝突しない。FedSDAのclientは、一時IDの保有モデルを手放す操作を持たないので、送信保留を捨てる経路は要らない）。

## Boundary Commitments
### This Spec Owns
送信保留のownerの3つ目の方針と、現行のモデルIDを受け取る操作、正式IDの確認の対象の選び方、FedSDAのclientの登録の件数と待ちの進行、最終構成の既定、基準版2・3の設定の明示と基準版4、影響の計測script、修正の有無の比較script（実行と集計）の水準の追加。
### Out of Boundary
固定旧と旧golden、FedDriftのsource（`feddrift_run_client.py`・`feddrift_run_server.py`・`feddrift_detection_batch_processing.py`）、サーバの登録の繰返し（`register_ready_client_models`。件数をclientから受け取る形のまま）、集約・クロス評価・配布の中身、短い警報の扱い、候補の採否。
### Allowed Dependencies
既存のownerの付替えの操作。新しいmoduleは無い（方針の値の定数は、既存の設定のmoduleに足す）。確認のmodule（`held_model_registration_confirmation.py`）は、分岐の条件を「現行を付け替える方針と等しくない」へ改めるので、方針の値の定数のimportが、`REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME`から`KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME`へ替わる。依存境界の許可（`test_single_run_dependency_boundaries.py`の、moduleごとの名前の組）に、この名前の入替えが要るかを、実装の時点で確かめ、要れば入れ替える（許可が広すぎないことのtestがあるので、使わなくなった名前も外す）。
### Revalidation Triggers
送信保留の待ちと有無を読む箇所（FedSDAのclient、確認）、最終構成の既定、基準版の設定の組立て、掃引のrunの識別子（既定の値が変わるので、FedSDAの全runで変わる）。

## Architecture
### Existing Architecture Analysis
- `methods/fedsda/model_registration/pending_model_upload.py`: （送信保留、残りの待ち）の列。待ちの済んだものの有無・数・最も古いものの取得（`has_ready_model_upload`・`count_ready_model_uploads`・`get_ready_pending_model_upload`）と、待ちの進行（`advance_upload_readiness_at_round_boundary`）は、現行を知らない。
- `runtime/fedsda_run_client.py`: `has_model_ready_for_server_registration`・`count_models_ready_for_server_registration`・`advance_new_model_upload_wait_after_synchronization`が、上の操作を呼ぶ。clientは、owners経由で現行（`current_training_model_assignment`）を読める。
- `runtime/held_model_registration_confirmation.py`の`confirm_held_model_registration`: 方針が「それ自身のID」なら、`get_ready_pending_model_upload()`のモデルを付け替え、現行がそのモデルなら現行も変える。引数に、現行の持ち主を受け取っている。
- `runtime/server_model_registration_and_aggregation.py`の`register_ready_client_models`: clientごとに、件数を先に数え、その回数だけ、採番→来歴→確認。
- 待ちの進行は、同期の後（`execution/stream_protocol_execution_loop.py`が、clientの`advance_new_model_upload_wait_after_synchronization`を呼ぶ）。同期の中の登録は、その前。
- FedDriftは、同じownerを、1つだけ保持する方針で使い、現行を知らない操作を読む。

### Components and Interfaces
**方針の値**（`methods/fedsda/model_registration/pending_model_registration_settings.py`）
- `REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME = "register_pending_model_after_waiting_as_current_model"`を足し、`PENDING_MODEL_REGISTRATION_POLICIES`の末尾へ加える（既存の2つの並びは変えない）。

**`PendingModelUploadState`**
- 足す操作（3つの方針の全部で使える。現行のモデルIDを、keyword引数`current_training_model_id`で受け取る。型の検査は、どの状態を変える前にも行う）:
  - `count_model_uploads_to_register(*, current_training_model_id)`: 同期で登録する件数。新しい方針: 現行のIDの送信保留の待ちが0なら1、それ以外は0。ほかの2つの方針: `count_ready_model_uploads()`と同じ（現行を使わない）。
  - `get_next_pending_model_upload_to_register(*, current_training_model_id)`: 次に登録する送信保留か、None。新しい方針: 現行のIDの送信保留で待ちが0のもの。ほかの2つ: `get_ready_pending_model_upload()`と同じ。
  - `advance_upload_wait_after_synchronization(*, current_training_model_id)`: 新しい方針: 現行のIDの送信保留だけ、待ちを1減らす（0未満にしない）。現行のIDの送信保留が無ければ、何も変えない。ほかの2つ: `advance_upload_readiness_at_round_boundary()`と同じ。
- 現行を知らない既存の5つの操作（`has_ready_model_upload`・`count_ready_model_uploads`・`get_ready_pending_model_upload`・`advance_upload_readiness_at_round_boundary`と、読取りのproperty`remaining_upload_delay_round_count`）は、新しい方針では、状態を変えずに拒否する（要求2.4）。ほかの2つの方針では、これまでどおり（FedDriftと、既存のtestが使う。`remaining_upload_delay_round_count`を読むsourceは、FedDriftのサーバだけ）。
- 残りの公開操作（`get_pending_model_upload`・`get_all_pending_model_uploads`・`remove_pending_model_upload`・`clear_pending_model_upload`・`pending_model_registration_policy`）は、3つの方針の全部で、これまでどおり使える（現行に依らない、送信保留の一覧の読取りと除去）。FedSDAの経路で`clear_pending_model_upload`を呼ぶのは、確認の「現行を付け替える」の枝だけで、新しい方針では呼ばれない（新しい方針の確認は、`remove_pending_model_upload`で、登録した1つだけを外す）。初期登録の重複の検査（`get_all_pending_model_uploads`）と、採用の追加（`queue_model_upload`）は、新しい方針でも使う。拒否は、公開の操作の入口で行い、ownerの中の呼出し（ある操作が別の操作を呼ぶ）で、拒否を抜けない形にする。
- 足す操作（`queue_model_upload`・`queue_ready_model_upload`）: 新しい方針は、「それ自身のID」と同じく末尾へ足し、同じIDの送信保留があれば拒否する。
- 現行のIDが、どの送信保留のIDでもないとき（非負のモデルが現行など）、新しい方針の3つの操作は、0・None・何もしない、を返す（拒否しない。現行が送信保留でないのは、普通の状態）。
- 名前を、ownerの方針と照らす: 新しい方針は、送信保留を全部保持する方針の1つ。以降、「全部を保持する方針」は、新しい方針と「それ自身のID」の2つを指す。

**`confirm_held_model_registration`**（引数は変えない）
- 分岐の条件を、「方針が『それ自身のID』と等しい」から、「方針が『現行を付け替える』と等しくない」（全部を保持する2つの方針）へ改める（局所名も、全部を保持する方針であることを言う名前へ改める）。等しいかで判定したままだと、新しい方針が、現行を付け替える枝へ入る。
- 方針が「現行を付け替える」: これまでのまま。
- 方針が「それ自身のID」または新しい方針: `get_next_pending_model_upload_to_register(current_training_model_id=現行のID)`を対象にする（「それ自身のID」では、これまでの`get_ready_pending_model_upload()`と同じ結果）。サーバの登録は、採番と来歴の記録を、確認より先に行う（確認が拒否すると、採番が残る。既存の契約）。新しい方針では、件数（`count_model_uploads_to_register`）と確認の対象が、同じ条件（現行で待ちが0）で決まるので、件数が正なら対象があり、この拒否の経路は、正常の流れでは起きない。Noneなら拒否。以降（負であること・保有の検査→付替え→現行が対象なら現行を正式IDへ→その送信保留を外す）は、これまでの「それ自身のID」の処理と同じ。新しい方針では、対象は常に現行なので、現行が正式IDへ変わり、変更の記録を返す。

**`FedsdaRunClient`**
- `has_model_ready_for_server_registration()`: `count_model_uploads_to_register(current_training_model_id=現行) > 0`（bool）。
- `count_models_ready_for_server_registration()`: `count_model_uploads_to_register(current_training_model_id=現行)`。
- `advance_new_model_upload_wait_after_synchronization(round_index=...)`: `advance_upload_wait_after_synchronization(current_training_model_id=現行)`。
- 3つの方針の全部で、この形にする（方針の分岐は、ownerの中に閉じる）。「それ自身のID」と「現行を付け替える」では、これまでと同じ結果。

**設定と既定**
- `fedsda_final_configuration_run_settings.py`の既定を、新しい方針にする。構成の派生（11構成）は、既定を引き継ぐ。
- FedDriftのclientは、変えない（1つだけ保持する方針の明示のまま）。
- 実験のコマンドの引数には出さない。修正の有無の比較script（測定用の例外。これまでと同じ）だけが、方針を替える。

**基準版のgolden**（`tests/refactoring/behavior_golden_cases.py`）
- 版4の追加で更新する定義: `SUPPORTED_BASELINE_VERSIONS`、`BASELINE_VERSION`（4へ）、`CASE_NAMES_BY_BASELINE_VERSION`、（`_CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION`には、版4を入れない。版4の保存する設定は、警報を待つ扱いの項目を、既定の値——即時——のまま持つ。入れると、比べる前に項目が除かれる）、`golden_directory`が返す`v4`。研究室サーバ用は、版2〜4のどれも対象外（Windowsの基準環境とWSLだけ）。
- 版2・3の保存した設定は、登録の扱いの項目を、`register_each_pending_model_by_its_own_id`として持つ。既定を替えると、組み立てた設定と一致しなくなるので、「版の後に、既定の値が変わった項目」の表（版→経路→その版の値）を足し、`build_case_settings`が、その版の値を入れて組み立てる（保存した設定に項目があるので、比べるときに除かない）。既存の「版の後に足された項目」の表（`_CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION`）へ足してはいけない（`comparable_run_settings`が、その表の項目を、比べる前に除くので、保存した設定の項目と比べられなくなる）。
- 版4: 版3と同じケースに、`fedsda-sine2-alternative-reference-reuse`（`fedsda-sine2-events`と同じ設定で、seedが5）を足した19ケース。設定は、新しい既定のまま（警報を待つ扱いは、既定の即時）。実装の時点（2026-10-14）で、`fedsda-sine2-reference-reuse`（seed 4）が、版4で、別の保有モデルの再利用の判定（`alternative_reference_within_historical_loss_tolerance`）を通らなくなった（版2のときの`fedsda-sine2-events`と同じ現象）。seed 0〜15を調べ、版4の扱いで必須の経路を全部通るseed 5を選んだ。版4だけに足すので、版3のgoldenは変えない（このケースは、版の差の検査の対象外。経路を守るためのケース）。`fedsda-sine2-reference-reuse`の版4の必須の経路から、別の保有モデルの再利用を外した。必須の条件と版の差の説明は、下の「版の差の事象」で行う。版4で、版3の必須の経路（coverageの項目）が通らなくなるケースがあれば、通るseedのケースを足して守る。版2のときは、新しいケースを版2と版3の両方へ足した。版3のgolden（既存の基準）へケースを足すことになる場合は、ユーザーへ確認してから行う（版4だけに足すと、版の差の検査の対象にならない）。
- 版の差の事象: `capture_case_and_count_divergence_events`へ、`pending_model_not_current_while_waiting_count`を足す（task 3で足し、task 4の全体runのtestと、task 5の版の差のtestの両方が使う）。事象の辞書は、渡した版の扱いのrunで数える。このキーが意味を持つのは、版3の扱いのrun（次の版4との差の説明）。版の差のtestは、版の遷移ごとに、見るキーを1つ決めて読む（`BASELINE_VERSION_DIFFERENCES`。版1→2・2→3の比べ方は変えない）。版4の扱いのrunでは、(ii)は常に0だが、(i)は普通に起きる（現行でない送信保留を保持するので）ため、次の版の側では、このキーをassertしない。FedSDAのケースで、(i) 後に同期が続く、clientの待ちの進行の直前の、現行でない送信保留の数（そのclientの全部の送信保留のうち、IDが現行と違うものの数）と、(ii) 確認の直前に、対象の送信保留が現行でなかった回数の和。FedDrift・FedAvgは0。数え方: 実行の間だけ、`FedsdaRunClient.advance_new_model_upload_wait_after_synchronization`と、サーバの登録のmodule（`runtime/server_model_registration_and_aggregation.py`。確認の関数を名前でimportしている）の`confirm_held_model_registration`を包む（既存の短い警報の事象と同じ形）。(i)は、待ちの進行のたびに仮の数へ足し、次の同期の登録が呼ばれたときに、本の数へ移す。登録は、同期のmodule（`runtime/server_round_synchronization.py`）が、`register_ready_client_models`を名前でimportして呼ぶので、包むのは、そのmoduleの名前（登録のmoduleの関数を差し替えても、捕捉できない）。包んだ3つは、どれも`try/finally`で戻す（既存の短い警報の事象の包みと同じ。そちらは、完了処理——定義は`runtime/alarm_response_completion.py`——を名前でimportしている、警報の処理のmodule`alarm_occurrence_handling`の名前を包む）。最後の同期の後の待ちの進行の分は、本の数へ入らない。そこで2つの方針が分けるのは、clientの中の残りの待ちの値だけで、その後に登録が無く、残りの待ちの値は、記録の表・指標・パラメータに入らない（実装の時点で、出力を作る処理が、残りの待ちを読まないことを確かめる）。包みが読むのは、3つの方針の全部で許される操作と状態だけにする（`get_all_pending_model_uploads()`と、現行の持ち主の現行のID）。現行を知らない既存の操作（新しい方針では拒否）は、版4の実行や新しい既定の全体runでも包みが通るように、読まない。(ii)は、確認の直前と直後の送信保留の一覧の差から、外れた送信保留のIDを得て、それが確認の直前の現行のIDと違えば数える（確認と同じ選択の関数——`get_next_pending_model_upload_to_register`——を、観測の経路に使わない）。包みは、runの参加者を組み立てる前に入れる（参加者の組立てで、clientの操作が束縛されるtestの組立てがあるので、後から入れると数えられない）。
  - 0なら、版3の扱いと新しい方針は、同じ結果になる: 版3の扱いでは、同期の登録が待ちの済んだ全部を登録するので、待ちの進行の時点の送信保留は、全部、待ちが正である。そのすべてが現行（高々1つ）なら、2つの方針の待ちの進行は同じ。確認の対象がすべて現行なら、2つの方針の登録の対象も同じ。
  - 版の差の説明のtest（既存の`test_next_baseline_differs_only_where_the_fixed_behavior_occurred`の形）へ、版3→4を足す（`BASELINE_VERSION_DIFFERENCES`の`(3, 4)`は、キー`pending_model_not_current_while_waiting_count`と、事象が1以上の実ケースの名前（事象の数は`None`。値は見ない）。ケースの名前は、task 3の実行で確かめてから、事象が1以上のケースを全部書く。`(3, 4)`では、名前の無いケースで事象が1以上なら失敗する検査も足す）: 版3の扱いの実行で、この事象が0のケースは、版4の値（`run_metrics`・`record_tables`・`global_parameters`。`run_settings`は方針の項目で違うので比べない）が版3と同じ、1以上のケースは差がある。1以上のケースが、少なくとも1つある。名前の無いケースで事象が1以上なら失敗する検査は、既存のtestの本体に、`(3, 4)`の分岐として足す（既存の版1→2・2→3の判定は変えない）。
  - 1以上のケースが無いとき（版3のケースの小さい条件で、待ちの途中に外れる事象が起きないとき）は、版の差のtestが、前の版のケースを回し、名前を挙げたケースが両方の版にあることを確かめる形なので、版4だけにケースを足しても検査の対象にならない。版3のgolden（既存の基準）へのケースの追加が要るので、task 3で事象の数を確かめた時点で、ユーザーへ確認する。

## System Flows
採用（区間kの途中）: 一時IDで保有→送信保留へ足す（待ち=1）。同期k: 待ちが1なので、登録しない。同期kの後のラウンド境界: 現行なら、待ちを0へ。同期k+1: 待ちが0で現行なら、登録（これまでの既定・旧と同じ時点）。
- 区間kの途中で、次の採用があった: 前の送信保留は、現行でないので、待ちが進まない（1のまま）。後の送信保留が、上の流れで登録される。前のモデルが、後に現行へ戻れば、その後のラウンド境界で待ちが0になり、現行のままの同期で登録される。
- 区間k+1の途中で、現行が別のモデルへ移った: 同期k+1で、待ちは0だが現行でないので、登録しない。送信保留のまま保持する。後に現行へ戻れば、その同期で登録する。
- runの終わり: 送信保留のまま残るもの（現行へ戻らなかったもの、最後の区間の採用）は、登録されない（終端の手順は変えない）。

## 旧と違う点
- 「それ自身のID」（基準版3の扱い）と比べて: 待ちの途中で現行から外れたモデルは、待ちが止まり、登録の時点で現行でなければ、登録されない。登録されるモデルが減る（そのrunでは、以後の集約・クロス評価・統合・配布・予測の重み・乱数の消費が変わる）。版の差の事象が0のrunは、同じ結果。
- 旧（「現行を付け替える」）と比べて: 旧は、待ちが0になった同期で、現行が非負なら、送信保留を外し（以後、そのモデルは登録されない）、正式IDだけを採番する（採番したIDのモデルが、どこにも無い状態を「欠番」と呼ぶ）。新しい方針は、その送信保留を外さずに保持し、後に現行へ戻れば登録し、欠番を作らない。旧は、次の採用で、前の送信保留を置き換える。新しい方針は、置き換えない。旧は、確認のとき、現行が一時IDなら、送信保留のIDと照らさずに現行を付け替える（送信保留を外された後に現行へ戻ったモデルが、別の採用の確認で登録されることがある。計数で、標本2500・5000のそれぞれで1件あった）。新しい方針は、送信保留のモデルだけを付け替える。
- 同期の前の準備の確認（`has_model_ready_for_server_registration`の結果を、サーバの同期へ`new_model_registration_available`として渡す。`fedsda_run_server.py`は、それでクラスタリングを起動するかを決める）も、現行でない送信保留を数えないので、起動の有無が変わりうる（版の差の事象の(ii)の経路の一部）。
- 待ちを進めない条件が増えるので、待ちの途中で一度外れて戻ったモデルは、これまでより遅い同期で登録されることがある。
- 現行でない送信保留のモデルは、runの終わりまで、一時IDのまま、そのclientに保有される。保存した学習標本での学習は続き（`held_model_training_request_handling.py`は、標本を持つ保有モデルを学習する）、予測にも参加し、警報の区間の再利用の評価の対象にもなる（`alarm_interval_model_reuse_assessment.py`は、負のIDを含む全保有モデルを評価する）。再利用で現行へ戻れば、その後の同期で登録される。候補検証の参照に、そのモデルが含まれていて、検証の途中で登録されると、再利用の対象から外れる（LEGACY-019の既存の扱い）。発生の条件は、登録の時点が変わる分、「それ自身のID」と違いうる。どれも、版の差の事象が0のrunでは起きない。
- 送信保留のまま残るモデルは、保有モデルの数・学習量・予測と学習の計算量・保有モデル数の系列（`fedsda_model_diagnostic_metric_derivation.py`は、一時IDのモデルも数える）に入る。「それ自身のID」と比べると、登録されないモデルが、統合されずにclientに残るので、保有モデルの数が増えうる。影響の計測で、runの終わりの送信保留の数と、保有モデルの数を記録する。

## Validation Design
- owner: 3つの方針での、新しい3つの操作（新しい方針: 現行の送信保留だけ進む、現行でない待ち0は登録の対象にならない、現行が非負なら0・None・不変。ほかの2つの方針: 現行を知らない操作と同じ結果）。新しい方針での、現行を知らない5つの操作（4つの操作と、`remaining_upload_delay_round_count`）の拒否と状態の不変。足す操作の、末尾への追加と、同じIDの拒否。型の不正の拒否。
- 確認: 新しい方針の状態の遷移を直接確かめる（実旧に正解が無い）。(a)現行の送信保留が、全ownerで付け替わり、現行が正式IDへ変わる。(b)待ちが0でも現行でない送信保留は対象にならず、ほかに対象が無ければ、どの状態も変えずに拒否される。(c)現行の別の一時IDのモデル（送信保留でない）は付け替えられない。(d)そのclientの送信保留が1つだけで、それが現行の場合、同じ入力の「それ自身のID」の結果と、全ownerの状態が同じ（要求1.8の部品の側の根拠）。(e)待ちが0で現行でない送信保留Qと、後に採用した現行の送信保留Pがあるとき、「それ自身のID」はQを先に登録し、新しい方針はQを保持して、Pを同じ同期で登録する（正式IDがずれる列を、testで固定する）。「それ自身のID」と「現行を付け替える」は、既存のtest（実旧との対照を含む）が、変わらず通る。
- client: 実体のclient（既存の、登録のtestの組立て）で、新しい方針の、待ちの進行（現行だけ）、件数（現行が送信保留で待ち0なら1）、外れて戻ったモデルの登録。
- サーバ: 新しい方針で、1回の同期で、1 clientから高々1件。複数clientの採番の順（clientの順）。
- 全体run（要求3.4）: 条件は、基準版のケース`fedsda-sine2-events`の設定（task 3の後は、`build_case_settings(name, 3)`が方針を「それ自身のID」に固定するので、新しい既定のrunは、最終構成の既定のbuilderから、同じケースの条件で組み立てる。または、版4の定義を足した後の`build_case_settings(name, 4)`を使う）。新しい既定で、確認のたびに対象が現行であること（確認を包み、確認の直前と直後の送信保留の一覧の差から、外れた送信保留のIDを得て、それが確認の直前の現行のIDと等しいことを確かめる。確認と同じ選択の関数で確かめると、新しい方針では必ず真になり、回帰を検出しないので、使わない。版の差の事象の(ii)と同じ観測で、新しい既定では(ii)が0になることを確かめる、と同じ意味）、runの終わりに登録されなかった採用が0、送信保留のまま残る数を記録（assertしない）。同じ設定の「それ自身のID」の実行で、版の差の事象が1以上（経路を通ること）。0なら、事象の起きる別の条件を探して、testの条件にする（実装の時点で確かめる）。
- 構成: FedSDAの11構成の既定が新しい方針であること（既存の`test_every_fedsda_configuration_defaults_to_registration_by_own_id`を、新しい方針へ改め、名前も改める）。
- 既定の変更で、FedSDAの側で、既定のbuilderを通る設定を使い、方針に依る操作を読むtestと、既定の値をassertするtestの候補（task 4で、着手時に`grep`で確かめて直す）: `test_fedsda_final_configuration_run_settings.py`（既定が「それ自身のID」であることのassert）、`test_held_model_registration_confirmation.py`（ownerの操作の呼出しの順の監視）、`test_fedsda_run_client.py`（`remaining_upload_delay_round_count`を読む箇所）、`test_post_alarm_candidate_validation_progress.py`、`test_incomplete_post_alarm_candidate_validation_finalization.py`、前のspecの構成の既定のtest、共用のfresh scriptの方針の名前のassert。実旧と比べるものは、既に旧と同じ方針を明示しているので、変わらない。
- 旧との照合: 実旧と比べるtestは、旧と同じ方針を明示している（前のspecで、実旧の全体runと比べるhelperの中に、方針のassertを置いた）。既定を替える前に、既定のbuilderを呼ぶtestを`grep`で拾い、「それ自身のID」を前提にするtest（前のspecの全体runのtest、基準版2・3）に、その方針を明示する。その後で既定を替え、既定の値に依るtest（runの識別子の期待値など。失敗したtestから拾う）を合わせる。
- fresh process: 共用script（`tests/refactoring/fresh_process_smoke.py`）のFedSDAの全体runの流れは、既定を通るので、新しい方針を通る。その流れは、最終構成の既定の方針の名前（`register_each_pending_model_by_its_own_id`）をassertしているので、新しい方針の名前へ改める。runの終わりに、保有モデルが非負のIDか送信保留の中のものであること（登録されなかった採用が0）のassertは、新しい方針でも成り立つので、そのまま使う。
- 影響の計測（手元）: このspecのdirectoryの計測script（`measure_registration_after_waiting_effect.py`。Git管理。sourceを変えず、方針だけを替えて全体runを実行する）で、8条件×3方針（`keep_latest_pending_model_and_reassign_current_model_id`・`register_each_pending_model_by_its_own_id`・新しい方針）。指標は要求4.1。Windowsの基準環境で8並列（見積り1時間前後。手元の簡単な計測は構わない、と2026-10-14にユーザーが示した）。結果と実行したcommitを、LEGACY-016の記録とtasks.mdへ書く。本規模の影響は、このspecの後の、研究室サーバでの比較で測る（このspecは、コマンドの提示まで）。結果と実行したcommitを、LEGACY-016の記録とtasks.mdへ書く。
- 修正の有無の比較script: `run_behavior_fix_comparison.py`の登録の水準へ、`waited-as-current`（新しい方針）を足す（`legacy`・`fixed`は、名前と方針を変えない）。登録の水準と、短い警報の水準は、今は同じ定数（`LEVELS`）を共用しているので、軸ごとの定数に分ける（`analyze_behavior_fix_comparison.py`も同じ。短い警報の軸に`waited-as-current`が入らないようにする）。`run_behavior_fix_comparison.py`のdocstringの、`fixed`を既定とする説明を改める。`legacy`・`fixed`の組合せは、方針を明示して組み立てるので、既定の変更で、その設定とrunの識別子は変わらない（掃引の既定の構成のrunの識別子は、既定の変更で、FedSDAの全runで変わる。既存のサーバでの新旧比較——`compare_sweep_results_with_legacy.py`、提案構成の120 run——は、全runで旧と一致しており、旧と同じ方針の時期に作られた。既定が「それ自身のID」になった時点で、既定の結果は、その比較と一致しなくなっている（前のspecで記録済み）。新しい既定でも、一致しない。意図した差）。既定の水準の並び（`--registration-levels`を省いたとき）は、`legacy`・`fixed`のまま（既存の480 runの計画を変えない）。`analyze_behavior_fix_comparison.py`は、3つの水準を読み、データと集約間隔ごとに、組合せの絶対値（平均±標準偏差。今の第1節は平均だけなので、そのセルへ標準偏差を足す。標準偏差は、n−1で割った分散の平方根を計算する関数を新しく作る——既存の`mean_and_standard_error`は標準誤差を返すので使わない）と、対の差（新しい方針と旧・新しい方針と「それ自身のID」、短い警報の扱いの2水準それぞれ。短い警報の扱いの効果は、登録の3水準それぞれ。警報を待つ扱いの効果は、組合せがあるもの）を出す。データを越えた平均の行（今の第2節の「全部」の行。グループの列に足している`None`）は出さない。対の差の組合せは、今は固定の列挙なので、軸の水準から生成する形にする（対の差は、これまでどおり平均と標準誤差。表の見出しで、絶対値は標準偏差、対の差は標準誤差と書き分ける。runが1つのセルは、どちらも「—」にする。登録の効果: 登録の水準の対の全部×短い警報の各水準×警報を待つ各水準。短い警報の効果: 2水準の対×登録の各水準×警報を待つ各水準。警報を待つ扱いの効果: off→on×登録・短い警報の各水準。どれも、両方の組合せのrunがあるものだけを出す）。testは、既存の比較scriptのtestへ足す。
- サーバでの比較（このspecの後）: 新しい方針×（短い警報の旧／直した側／直した側と警報を待つ扱い）の3組合せ。比較scriptは、指定した水準の全部の組合せを作るので、2回に分ける: (1) `--registration-levels waited-as-current --short-alarm-levels legacy fixed`（警報を待つ扱いは既定のoff。240 run）、(2) `--registration-levels waited-as-current --short-alarm-levels fixed --alarm-wait-levels on`（120 run）。どちらも既存の480 runと同じ出力先へ足す。コマンドは、task 8で、計画を確かめてからユーザーへ示す。

## File Structure Plan
- `src/federated_learning_experiments/methods/fedsda/model_registration/pending_model_registration_settings.py`: 値の定数。
- `src/federated_learning_experiments/methods/fedsda/model_registration/pending_model_upload.py`: 3つの操作、既存の4つの操作の拒否、足す操作。
- `src/federated_learning_experiments/runtime/held_model_registration_confirmation.py`: 対象の選び方。
- `src/federated_learning_experiments/runtime/fedsda_run_client.py`: 件数と待ちの進行。
- `src/federated_learning_experiments/runtime/fedsda_final_configuration_run_settings.py`: 既定。
- `.kiro/specs/pending-model-registration-after-waiting-as-current-model/measure_registration_after_waiting_effect.py`: 手元の影響の計測。
- `tests/refactoring/`: `test_pending_model_upload.py`、`test_held_model_registration_confirmation.py`、`test_server_model_registration_and_aggregation.py`、新規`test_pending_model_registration_after_waiting_as_current_model.py`（状態の遷移・client・全体runの不変条件）、`test_pending_model_registration_by_own_id.py`（方針の明示と、構成の既定のtestの移動）、`behavior_golden_cases.py`と`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v4/`、`fresh_process_smoke.py`、比較scriptのtest、既定に依るtest。
- 直下: `run_behavior_fix_comparison.py`、`analyze_behavior_fix_comparison.py`。
- 文書: `docs/experiments/refactoring-baseline.md`（基準版4、版2・3の明示）、`docs/experiments/experiment-sweep.md`（既定の変更、比較scriptの水準）、LEGACY-016の記録と一覧、再開案内。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1 | ownerの3つの操作、確認の対象、clientの件数と待ちの進行、状態の遷移のtest、全体runの不変条件、現行が送信保留自身の場合の「それ自身のID」との一致 |
| 2 | 方針の値、既定、FedDriftの不変、既存の4つの操作の拒否、構成の既定のtest |
| 3 | 実旧との照合の方針の明示、版2・3の設定の明示、版4と版の差の事象、全体runの不変条件 |
| 4 | 手元の8条件×3方針の計測、比較scriptの水準と、データごとの集計（絶対値は標準偏差、対の差は標準誤差）、サーバでの比較のコマンドの提示 |
