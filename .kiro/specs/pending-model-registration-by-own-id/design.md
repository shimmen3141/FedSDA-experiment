# Design Document

## Overview
送信保留のowner（`PendingModelUploadState`）に、登録の扱い（方針）を持たせる。方針は2つ。

| 方針の値 | 送信保留の保持 | 正式IDの確認で付け替えるモデル | 使う所 |
|---|---|---|---|
| `register_each_pending_model_by_its_own_id` | 採用の順に、全部（それぞれに待ち） | 送信保留のモデル自身 | FedSDAの既定 |
| `keep_latest_pending_model_and_reassign_current_model_id` | 1つだけ（次の採用で置き換える） | 現行モデル（現行が非負なら、送信保留を外すだけ。現行が負なら、送信保留のIDと照らさずに、現行を付け替える） | 旧との照合、基準版1、FedDrift |

旧と同じ扱いを設定で残すのは、次の条件を満たすからである（2026-10-13、ユーザーへ説明）: 分岐が、登録の機能の中（送信保留のownerと、正式IDの確認）に閉じる。既定は直した側で、実験のコマンドの引数には出さない。旧実装を外すときに、値と分岐を一緒に外す（外す箇所は、NEW-004の記録へ足す）。分岐が複数の機能にまたがる修正には、この形を使わない。

方針をownerに持たせるのは、送信保留の保持の仕方と、確認の対象が、1つの決定の2つの面で、別々に選べると、旧にも新にも無い組合せができるため。サーバの関数へ、新しい引数を通さずに済む。

## Boundary Commitments
### This Spec Owns
送信保留のownerの方針と複数保持、正式IDの確認の対象の選択、サーバの登録の繰返し、clientの設定の1項目、FedDriftのclientの方針の明示、基準版2のgoldenと基準版1の照合の続け方、影響の計測。
### Out of Boundary
固定旧と旧golden、FedDriftのサーバの登録の手順、候補の採否、集約・クロス評価・配布の中身、ほかのLEGACY。
### Allowed Dependencies
既存のownerの付替えの操作（`reassign_*`、`transfer_model_training_and_assignment_counts`、`assign_model_for_training`）。sourceの新しいimportは、新しい設定型のmoduleと、方針の値の定数だけ（どれも、登録の機能のmodule。依存境界は、新しいmoduleの許可集合と、読む側の登録を足す）。
### Revalidation Triggers
送信保留を読む箇所（初期登録の重複の検査、FedDriftのサーバと検出バッチの処理）、clientの設定を組み立てる箇所、run設定の保存、掃引のrunの識別子（設定が変わるので、FedSDAの全runで変わる）。

## Architecture
### Existing Architecture Analysis
- `methods/fedsda/model_registration/pending_model_upload.py`: 1枠（`_pending_model_upload`と残りの待ち）。`queue_model_upload`と`queue_ready_model_upload`は、呼ばれると、枠の中身を、新しい送信保留で上書きする（この2つの操作の名前と引数は、このspecで変えない）。
- `runtime/held_model_registration_confirmation.py`の`confirm_held_model_registration`: 現行モデルのIDを付け替える。現行が非負なら、送信保留を外して戻る。
- `runtime/server_model_registration_and_aggregation.py`の`register_ready_client_models`: clientごとに、待ちの済んだ送信保留があれば、1回だけ採番・来歴の記録・確認。
- 配布の反映は、一時IDのモデルを、そのまま保つ（`kept_temporary_model_training_states`）。予測の重みは、保有モデルのIDの集合が変わると、均等へ戻る（どちらの方針でも、付替えで起きる）。
- FedDriftは、同じownerと確認の関数を使う（新しいモデルは、作った時点で現行で、次のサーバの手順で登録される）。

### Components and Interfaces
**`PendingModelUploadState`**（方針つき）
- 生成: `PendingModelUploadState(pending_model_registration_policy=...)`。値は`PENDING_MODEL_REGISTRATION_POLICIES`の2つ。既定値は置かない（呼ぶ側が、必ず選ぶ）。値の定数は、設定のmodule（`pending_model_registration_settings.py`）に置き、ownerのmoduleが、そこから読む（設定のmoduleは、`core`のほかに依存しない、既存の規則のため）。
- 内部は、（送信保留、残りの待ち）の列（採用の順）。足す操作は、これまでの`queue_model_upload`と`queue_ready_model_upload`のまま（名前・引数を変えない。呼ぶ側——`adopted_candidate_initial_local_registration.py`、`feddrift_detection_batch_processing.py`、`feddrift_run_server.py`——のsourceは、変えない）。1つだけ保持する方針では、足す操作が列を空にしてから足す（これまでと同じ結果）。全部保持する方針では、末尾へ足す。同じモデルIDの送信保留が既にあれば、足す前に拒否する（全部保持する方針だけ。1つだけの方針は、置き換えるので、これまでどおり受け入れる）。
- `get_pending_model_upload()`: 列の先頭（最も古い。待ちが済んでいるとは限らない）か、None。`remaining_upload_delay_round_count`: 先頭の残りの待ち（空なら0）。1つだけの方針では、これまでと同じ値。FedDrift（1つだけの方針）だけが、この2つを読む。
- `get_all_pending_model_uploads()`: 全部を、採用の順のtupleで返す（新設）。
- `has_ready_model_upload()`: 残りの待ちが0のものが、1つでもあれば真。`get_ready_pending_model_upload()`: 残りの待ちが0のもののうち、最も古いものか、None（新設）。
- `advance_upload_readiness_at_round_boundary()`: 残りの待ちが正のもの全部を、1減らす。
- `remove_pending_model_upload(model_id=...)`: そのIDの送信保留を1つ外す。無ければ拒否（新設）。`clear_pending_model_upload()`は、全部を外す（これまでと同じ）。
- `pending_model_registration_policy`: 読取りのproperty。

**`confirm_held_model_registration`**（引数は変えない）
- 方針が「現行を付け替える」: これまでの処理のまま（現行が非負なら、`clear_pending_model_upload()`で送信保留を外して戻る短絡を含む）。
- 方針が「それ自身のID」のときは、この短絡と`clear_pending_model_upload()`を通らない（通ると、送信保留を捨てる。LEGACY-016そのもの）。testで、現行が非負のときに、送信保留のモデルが付け替わることを確かめる。
- 方針が「それ自身のID」: 検査（どの更新より前）→ `get_ready_pending_model_upload()`がNoneなら拒否。そのモデルIDが負でない、または保有されていないなら拒否。→ 付替え（学習状態→損失統計→学習標本→評価標本→計数。これまでと同じ順）→ 現行がそのモデルなら、現行を正式IDへ（変更の記録を返す）。違えば、現行を変えず、Noneを返す→ その送信保留だけを外す。
- 付替えの前の検査で、正式IDが、どのownerでも使われていないことは、各ownerの付替えの操作が確かめる（これまでと同じ）。途中のownerで拒否されると、先のownerの付替えが残る点も、これまでと同じ（この関数の既存の契約。直さない。要求1.4は、この場合を対象外としている）。
- 候補検証のsessionが持つモデルID（参照モデルの対応、開始時の学習帰属ID）は、付け替えない（要求1.6）。進行中の検証は、保有モデルのIDの集合に無い参照を、再利用の対象から外す（`post_alarm_candidate_loss_evaluation.py`の`_select_validated_reusable_reference`。旧`_finalize_forward_validation`の`if model_id in self.models`と同じ）。統合のID対応でも、旧と同じ方針の現行の付替えでも、同じことが起きる既存の扱いで、既定の方針では、現行でないモデルの登録でも起きるようになる。検証の比較の相手（損失の合計が最小の参照）は、開始時に固定した全参照から選ぶので、付け替わった後の、保有していないIDが、判定の記録の比較の相手のIDとして残ることがある（保有モデルは引かないので、例外にはならない。旧と同じ）。扱いそのものは、[LEGACY-019](../../../docs/research/implementation-findings/legacy-019-renamed-reference-excluded-from-validation-reuse.md)へ記録する（このspecでは直さない）。

**`register_ready_client_models`**
- clientごとに、`has_model_ready_for_server_registration()`が真なら、登録を始める前に、待ちの済んだ送信保留の数（`count_models_ready_for_server_registration()`。ownerの`count_ready_model_uploads()`）を数え、その回数（最低1回）だけ、採番→来歴→確認を繰り返す（1件の中の順は、これまでと同じ。確認が拒否すると、採番と来歴が残る点も、これまでと同じ）。1つだけの方針では、送信保留は1つなので、これまでどおり、1回。回数を先に決めるので、繰返しは必ず止まる（実装の時点で、「真の間、繰り返す」形から改めた: clientの「有無」の答えを差し替える既存のtestで、止まらなくなるため）。
- `RegisteredClientModel`は、これまでの3項目のまま（登録1件につき1つ。同じclientが複数回現れうる）。

**設定**
- 新しい設定型`PendingModelRegistrationSettings`（`methods/fedsda/model_registration/pending_model_registration_settings.py`）が、項目`pending_model_registration_policy: str`を持つ。宣言した値は、fieldのmetadataの`allowed_parameter_values`（`PENDING_MODEL_REGISTRATION_POLICIES`）で、`validate_settings_field_values`が拒否する（`ModelConsolidationSettings`と同じ形）。`FedsdaRunClientSettings`へ、項目`pending_model_registration_settings`として足す（機能別の設定型。置き場所のない値の束`FedsdaRunClientScalarSettings`へは足さない）。`runtime/fedsda_run_client.py`の`assemble_fedsda_run_client`が、ownerの生成へ、方針の値を渡す。
- 実験のコマンド（`run_experiment_sweep.py`）の引数には出さない。掃引は、常に既定の方針で実行する。旧と同じ方針は、testの設定の組立てと、影響の計測script（測定用）からだけ選ぶ。設定が変わるので、FedSDAのrunの識別子（設定の内容から作る）は、全runで変わる。このspecより前に新実装で実行した結果（2026-10-11のサーバでの新旧比較など）は、旧と同じ方針の結果で、既定の方針の結果とは別のrunとして扱われる（出力先が同じでも、済んだrunとして飛ばされない）。`compare_sweep_results_with_legacy.py`での新旧の比較は、既定の方針では、一致しないrunが出る（意図した差）。
- 最終構成の既定（`fedsda_final_configuration_run_settings.py`）は、`register_each_pending_model_by_its_own_id`。構成の派生（11構成）は、既定を引き継ぐ。
- FedDriftのclientは、`keep_latest_pending_model_and_reassign_current_model_id`を、定数で明示して渡す（設定の項目は足さない。FedDriftの保存するrun設定は、変わらない）。

**初期登録の重複の検査**（`adopted_candidate_initial_local_registration.py`）
- 採番予定の一時IDが、送信保留の**どれか**のIDと同じなら拒否する（`get_all_pending_model_uploads()`を読む）。

## System Flows
採用: 候補を一時IDで保有→送信保留へ足す（待ち=設定のラウンド数。最終構成は1）。ラウンド境界: 同期（登録→集約→…→配布→再較正）の後で、待ちを1進める（これまでと同じ位置。`execution/stream_protocol_execution_loop.py`）。区間kで採用したモデルは、同期kでは待ちが残り、同期kの後に待ちが0になって、同期k+1で登録される。したがって、待ちを進めた後には、待ちの済んだ送信保留が、普通に残る（次の同期で登録される）。「待ちの済んだ送信保留が残らない」の判定の時点は、登録の処理（`register_ready_client_models`）が戻った時点である。最後の区間で採用したモデルは、以後の同期が無いので、送信保留のままrunが終わる（旧と同じ。終端の手順は変えない）。登録: clientの順→そのclientの、待ちの済んだ送信保留を、古い順に、採番・来歴・確認。

登録されたモデルは、そのラウンドの集約から、非負のIDのモデルとして、集約・クロス評価・統合・配布の対象になる（これまで、現行のまま登録されたモデルと同じ扱い）。現行でないまま登録されたモデルも、学習標本（採用のときの保留標本）と統計を持つので、集約の重みは、これまでの式で決まる。採用のときの保留標本は、変化区間の標本で、最小件数（`minimum_change_interval_sample_count`、1以上）より少なければ候補検証を始めないので、1件以上ある（集約は、学習標本が0件のモデルの値を作らないが、その場合は起きない）。

FedDriftでは、2つの方針の結果は分かれない: 集約の間隔が検出バッチの件数と同じで（`feddrift_run_settings.py`）、新しいモデルは、時刻の最後の標本の処理で作られて現行になり（`feddrift_client_stream_processing.py`）、同じ時刻のサーバの手順で登録される。確認のとき、送信保留のモデルは、常に現行である（要求4.5のtestで固定する）。FedDriftは、1つだけの方針を明示して使う。

## 旧と違う点（既定の方針のとき）
1. 待ちの間に現行モデルが変わっても、送信保留のモデルが登録される（旧: 採番だけして、登録されない）。
2. 待ちの間の次の採用で、前の送信保留が残る（旧: 置き換えられ、前のモデルは登録されない）。
3. 現行でないモデルの登録でも、保有モデルのIDの集合が変わるので、次の予測で、予測の重み（Fixed-Share）が均等へ戻る（旧は、現行モデルの付替えのときだけ。`fixed_share_weights.py`の`_synchronize_model_pool`）。AdaHedgeの証拠は、登録の確認からは再始動されない（確認による学習帰属の変更は、診断へ通知しない。旧も同じ。`server_model_registration_and_aggregation.py`の注記）。再始動は、警報と候補検証の確定での学習帰属の変更のときだけで、証拠は、同期のたびに、集約後の再較正で再生される。現行でないモデルの登録で、新しい再始動は起きない。進行中の候補検証があれば、付け替わったモデルは、その検証の再利用の対象から外れる（要求1.6）。
4. 上が起きたrunでは、登録されるモデルが増え、以後の集約・クロス評価・統合・配布・予測の重み・乱数の消費が変わる。起きないrunは、旧と同じ。

旧と同じ方針では、違いは無い（実旧との照合testが、これまでどおり確かめる）。

## Validation Design
- owner: 2つの方針での、足す・置き換える・待ちを進める・外す・拒否。1つだけの方針は、これまでのtestを、方針を明示して、そのまま通す。
- 確認: 実旧のclientとの対照（旧と同じ方針。これまでのtest）。それ自身のIDの方針は、実旧に正解が無いので、状態の遷移を直接確かめる: 現行が非負・現行が別の一時ID・現行が送信保留自身の3つで、全ownerの付替えと、現行・送信保留の結果。確認の冒頭の検査での拒否（要求1.4の対象: 待ちの済んだ送信保留が無い、送信保留のモデルが保有されていない、一時IDでない、ownerや正式IDの型・範囲の不正）は、全ownerの状態の不変。正式IDが、あるownerで使用済みのための、付替えの途中の拒否は、この対象外（先のownerの付替えが残る、既存の契約）。現行が送信保留自身の場合は、同じ入力の、旧と同じ方針の結果（実旧と一致済み）と、全ownerの状態が同じであること。
- サーバ: 1 clientに待ちの済んだ送信保留が2つ、複数client、の採番の順と来歴。
- 観測の方法（新しい記録は足さない）: 「登録されなかった採用」は、runの終わりのclientの状態から数える。clientの保有モデルのうち、一時ID（負）で、送信保留のどれのIDでもないものの数（旧と同じ方針で、欠番になったモデルと、上書きされたモデルが、これに当たる）。runの終わりに送信保留のまま残るモデル（待ちの途中、または、待ちが済んで次の同期を待っていたもの）は、別の数として数える（どちらの方針でも起きる）。全体runの結果（`FedsdaExperimentRun.measured_run.participants`）から読める。testのhelper 1つにまとめ、全体runのtest・基準版のcoverage・影響の計測で、同じものを使う。
- 全体run: 条件は、基準版1のケース`fedsda-sine2-events`の設定（sine2、client 3、標本1500、集約間隔50、seed 0）。2026-10-13に、この設定の、旧と同じ扱いの実行で、登録されなかった採用が2件（client 0の-100と、client 2の-102）あることを確かめた（ほかの12のFedSDAのケースは0件）。既定の方針では、登録されなかった採用が0で、同期のたびに、登録の処理が戻った時点で、待ちの済んだ送信保留が残らないこと（testの中で、登録の関数の戻りを観測する）。同じ設定の、旧と同じ方針では、登録されなかった採用が2であること。この設定が通る経路（現行でないモデルの登録か、送信保留がある間の次の採用か）は、testの中で、確認と採用の呼出しを観測し、少なくとも一方を通ったことをassertする（通らなければ、testが失敗する）。通らなかった側の経路は、部品のtest（ownerと、実体のclientを使う登録のtest）と、影響の計測（上書きが起きたsine2 seed 1・mnist2 seed 1で、既定の方針の登録されなかった採用が0になること）で確かめる。全体runのtestへ、もう一方の経路を通る小さい条件を足すのは、任意とする。
- 進行中の候補検証（要求1.6）: 実装は変えない。testは、新しく足さず、既存の`tests/refactoring/test_post_alarm_candidate_validation_progress.py`の`test_validation_progress_uses_current_reference_availability`の`reference_id_no_longer_held`の条件を、この要求の証拠とする（検証の途中で、保有モデルのIDを、ownerの付替えの操作で付け替え、検証が、そのモデルを再利用の対象にせずに確定し、実旧の結果と一致することを確かめている。確認の関数が呼ぶ付替えと、同じ操作）。以下は、当初の案:旧の`if model_id in self.models`に当たるのは、検証の確定が、その時点の保有モデルのIDを、利用できる参照として評価へ渡す所（`runtime/post_alarm_candidate_validation_progress.py`が、保有モデルの一覧から`available_reference_model_ids`を作って、`evaluate_candidate_using_post_alarm_losses`へ渡す）。testは、実体のclient（既存の、候補検証の進行のtestの組立て）で、一時IDの保有モデルを参照に含む検証を始め、途中で、既定の方針の確認を呼んで、そのモデルを正式IDへ付け替え、残りの標本を観測して、検証が、付け替わったモデルを再利用の対象にせずに、確定まで進むことを確かめる。
- FedDrift: 既存の全体runのtestの中で、確認の呼出しを観測し、毎回、送信保留のモデルが現行であること（要求4.5。FedDriftは1つだけの方針を明示するので、動作は変わらない。この確認は、その方針がFedDriftの手順に合っていること——登録されないモデルを作らないこと——の根拠）。
- 構成: 掃引の全構成のうち、FedSDAの11構成の設定が、既定の方針であること（要求3.3）を、1つのtestで確かめる。
- 同期のたびの確認（登録の処理が戻った時点で、待ちの済んだ送信保留が残らない）は、上の1条件（`fedsda-sine2-events`の設定）で行う。runの終わりの、登録されなかった採用が0であることは、基準版2の全ケースで確かめる。
- 旧との照合: testの設定の組立て（`tests/refactoring/test_fedsda_run_client.py`の設定、実旧と比べる全体run・構成のhelper）は、旧と同じ方針を明示する。照合の対象・項目は、減らさない。最終構成の既定のbuilder（`build_experiment_run_settings`、`build_final_configuration_fedsda_run_settings`ほか）を通って設定を作るtestは、既定を替えると、黙って既定の方針になる（経路を通らない条件では、そのまま通ってしまう）。そこで、(a)既定を替える前に、既定のbuilderを呼ぶtestの全箇所を`grep`で拾い、実旧と比べるものへ、旧と同じ方針を明示する。(b)実旧の全体runと比べるhelper（`test_fedsda_run_metric_derivation.py`の`compare_small_run_with_real_legacy`、`test_fedsda_stream_protocol_run.py`の`run_real_legacy_whole_run`、`test_legacy_computation_count_audit.py`の`run_real_legacy_whole_run_counting_model_computation`。新の設定を受け取る所）の中で、新の設定の方針が、旧と同じ方針であることをassertする（明示を忘れたtestは、失敗する）。
- golden: `behavior_golden_cases.py`を、基準版ごとの定義（版の番号、directory、ケースの名前、設定の組立て、coverageの項目）に分ける。helperの関数（設定の組立て、読込み、環境からの選択、生成とそのworker、実行）は、版を引数に取り、ファイルへ書く`baseline_version`、読込みで確かめる`baseline_version`、directoryの全部に、その版の値を使う（定数`BASELINE_VERSION`を、生成のworkerがそのまま書く今の形を、やめる）。`schema_version`は1のまま（ファイルの形式は変えない。coverageの項目が増えるのは、版の定義の差）。現在の版は、helperの定数で明示する（最新のdirectoryを自動で選ばない）。
  - 保存してある`run_settings`と比べる所は2つある: 読込みの構造の検査（`load_golden`）と、実行した結果との照合（`capture_case`が返す`run_settings`を、`assert_case_matches`で比べる。キーの集合も比べる）。両方が、同じhelper（版と設定から、「比較用のrun_settings」を返す関数）を通る。基準版1の、下の(1)〜(3)は、このhelperの中で行う。
  - 基準版1: 設定は、旧と同じ方針で組み立てる。保存してある`run_settings`との比較は、(1)保存してあるv1の`run_settings`に、新しい項目`pending_model_registration_settings`が無いこと、(2)組み立てた設定の、その項目の値が、旧と同じ方針であること、を確かめてから、(3)組み立てた設定から、その項目（保存の形では、`run_participant_settings`→`run_client_settings`→`pending_model_registration_settings`）を除いて、完全な一致で比べる（(2)を飛ばすと、既定の方針のまま、項目を除いて通ってしまう）。値（指標・表・パラメータ・coverage）は、これまでどおり、全部を比べる。v1のファイルは、変更しない。
  - 基準版2: 設定は、既定の方針。全項目を、完全な一致で比べる。coverageへ、「登録されなかった採用の数」を、全ケースに足す（FedDrift・FedAvgは、定義により0）。0であることを、必須の条件にする。ケースは、基準版1の17ケース（同じ設定）に、`fedsda-sine2-reference-reuse`（`fedsda-sine2-events`と同じ設定で、seedだけ4）を足す。理由: 基準版2の扱いでは、`fedsda-sine2-events`が、別の保有モデルの再利用の判定（`alternative_reference_within_historical_loss_tolerance`）を通らなくなる（登録されずにclientへ残っていたモデルが、登録・統合されるため）。その経路を、基準版2でも守るために、通るseed（0〜7を調べて、4を選んだ。基準版2で、採用4・棄却7・統合2・別の保有モデルの再利用あり）のケースを足した。ケース名は、版ごとに持つ（`CASE_NAMES_BY_BASELINE_VERSION`）。必須の条件: `fedsda-sine2-events`で、同じ設定の旧と同じ方針の実行に、登録されなかった採用が2件あり、基準版1の値と、基準版2の値に差がある。
  - 版の差の説明: ケースごとに、旧と同じ方針の実行の「登録されなかった採用の数」が0なら、基準版2の値（指標・表・パラメータ）が、基準版1の値と同じであること、1以上なら、差があること、を確かめるtestを置く。基準版1のファイルは、この数を持たない（変更しない）ので、testが、旧と同じ方針で各ケースを実行して数える。比べるのは、指標・表・パラメータ（`run_settings`と`coverage`は、版で項目が違うので、比べない）。FedDrift・FedAvgのケースは、登録されなかった採用の数を0と定義する（方針の設定を持たず、基準版1と同じ設定で実行する）ので、基準版2の値が基準版1と同じであることを確かめる。旧と同じ方針の実行は、基準版1の照合のtestが行う実行と同じなので、ケースごとに1回だけ実行して結果を持つ、module範囲のcache（ケース名→結果）を使い、両方のtestで共有する（環境に合うgoldenが無くて数値の照合をskipする環境では、版の差のtestもskipする）。基準版2は、Windowsの基準環境とWSLの2つ（基準版1と同じ。研究室サーバ用は、対象外）。
  - 基準版2の生成: 既存の生成手順（commit済みのsource、2つのprocessの一致、未使用の出力先）を、版を指定して使う。sourceとtestをcommitしてから、Windowsの基準環境とWSLで生成し、`behavior_regression_goldens/v2/`へ足す。commit済みのgoldenが、2つの基準環境を覆うことの検査を、版ごとに行う。
- 影響の計測: このspecのdirectoryへ置く計測script（`measure_registration_policy_effect.py`。Git管理。sourceを変えず、設定の方針だけを替えて、全体runを実行する）で、8条件（client 10、標本2500、集約間隔50、sine2・sea4・circle2・mnist2のseed 0・1）×2方針。精度・定常精度・通信量・最終のグローバルモデル数・登録されなかった採用の数・runの終わりに送信保留のまま残った数を出す。結果と、実行したcommitを、LEGACY-016の記録とtasks.mdへ書く。

## File Structure Plan
- `src/federated_learning_experiments/methods/fedsda/model_registration/pending_model_upload.py`: 方針と複数保持。
- `src/federated_learning_experiments/runtime/held_model_registration_confirmation.py`: 方針による対象の選択。
- `src/federated_learning_experiments/runtime/server_model_registration_and_aggregation.py`: 登録の繰返し。
- `src/federated_learning_experiments/runtime/adopted_candidate_initial_local_registration.py`: 重複の検査。
- `src/federated_learning_experiments/methods/fedsda/model_registration/pending_model_registration_settings.py`（新規）: 設定型。
- `src/federated_learning_experiments/runtime/fedsda_run_client_settings.py`・`fedsda_run_client.py`・`fedsda_final_configuration_run_settings.py`: 設定の束への追加と、組立て。
- `src/federated_learning_experiments/runtime/feddrift_run_client.py`: 方針の明示。`feddrift_run_server.py`・`feddrift_detection_batch_processing.py`は、sourceを変えない（送信保留を足す操作と、最も古い1つ・残りの待ちを読む操作を使う。1つだけの方針で、結果が変わらないことを、既存のFedDriftのtestと基準版1で確かめる）。
- `tests/refactoring/`: `test_pending_model_upload.py`、`test_held_model_registration_confirmation.py`、`test_server_model_registration_and_aggregation.py`、新規`test_pending_model_registration_by_own_id.py`（全体runの不変条件）、設定のtest、`behavior_golden_cases.py`と`test_important_behavior_regression_goldens.py`、`behavior_regression_goldens/v2/`、`fresh_process_smoke.py`、依存境界、ownerを生成する既存test（方針の明示だけ）。
- `.kiro/specs/pending-model-registration-by-own-id/measure_registration_policy_effect.py`: 影響の計測。
- `docs/research/implementation-findings/legacy-019-renamed-reference-excluded-from-validation-reuse.md`（新規）と一覧、NEW-004の記録（外す箇所の追記）。
- 文書: `docs/experiments/refactoring-baseline.md`（基準版2と、版ごとの設定）、`docs/experiments/experiment-sweep.md`（既定の変更と、既存の結果との関係）、LEGACY-016の記録と一覧、再開案内。

## Requirements Traceability
| 要求 | 設計・証拠 |
|---|---|
| 1 | 確認の「それ自身のID」の分岐、状態の遷移のtest、全体runの不変条件、進行中の候補検証のtest（1.6） |
| 2 | ownerの複数保持、サーバの繰返し、採番の順のtest |
| 3 | 方針の2つの値、設定、FedDriftの明示、実旧との照合（旧と同じ方針） |
| 4 | 旧との照合の設定の明示、基準版1の照合の継続、基準版2と必須の条件、版の差の説明のtest、FedDriftの確認の観測（4.5） |
| 5 | 8条件×2方針の計測（精度・定常精度・通信量・グローバルモデル数・登録されなかった採用の数・送信保留のまま残った数）と記録 |
