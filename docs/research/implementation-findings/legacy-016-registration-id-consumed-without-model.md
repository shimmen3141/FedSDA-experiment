# LEGACY-016: 採用の後に現行モデルが戻ると、正式IDだけが採番され、採用したモデルは登録されない

- 発見日: 2026-10-10。対象: 旧`federated_drift_experiment/servers/fedsda.py::_register_new_models`（72〜87行）と、`clients/base.py::confirm_model_registration`（427〜462行）、`has_pending_model`（414〜415行）。固定基準748c3aa。
- 発見spec: server-model-registration-and-aggregation。種別: 正常な経路で起きる、状態の食い違い。状態: 再現済み・未修正。意図した仕様かどうかは未確認。

## 再現と観測

2026-10-10、Windows基準環境で、下書きのtest（commitしていない。同じ構成は`tests/refactoring/test_server_model_registration_and_aggregation.py`の対照testにある）を実行した。実旧の事前学習の結果を`SharedBackboneFedSDANoCachedServer`へ登録し、実`__init__`で作った`ResidualAdapterRestartingSoftRoutingFedSDAClient`を3つ登録して、ラウンドごとに、標本処理→保留中の学習→`_register_new_models(t)`→`update_global_models(…)`→`promote_pending_to_ready()`を実行した（配布は行わない）。

- あるclientが、候補を採用して一時ID（-100）のモデルを現行にした。送信までの待ち（2ラウンド）の間に、警報で、現行モデルが元のモデル（ID 0）へ戻った（保有モデルの再利用）。
- 待ちが済んだラウンドで、`has_pending_model()`が真になり、サーバは正式ID（2）を採番して、来歴（モデル2、そのラウンド、そのclient）を記録し、`confirm_model_registration(2)`を呼んだ。
- `confirm_model_registration`は、現行モデルのIDを「付け替える一時ID」として使う。現行が非負（0）なので、送信保留を外しただけで戻った。
- その後、サーバの`next_model_id`は3へ進み、来歴にはモデル2が残るが、グローバルモデルにID 2は現れない。clientの中には、一時ID（-100）のモデルが、学習データと統計とともに残り続ける。集約の対象は非負のIDだけなので、このモデルは集約されない。予測は保有する全モデルを使う（`_record_prediction`が`self.models`の全IDを評価する）ので、clientの予測には参加し続けるはずである（コードの読みによる。この点は、実行で確かめていない）。

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

## 判断（2026-10-13、Claude Codeの提案をユーザーが了承）

採用（最優先）。計測（2026-10-13、新実装`ab10ad4`へ読取りだけの計数を差し込んだ。提案構成、client 10、標本2500、集約間隔50、sine2・sea4・circle2・mnist2のseed 0と1の計8 run、Windows。8 run・1つの規模だけなので、頻度は目安）: 採用48件のうち、現行が非負のIDへ戻っていて欠番になったものが9件（7 runで発生）、送信保留が、待ちの間の次の採用で上書きされたものが4件（2 runで発生）。上書きされたモデルも、登録されない（送信保留の枠が1つしかないため。この記録の「原因」には無かった経路）。修正は、(1)正式IDの確認が、現行ではなく送信保留のモデルIDを付け替える、(2)送信保留を複数持てるようにする、の両方。登録されるモデルが増えるので、モデル数・通信量・精度が変わる。旧との照合を保つため、旧と同じ扱いを設定で選べる形で入れる。

## 修正（2026-10-13）

[pending-model-registration-by-own-id](../../../.kiro/specs/pending-model-registration-by-own-id/tasks.md)で、新実装を修正した（実装`93b902d`、基準版2のgolden`200cd73`）。固定旧748c3aaと旧goldenは、変更していない。

- 送信保留を、採用の順に、全部保持する。正式IDの確認は、待ちの済んだ送信保留のモデル自身の一時IDを、正式IDへ付け替える（現行モデルが別のモデルでも）。サーバは、1つのclientの、待ちの済んだ送信保留の全部を、採用の順に登録する。
- 旧と同じ扱い（送信保留を1つだけ保持し、確認で現行モデルを付け替える）は、設定`PendingModelRegistrationSettings.pending_model_registration_policy`の値`keep_latest_pending_model_and_reassign_current_model_id`で選べる（実旧との照合のtestと、基準版1のgolden、FedDriftが使う。実験のコマンドからは選べない）。既定は、`register_each_pending_model_by_its_own_id`。
- 候補検証の途中で登録が起きると、付け替わったモデルが、その検証の再利用の対象から外れる（[LEGACY-019](legacy-019-renamed-reference-excluded-from-validation-reuse.md)。旧からある扱いで、直していない）。

### 修正の前後の比較

`.kiro/specs/pending-model-registration-by-own-id/measure_registration_policy_effect.py`（source `93b902d`、Windows、提案構成、client 10、標本2500、集約間隔50）。矢印の左が、旧と同じ扱い、右が、既定。

| 条件 | 精度 | 定常精度 | 下りの通信量 | 最終のグローバルモデル数 | runの終わりの保有モデル数（全client） | 登録されないままの採用 |
|---|---|---|---|---|---|---|
| circle2 seed 0 | 0.9534→0.9584 | 0.9664→0.9692 | 5.59→6.34 MB（+13.4%） | 3→4 | 31→40 | 1→0 |
| circle2 seed 1 | 0.9522→0.9600 | 0.9645→0.9699 | 4.72→5.51 MB（+16.6%） | 2→3 | 24→30 | 4→0 |
| mnist2 seed 0 | 0.9130→0.9042 | 0.9168→0.9096 | 2941→3106 MB（+5.6%） | 3→5 | 31→50 | 1→0 |
| mnist2 seed 1 | 0.9149→0.9120 | 0.9189→0.9168 | 2759→3185 MB（+15.4%） | 2→5 | 23→50 | 3→0 |
| sea4 seed 0 | 0.8460→0.8463 | 0.8504→0.8509 | 3.58→3.74 MB（+4.6%） | 1→2 | 11→20 | 1→0 |
| sea4 seed 1 | 0.8434→0.8434 | 0.8475→0.8475 | 3.66→3.66 MB | 2→2 | 20→20 | 0→0 |
| sine2 seed 0 | 0.9730→0.9730 | 0.9817→0.9817 | 5.34→5.34 MB | 2→2 | 20→20 | 0→0 |
| sine2 seed 1 | 0.9715→0.9744 | 0.9820→0.9836 | 4.57→6.66 MB（+45.9%） | 2→4 | 23→40 | 3→0 |

登録されないままの採用が無かった2条件は、同じ結果。精度は、circle2とsine2 seed 1で上がり、mnist2で下がった（−0.3〜−0.9ポイント）。下りの通信量は、最大で+46%。seedが2つなので、傾向は、ablationの規模の再実験で確かめる（未実施）。

