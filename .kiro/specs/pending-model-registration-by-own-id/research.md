# Research & Design Decisions

## Summary
- Feature: pending-model-registration-by-own-id（LEGACY-016の修正）。
- 2026-10-13、ユーザーが採用を決定。旧と同じ扱いを設定で残すことも了承。

## Research Log
### 発生の頻度（2026-10-13、`ab10ad4`、読取りだけの計数）
提案構成、client 10、標本2500、集約間隔50、sine2・sea4・circle2・mnist2のseed 0・1（8 run、Windows）。採用48件。正式IDの確認のとき、現行が非負へ戻っていたもの9件（7 run）。送信保留がある間の次の採用（上書き）4件（mnist2 seed 1とsine2 seed 1で2件ずつ）。LEGACY-016の記録は、前者だけを書いていた。

### 旧の処理
`clients/base.py`の`confirm_model_registration`は、`current_model_id`を付け替える。非負なら、`pending_model_params`ほかを外して戻る。`_register_trained_new_model`（322〜324行）は、`pending_model_params`を上書きする。`servers/fedsda.py`の`_register_new_models`は、clientごとに1回。

### 新の現状
送信保留は、モデルIDを持つ（`PendingModelUpload.model_id`）。付替えの操作は、ownerごとにある。配布の反映は、一時IDのモデルを保つ。予測の重みは、IDの集合の変化で均等へ戻る（名前の付替えの操作は無いが、要らない）。FedDriftは、同じownerを使い、`get_pending_model_upload()`と`remaining_upload_delay_round_count`を読む。

### 基準版のgolden
`behavior_golden_cases.py`は、版1だけを前提にしている（定数`BASELINE_VERSION`・`GOLDEN_DIRECTORY`、保存した`run_settings`と現在の設定の完全な一致）。設定の項目を足すと、版1の`run_settings`と一致しなくなるので、版ごとの定義が要る。

## Design Decisions
- 方針をownerに持たせる（保持の仕方と確認の対象を、別々に選べなくする。サーバの関数の引数を増やさない）。
- 旧と同じ方針を残す（ユーザー了承）。旧実装を外すときに、方針の値と分岐も外す（NEW-004の作業へ加える）。
- FedDriftは、1つだけ保持する方針を明示する。FedDriftの手順（作ったモデルが現行で、次の手順で登録）では、2つの方針の結果は同じはずだが、確かめる対象を増やさないため、動作を変えない側を選ぶ。
- 方針の名前は、動作を言う（`legacy_…`としない: FedDriftは、旧互換のためでなく、その手順に合うので使う）。

## Risks & Mitigations
- 既定の方針の正解が実旧に無い → 状態の遷移の直接のtest、現行が送信保留自身の場合の旧と同じ方針との一致、全体runの不変条件、独立レビュー。
- 旧との照合のtestが、既定の方針で実行されて、気づかずに差が出る → 設定の項目に既定値を置かない（組み立てる全箇所が、選ぶ）。
- 基準版2が、意図しない差を含む → 版1と2の差を、欠番・上書きの起きたケースに限ることを、testで確かめる。

## References
- [LEGACY-016の記録](../../../docs/research/implementation-findings/legacy-016-registration-id-consumed-without-model.md)
- 利用skill: kiro-spec-init/requirements/design/tasks、kiro-impl（--review off）、kiro-validate-impl。共通引継ぎ手順に従い、独立CLIの仕様レビューと実装レビューを受ける。
