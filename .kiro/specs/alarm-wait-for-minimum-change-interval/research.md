# Research & Design Decisions

## Summary
- Feature: alarm-wait-for-minimum-change-interval（IMPROVE-006の「待つ案」）。2026-10-13、ユーザーが、案（待つ案）と進め方（待ち時間のうちに、設定で選べる形で実装し、サーバが空いたら比べる。既定は変えない）を了承。

## Research Log
- 連鎖の仕組みの観測と、2つの案（待つ／証拠を保つ）の試行の結果は、[IMPROVE-006の記録](../../../docs/research/improvement-candidates/improve-006-keep-detection-evidence-after-too-short-alarm.md)の「連鎖の仕組みの観測と、2つの案の試行（2026-10-13）」。
- 手法の文書（`docs/overview/fedsda-algorithm.md`）: 検知したら、保留を新旧に分け、旧側を現行モデルへ渡し、新しい概念のデータで、再利用か新規作成かを決める（ResolveDrift）。「データが足りないので何もしない」という分岐は、手法には無い。最小件数は、解決に要る新概念のデータの最小数（旧`config.MIN_DRIFT_DATA`の注記）。元の設計（ADWIN）は、検知の後、窓の旧側だけを削り、新側のデータを検出器に残す。
- コード: 警報かどうかは、`process_observed_sample`の1か所（`alarm_raised`）で決まる。警報の処理が変化区間とする件数は、`min(保留の件数, 推定の長さ)`（`PendingTrainingAssignmentBuffer.get_change_interval_partition`）。検出器の再開は、警報の完了処理だけが行う。

## Design Decisions
- 待つ案を選ぶ理由（ユーザーへ説明済み）: 検知1回に解決1回、という手法の流れを保つ。保留の意図（解決できるまで、データをストアへ入れない）に合う。警報の条件を厳しくするだけなので、検出器の誤警報の制御の解釈を崩さない。証拠を保つ案は、同じ証拠で警報が複数回出て、解決していないのに、旧側の取込みと記録だけを行う。
- 条件を、標本1件の処理に置く（検出器の中でなく）。検出器は、最小件数（データの帰属の条件）を知らない。警報の処理が変化区間とする件数と、同じ式で判定できる。
- 設定は、検出の設定（`LossChangeDetectionSettings`）に置く（「いつ警報とするか」の方針）。
- 既定は変えない。基準版も足さない（採用が決まったら、別のspecで、既定を替えて、基準版を足す）。

## Risks & Mitigations
- いまの扱いの動作を、気づかずに変える → いまの扱いの、実旧との対照test（全体run・構成・部品）と、基準版1〜3が、値を変えずに通ること。
- 待つ間に、検出器の値がしきい値を下回って、変化を取り逃がす → 設計上の動作（一時的な跳ねを、変化として扱わない）。影響は、サーバでの比較で見る（検出の再現率、切替までの遅れ）。
- 待つ扱いの正解が、実旧に無い → 「合図が無かった場合」と「いまの扱いで処理した場合」という、既存の動作との一致で確かめる。

## References
- [IMPROVE-006の記録](../../../docs/research/improvement-candidates/improve-006-keep-detection-evidence-after-too-short-alarm.md)、[LEGACY-002の修正のspec](../too-short-alarm-earlier-sample-release/design.md)
- 利用skill: kiro-spec-init/requirements/design/tasks、kiro-impl（--review off）、kiro-validate-impl。共通引継ぎ手順に従い、独立CLIの仕様レビューと実装レビューを受ける。
