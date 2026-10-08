# Development finding: 小さなspecでの機械照合とレビュー読取りの重複

- 観測日: 2026-10-08
- 観測した作業: alarm-adaptation-recording（source/test 9b72182、完了4a7fc53）
- 改善先: projectの共通手順・検査script、主担当の依頼方法
- 証拠: 対象specのreview.md/integration-validation.md、元checkoutのvenv/refactoring-tests/alarm-adaptation-recording-*-review.log

## 観測した事実

少量の記録実装に対し、実装後のLunaレビュー3回のCLI表示tokens usedは69,694・68,269・44,812だった。セッション全体の消費量や課金を示す値ではない。広い仕様・上流・巨大なguardを繰り返し読ませた。source hashは271ファイルごとにgit showを起動し、Task3の独立再計算が時間内に完了しなかった。最終担当は一括読取りで同一hashを確認した。

全pytestの初回は一時ファイル権限で29件失敗し、同じコードを必要な権限で再実行して成功した。不正入力で更新しない検証と上流/RNG不変の契約は妥当だが、それを理由に資料・検証を一律に拡大する必要はない。

## 影響とworkaround

待ち時間・繰返しの読取りが増えた。最終レビューでは対象を絞り、source hashを一括計算した。承認ゲート・固定旧基準は維持した。

## 仮説と改善案

読取り範囲の指定不足と、同じ機械検査の再実行、Gitプロセスの大量起動が負担を増やした。寄与率は測定していない。差分と契約を入口にし、再実行は対象変更・独立性・残るリスクから判断する。

## 改善結果

共通引継ぎ手順へ5項目の条件付き指針を追記。spec_checksのsource hashはgit cat-file --batchでblobを一括取得し、従来のパス順・LF・NUL区切りを維持する。

専用一時repoで空集合、日本語/空白path、CRLF、重複blob、symlink、export-ignore、golden2種、指定revisionと変更検出が旧git show定義と一致。既知9b72182の271パス/hashも一致し、このPCの一回の測定は1.008秒だった。Ruff/format成功。証拠は元checkout/venv/refactoring-tests/spec-checks-batch-verification.json。実験src/tests/goldenは変更せず、全pytestは今回再実行していない。

Luna mediumの独立レビュー（session 01a11ae1-6537-7611-a8a0-deecc8a1f078、spec-checks-batch-review.log）はAPPROVED。独立identity実行で既知source hash・4承認hash・固定旧diffを確認。全パスの旧方式との比較・Ruff・全pytestは独立再実行していない。ユーザーの指示で対象を限定した保守変更として行い、新しいspec文書群は追加しなかった。
