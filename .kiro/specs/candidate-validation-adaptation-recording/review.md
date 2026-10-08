# 独立レビューと検証記録

主担当はClaude Code。レビュー担当の選択は共通引継ぎ手順に従う。

## 要求r1・設計r2・命名r1・tasks r1 — APPROVED

- 選択: 文書と命名の通常のレビューなのでGPT-6 Luna。`codex exec -m gpt-6-luna -c 'model_reasoning_effort="medium"' --sandbox read-only --ephemeral`で起動し、ログのmodel行`gpt-6-luna`と`reasoning effort: medium`を確認した。代替は行っていない。依頼文には設計レビューの確認項目（検査がすべて最初の状態更新より前にあるか、上流が検査しないfieldへの手組みの値、要求との1文ずつの照合）を入れた。
- 1回目（session `01a11af3-b45a-7993-af2b-5d8881716780`、4文書のr1）: 要求APPROVED、設計REJECTED、命名APPROVED、tasks APPROVED。旧の2メソッドの事実（actionの対応、切替位置の条件、終端位置、再利用件数を加算しないこと）、既存の確定の4結果で保留標本の帰属先が確定後の学習帰属IDと一致すること、field改名の影響範囲は問題なしと報告。
  - 設計Major: 棄却（または維持）と、前後のIDが同じ変更記録を組み合わせ、完了情報の変更前IDと帰属先IDも同じにすると、IDが異なるかの検査を通過して記録が追加される。→ 事実と確認して採用。検査「変更記録の変更後IDが変更前IDと異なる」を追加し、結果種別ごとに受理する組合せの表を設計へ加えた（設計r2）。下書きのsourceと拒否条件のtestへ反映した。
  - 設計・任意: 結果種別ごとの整合規則を表で示す。→ 採用（上の表）。
- 2回目（別session `01a11af5-95d3-7ad2-8893-7a4adcdc3d6c`、設計r2）: APPROVED、指摘なし。受理表以外の組合せが更新前に拒否されること、追加した検査が既存の確定の正常な4結果を拒否しないこと、要求r1・tasks r1との矛盾がないことを確認したと報告。
- 手順上の事実: 命名の事前登録のため、sourceとtestをリポジトリ外で下書きし、作業ツリーの一時複製へ適用して実行した（worktreeは変更していない。設計r2の反映後、対象103件が成功）。`spec_checks.py names`は、下書きを適用した複製で未登録・役割の再利用とも報告なし。承認の時点でworktreeに新src・新testはない。
- 外部証拠: 元checkoutの`venv/refactoring-tests/candidate-validation-adaptation-recording-spec-review-r1.md`/`.log`、`-r2.md`/`.log`。
