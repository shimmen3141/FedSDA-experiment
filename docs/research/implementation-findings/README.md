# 実装の不具合・改善事項

旧実装の調査と移植中に見つかった、後から修正・再検討する事項の入口。詳細と現在の状態は個別ファイルが正本。specのresearch.mdにはリンクと、その移植で採用した扱いを残す。

| ID | 件名 | 対象・状態 | 詳細 |
|---|---|---|---|
| LEGACY-001 | 候補採否と区間margin理由の丸め不整合 | 再現済み・未修正、移植では維持 | [記録](legacy-001-candidate-decision-reason-rounding.md) |
| LEGACY-002 | 警報区間が短いと割当済み旧区間もFIFOに残る | 再現済み・未修正 | [記録](legacy-002-retained-assigned-samples-after-short-alarm.md) |
| LEGACY-003 | 実験終端でFedSDA FIFO末尾を確定しない | 改善案・未採用、コード上の保持を確認 | [記録](legacy-003-unassigned-fifo-tail-at-run-end.md) |
| LEGACY-004 | 不正な参照損失入力で旧収集sessionが部分更新 | 再現済み・未修正、正常経路への影響未確認 | [記録](legacy-004-partial-loss-collection-on-invalid-input.md) |
| LEGACY-005 | クラス統計の保持について旧説明と実装が不一致 | 再現済み・未修正、説明の不整合 | [記録](legacy-005-class-statistics-documentation-mismatch.md) |
| LEGACY-006 | 不正class入力で旧モデル全体統計が部分更新 | 再現済み・未修正、正常経路への影響未確認 | [記録](legacy-006-partial-model-statistics-on-invalid-class.md) |
| LEGACY-007 | 空batchでNaN初期統計を登録する | 再現済み・未修正、正常経路への影響未確認 | [記録](legacy-007-empty-batch-nan-initial-statistics.md) |
| LEGACY-008 | 極大件数でサーバ損失平均の範囲超過/除算失敗 | 再現済み・未修正、通常実験への影響未確認 | [記録](legacy-008-extreme-count-server-loss-aggregation.md) |
| LEGACY-009 | 極大parameter値の単純平均が非有限になる | 再現済み・未修正、通常実験への影響未確認 | [記録](legacy-009-extreme-parameter-mean-overflow.md) |
| LEGACY-010 | 空の共有特徴抽出部でoptimizer生成が失敗 | 再現済み・未修正、通常実験への影響未確認 | [記録](legacy-010-empty-shared-feature-optimizer.md) |
| LEGACY-011 | 同一負IDの登録確認で件数倍増・概念件数消失 | 再現済み・未修正、不正通知での再現/通常影響未確認 | [記録](legacy-011-same-id-registration-count-corruption.md) |
| LEGACY-012 | 新規モデル登録の途中失敗で共有部の上書きと採番消費が残る | 再現済み・未修正、正常経路への影響未確認 | [記録](legacy-012-partial-registration-on-invalid-statistics-input.md) |
| LEGACY-013 | 使用済みの一時IDへの再登録が既存モデルを黙って置換する | 再現済み・未修正、正常経路での発生は未確認 | [記録](legacy-013-temporary-id-reregistration-overwrite.md) |
| LEGACY-014 | 採用時の保留標本が割当概念計数と損失統計へ反映されない | 再現済み・未修正、意図した仕様か未確認、oracle診断への影響未確認 | [記録](legacy-014-adopted-model-pending-samples-not-counted.md) |
| LEGACY-015 | 標本吸収の途中失敗で先行標本の更新と不正標本が残る | 再現済み・未修正、正常経路への影響未確認 | [記録](legacy-015-partial-absorption-on-invalid-sample.md) |

## 記録・更新の規約

発見ごとにIDを割り当て、一件一ファイルで以下を残す。

- 発見日、対象コードと基準commit、発見したspec。
- 種別と状態。未再現の懸念と再現済みの事実を分ける。
- 観測した挙動、再現条件・コマンド/テスト、期待する挙動。
- 確認済み/未確認の影響。過去の実験成果・採否・数値・診断への影響。
- 今回の移植で維持するか等の扱い。
- 修正案、担当機能・将来spec、必要な検証・再実験。未定は未定と記す。
- 修正/見送りの理由、commit、検証結果。完了後も記録を削除しない。

状態は「未再現」「再現済み・未修正」「調査中」「修正済み」「見送り」、未採用の改善案は「改善案・未採用」とする。
記録はアルゴリズム変更やgolden更新の承認ではない。旧挙動を維持する移植と、挙動を直す変更を別のcommit/specで検証する。
研究上の代替手法や実験候補は[研究バックログ](../research-backlog.md)とリンクで接続する。
具体的な簡略化・効率化・アルゴリズム調整の仮説は[改善候補](../improvement-candidates/README.md)へ記録する。期待する契約との不整合が疑われる場合は本フォルダ、性能改善の仮説なら改善候補を使い、判断未確定ならその旨を記す。同じ事項は詳細の正本を一つにして相互リンクする。
