# 調査・設計判断

旧BaseClient._update_running_stats/get_model_stats/update_model_statsを調査し、WelfordのPythonfloat演算順と全体/class別所属を分ける。統計更新は監視ごとの損失ではなくFIFO解放や帰属時。初期統計は最後のpretrain shuffle順で損失を逐次追加、新規model seedはtorchfloat32 batch mean/var。seed生成をWelfordへ置換しない。

n0は監視baseline.01、n1は保存meanをclipして監視に利用できるが平均/標本分散推定と候補履歴はn2以上。初回再利用は実mean0も除外する旧挙動、将来候補履歴はn2ならmean0を保持するため、用途別判定は後続へ分ける。ClassESRのclass baselineはclass_statsではなく当該class初回時の全体mean。

調査agentが新規model登録を直接実行し、loss.25一件seedのoverall n1/mean.25/M2.1、class n1/mean.25/M2.0を確認。loss.75追加後overall M2.225。n1の非零M2を修正しない。サーバM2=0の集計も受け取る。IDmap/統計mergeは今回範囲外。

stdlibのfrozen値型/pure関数を採用し、modelregistry・継承・数値依存を増やさない。保存前設計ゲートで12条件coverage/責務と依存/具体ファイル/旧oracle/3単位の実行可能性を主担当が確認した。kiro-spec-requirements/design/tasks/impl/validate-implとfable-methodを適用。既存コードの抽出で外部API調査は不要。

旧class統計説明と実装の不一致、旧モデル統計の不正class入力後の部分更新は共通記録へ接続する。正常実験への影響は断定しない。

正本: [LEGACY-005](../../../docs/research/implementation-findings/legacy-005-class-statistics-documentation-mismatch.md)、[LEGACY-006](../../../docs/research/implementation-findings/legacy-006-partial-model-statistics-on-invalid-class.md)。この移植では旧説明・旧map更新を修正せず、一系列の不変数値契約だけを実装する。

