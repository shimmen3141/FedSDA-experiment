# 調査と境界判断

- 旧clients/fedsda.pyの_observe_forward_validationはactiveなしno-op、候補→参照評価→規定件数→_finalize_forward_validation。
- 確定は判定record追加→採否適用→切替/episode→適応event→active解除→_on_drift_resolution。新resolutionの再利用は吸収→帰属変更であり、旧の切替/hook→吸収との順序差は既承認。今回は成功時の最終状態と記録情報を比較し、通知は返却後の呼出側へ委譲する。
- 既存start record、observe関数、collection公開snapshot、evaluate関数、apply resolutionを再利用。private列・数値式・採番/登録/吸収を複製しない。新ライブラリなし。
- ProvisionalModelDecisionの14fieldsと4propertiesは、metadata＋既存不変PostAlarmCandidateLossEvaluationから導出できる。forward_sourceは新recordの型名が表し、旧文字列aliasを追加しない。
- 終端不足回収は通常評価の最低件数契約に入らず、別specとする。shadow_tournamentは既決定どおり移植しない。
- read-only調査担当/root/session_progress_researchの既存APIと順序の調査を採用。新しい旧不具合は実測していない。LEGACY-014は維持。
- 要求生成のレビューgateでID/EARS/範囲/異常系を自己確認した。次に実Luna独立承認。設計では薄い進行関数と不変返却recordに絞り、active ownerや汎用callback abstractionを増やさない。
