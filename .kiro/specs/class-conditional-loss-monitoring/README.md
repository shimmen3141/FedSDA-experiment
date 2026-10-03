# 全体・正解クラス別の損失監視spec

正本はspec.json（承認と段階）→requirements.md（観測可能な契約）→design.md（責務・API）→naming.md（正式名とrevision）→tasks.md（実装順）で読む。
brief.mdは境界選択、research.mdは旧数値の調査、review.mdは採否と承認、integration-validation.mdは実行証拠を記録する。未生成文書は未完了段階を表す。
既存LossChangeDetectionSettingsのalphaと監視対象を使用する。隣接する予測確率計算・FIFO・モデル統計・候補・client進行は今回の所有範囲に含めない。
