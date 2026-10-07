# 警報の変化区間の解決

正本はrequirements.md（要求）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。brief.mdは目的と開発ゲート、research.mdはコード根拠、review.mdは独立レビューと採否、integration-validation.mdは検証証拠。

切出し済みの変化区間の標本に対して、既存の区間評価の結果を適用する組立。適合する保有モデルがあれば学習帰属の切替えまたは維持と標本の吸収、なければ既存の初期値選択と候補検証sessionの開始を行い、結果を不変の記録で返す。

区間の切出し、前区間の処理、最小区間件数、候補検証中の警報、検出器のreset、FIFOの消費、イベント記録、通知、計算量診断、新client・全体runは含めない。旧の保有順による選択を維持し、IMPROVE-001の現行優先を混ぜない。

主担当Claude Code。2026-10-08開始。
