# 警報時点の参照モデルの固定

正本: requirements.md（振る舞い）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。researchは根拠、reviewは採否、integration-validationは証拠。方針はdocs/research/refactoring-policy.md、入口はsteering/resume.md。

警報後の候補検証で比較対象にする参照モデルを、警報時点の値で固定する。保有している各モデルについて、同じ値を持ち保有モデルから独立した分類器を作り、あわせて各モデルの損失統計から参照比較用の履歴平均損失を取り出す。旧FedSDAの_snapshot_reference_modelsと、_begin_forward_validationの履歴平均の取得が範囲。

候補の生成と学習、損失収集の開始、検証標本の観測（完了済み）、採否評価と確定（完了済み）、参照も学習させる方針（旧shadow_tournament、当面不要）、計算量診断、新client/全体runは後続または既存specである。
