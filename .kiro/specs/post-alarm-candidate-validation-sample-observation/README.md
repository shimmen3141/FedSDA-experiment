# 警報後の候補検証標本の観測

正本: requirements.md（振る舞い）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。researchは根拠、reviewは採否、integration-validationは証拠。方針はdocs/research/refactoring-policy.md、入口はsteering/resume.md。

警報後にラベルが観測された1標本について、候補モデルと、警報時点で固定した各参照モデルの有界損失を評価し、既存の損失収集へ同じ観測回として追加する。規定件数へ到達したかを返す。旧FedSDAの_observe_forward_validationのうち、損失の評価と追加が範囲。

候補と参照モデルの生成、収集の開始、到達後の採否評価と確定の呼出し、判定record・切替位置・適応イベントの記録、通知、計算量診断、参照も学習させる方針（旧shadow_tournament）の更新、新client/全体runは後続または既存specである。
