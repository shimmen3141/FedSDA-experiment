# 帰属確定標本の吸収

正本: requirements.md（振る舞い）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。researchは根拠、reviewは採否、integration-validationは証拠。方針はdocs/research/refactoring-policy.md、入口はsteering/resume.md。

帰属先が確定した標本列を、保有済みの一つのモデルへ吸収する。標本ごとに、学習標本への追加、真の概念の割当計数、そのモデルでの有界損失の評価、全体・正解クラス別の損失統計の更新を行う。旧BaseClient._absorb_into_storeが範囲。

帰属先の決定、保留標本の収集と解放、現在の学習帰属IDの切替え、採用時の標本追加（統計を更新しない別経路。adopted-candidate-local-adoption）、FIFOから1件ずつ確定する経路の組立、計算量診断、評価標本の保存、学習の実行、新client/全体runは後続または既存specである。
