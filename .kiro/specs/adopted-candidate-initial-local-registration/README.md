# 採用候補の初期ローカル登録

正本: requirements.md（振る舞い）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。researchは根拠、reviewは採否、integration-validationは証拠。方針はdocs/research/refactoring-policy.md、入口はsteering/resume.md。

採否判定で採用された学習済み候補を、一時IDの保有モデルとしてクライアント内へ登録する。旧BaseClient._register_trained_new_model（共有部構成の_prepare_model_for_registrationを含む）と、旧FedSDAの送信待機ラウンド設定が範囲。既存の共有反映・損失評価・初期統計・parameter snapshot・学習状態一覧・統計store・送信保留を組み立てる。

一時IDの採番、候補の生成/初期学習/採否、学習量の帰属、保留標本の追加、現在の学習帰属IDの切替えと通知、計算量診断、実送信、新client/全体runは後続である。
