# 一時モデルIDの採番

正本: requirements.md（振る舞い）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。researchは根拠、reviewは採否、integration-validationは証拠。方針はdocs/research/refactoring-policy.md、入口はsteering/resume.md。

クライアントが新規モデルへ付ける、サーバ確認前の負の一時IDを採番する。旧BaseClientのnext_temp_id初期化と_alloc_temp_idが範囲。採番したIDでの登録・現在ID切替え・正式IDへの付替え、サーバ側のID対応、新client/全体runは後続または既存specである。
