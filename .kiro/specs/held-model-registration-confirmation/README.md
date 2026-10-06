# 保有モデルの正式登録確認

正本: requirements.md（振る舞い）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。researchは根拠、reviewは採否、integration-validationは証拠。方針はdocs/research/refactoring-policy.md、入口はsteering/resume.md。

既存7ownerを使って、保有済み一時モデルへの正式ID通知を処理する。旧BaseClient.confirm_model_registrationの保有済み分岐と現在ID非負分岐が範囲。旧のモデル欠落時のsnapshotからの再構築は、呼出し側が準備する前提とし本処理では行わない。その呼出し側と新モデル初期登録・通信・新client/runの完成は後続である。
