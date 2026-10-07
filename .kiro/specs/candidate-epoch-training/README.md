# 候補のエポック学習

正本: requirements.md（振る舞い）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。research.mdは調査、review.mdは独立レビューと採否、integration-validation.mdは検証証拠。

生成済みの独立候補を警報区間の標本で学習する。固定エポック・検証損失による早期停止・学習省略を扱う。候補生成・初期値選択・参照固定・検証session開始は含めない。
