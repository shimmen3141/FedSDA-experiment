# 候補分類器と学習状態の生成

正本: requirements.md（振る舞い）、design.md（境界）、naming.md（名前）、tasks.md（進捗）、spec.json（承認）。research.mdは根拠、review.mdは独立レビューと採否、integration-validation.mdは検証証拠。

選択済みの初期parameterから独立した候補分類器と、共有部・概念固有部の新規optimizer管理器を生成する。旧_begin_forward_validationの_new_model→set_params→reset_optimizerまでが範囲。複数epochの学習、early stopping、参照固定、session開始は後続spec。
