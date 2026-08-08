# エージェント向け開発規約

- コメントとプロジェクト固有文書は日本語で書く。
- 手法を追加するときは、`mode_names.py`と`experiment.py`の`MODE_SPECS`だけでなく、
  `option_schema.py`の手法能力・実装範囲・選択肢固有制約も同時に更新する。
- オプションを追加するときは、依存条件を散在する条件分岐だけで表現せず、まず`OptionSpec`、
  `ActivationRule`または`ChoiceConstraint`へ登録する。数値パラメータなら`parameter_schema.py`にも登録する。
- 掃引値を空にしたとき無効になる固定値は`SWEEP_DEPENDENCIES`へ登録する。
- 指標を追加するときは`metric_schema.py`へ用途・適用範囲・保存先を登録する。
- `docs/options.md`は直接編集せず、`python -m tools.generate_option_docs`で再生成する。
- 変更後はスキーマの整合性テスト、対象機能テスト、`tests/test_regression.py`を実行し、既存手法の値を変えていないことを確認する。
