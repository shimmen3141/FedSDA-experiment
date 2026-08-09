# エージェント向け開発規約

- コメントとプロジェクト固有文書は日本語で書く。
- 手法を追加するときは、`mode_names.py`と`experiment.py`の`MODE_SPECS`だけでなく、
  `experiment_spec/options.py`の手法能力・実装範囲・選択肢固有制約も同時に更新する。
- オプションを追加するときは、依存条件を散在する条件分岐だけで表現せず、まず`OptionSpec`、
  `ActivationRule`または`ChoiceConstraint`へ登録する。数値パラメータなら`experiment_spec/parameters.py`にも登録する。
- 掃引値を空にしたとき無効になる固定値は`SWEEP_DEPENDENCIES`へ登録する。
- 新しい掃引軸は`experiment_spec/sweep.py`の`SweepAxis`として追加し、固定側パラメータはその軸の
  `fixed_values`へ所属させる。実行関数へ新しい並列リスト引数を増やさない。
- 一つのrunで変わる値は`ExperimentConfiguration`へ含め、実行中だけ`activated()`で有効化する。
  実行スクリプトから`config`を手動で保存・復元する処理を追加しない。
- 指標を追加するときは`experiment_spec/metrics.py`へ用途・適用範囲・保存先を登録する。
- `docs/options.md`は直接編集せず、`python -m tools.generate_option_docs`で再生成する。
- 変更後はスキーマの整合性テスト、対象機能テスト、`tests/test_regression.py`を実行し、既存手法の値を変えていないことを確認する。

## 実験実行環境とコマンド

- 長時間実験は、Linux上の`.venv`を`source .venv/bin/activate`してからtmux内で実行する。
- 実行環境は16物理コア・SMTなし・十分なメモリを持つ。OpenMP/MKLは各worker 1スレッドに保つ。
- 1実験なら最大14 workers、2つのtmuxセッションで並行するなら各7 workersを目安とし、
  全セッションの`workers`合計を原則14以下にする。入れ子の並列化は追加しない。
- 長時間コマンドは`tools/run_server_sweep.sh`を使い、`FDE_WORKERS`、一意なラベル、必要なら
  `FDE_RUN_DIR`を指定する。ラッパーがログ・GNU time・Pareto・rawの保存先を構成するため、
  同じ出力指定を`run_pareto_sweep.py`へ重複して渡さない。
- ラッパーを使わないコマンドには`--workers`を明示し、並列tmux間で`--out-dir`と`--raw-dir`を
  共有しない。比較対象は同じ実験規模・seed・データセット・固定値で記述する。
