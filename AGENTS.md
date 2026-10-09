# リファクタリングworktreeの規約

- 2026-10-08のユーザー指示により、担当ツールを問わず日常的・分量の多いレビューはGPT-6 Luna、
  やや複雑・難易度の高い実装のレビューはClaude Haiku 5.5を優先する。分量より難易度を優先して選ぶ。
  優先モデルを利用できない場合は他方で独立レビューし、有用な指摘の反映で承認する。
  effortはHaiku・Lunaとも`medium`に明示指定する（2026-10-08、Lunaはhighで指摘の質が変わる証拠がなくmediumへ戻した。2026-10-09のユーザー指示でHaikuの既定も`high`から`medium`へ改めた）。代替で使うときも各モデルのこの指定とし、CLI既定値に依存しない。
  要求・設計・命名・tasks・実装task・別feature最終GOの各ゲートを維持し、自己レビューで代替しない。
  優先モデルを選んだ理由・実際のモデル/effort・代替時の利用不能理由・対象revision/hash・指摘の採否を共通specへ記録する。
  詳細は[共通引継ぎ手順](.kiro/steering/agent-handoff.md)を参照する。過去の承認記録は書き換えない。

## Codexからのレビュー起動

レビュー依頼は対象・revision/hash・確認事項を含み、独立した読取り専用レビューで再委任しないことを明記する。
PowerShellでは依頼を`$reviewPrompt`へ格納し、次のように起動する。詳しい検証・記録手順は共通引継ぎ手順を参照する。
対象に応じた優先モデルのコマンドを一つだけ実行し、利用不能時だけ他方へ切り替える。

```powershell
claude -p $reviewPrompt --model claude-haiku-5-5 --effort medium --output-format json --no-session-persistence --tools Read,Glob,Grep --permission-mode plan
codex exec -m gpt-6-luna -c 'model_reasoning_effort="medium"' --sandbox read-only --ephemeral $reviewPrompt
```

HaikuはJSONの`modelUsage`で実モデルと`is_error=false`を確認し、Lunaは実行ログのモデル名と`reasoning effort: medium`を確認する。
呼出し成功だけでレビュー承認とせず、対象についての判定・指摘とその採否を残す。

## worktreeの作業方針

- 本worktreeは`refactor/architecture`ブランチ。旧実装の固定基準は`748c3aa`。
- 方針の正本は[リファクタリング方針](docs/research/refactoring-policy.md)、進捗・変更仕様は`.kiro/specs/`で管理する。
- セッション開始・再開時と実装単位の開始前に、対象specの`README.md`で正本一覧を確認し、方針の正本、`.kiro/steering/roadmap.md`、対象specの
  `spec.json`・`review.md`・`naming.md`を読み、現在の合意と未承認事項を確認する。
- 名前は短さより、意味の明確さ・他の概念と混同しないこと・実態との一致を優先する。
  必要なら4語・5語以上を用いる。承認済みのルートパッケージ名は`federated_learning_experiments`。
- cc-sdd作業では`.kiro/steering/product.md`・`tech.md`・`structure.md`と対象specを読み、
  `.kiro/settings/rules/naming-review.md`に従う。利用するskillの`SKILL.md`も読んでから進める。
- 新実装は後方互換alias・旧形式読込み・旧import窓口を持たない。
- 新実装の設定・選択肢は機能別の型と宣言から管理し、実体生成をruntimeへ分離する。
  以下の旧実装向けファイル名・登録先の規約は、移植前の既存コードを変更する場合にだけ適用する。
- 各実装単位の開始前に、`naming.md`へファイル・型・関数・引数・状態変数の案と役割を列挙する。
  入出力、単位、状態更新、似た名前との違いを説明する。
  命名承認は上記のモデル選択方針に従う独立レビューと主担当による有用な指摘の反映で行う。
  全指摘の採用は必須ではない。採否と理由を記録し、命名だけの追加承認を人間へ求めず実装を続ける。
- 命名承認は`spec.json`の`approvals.naming.approved`と`approved_revision`へ記録する。
  承認したrevision以外の実装は開始しない。名前・役割を変えた場合は同じレビュー手順で再確認する。
- cc-sddの`-y`・`--auto`、以前の「開始して」という依頼、時間経過を命名承認として扱わない。
- 要求・設計・task・実装の承認も、上記のモデル選択方針に従う独立レビューと主担当による有用な指摘の反映で行う。
  不要と判断した指摘は採否と理由を残す。人間の承認を毎段階で再要求せず、承認対象・内容hash・レビュー証拠を対象specへ記録する。
  現在の承認状態は対象specの`spec.json`・`review.md`から確認する。
- 新しいsrc実装、移植、リネーム、実装を先取りしたテスト追加は、命名承認前には行わない。
  例外（2026-10-09ユーザー決定）: 命名表を作るために、source・testをリポジトリ外で下書きし、作業ツリーの複製（`git archive`）で実行して、名前と実行可能性を確かめてよい。下書きはworktreeへ置かず、worktreeのtestも実行しない。複製での結果は承認の証拠に使わない（判定は、承認の後にworktreeへ適用した実装の実測とレビューによる）。下書きを作った事実はresearch.mdへ書く。
  worktree作成、ツール導入、調査・仕様・命名案の作成は今回承認済み。
- 移植対象の数値・判断・時系列をgoldenと照合する。新APIとの対応付けはテスト側で明示する。
- 基準worktreeには3つのコミット保留資料とresultsがある。本worktreeへ自動コピー・stageしない。
- コミット・プッシュは2026-10-03にユーザーが許可済み。検証済みで切り戻しやすい責務の単位で行い、対象ファイルを明示してstageする。

## 既存実装向けの開発規約

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
- `docs/reference/options.md`は直接編集せず、`python -m tools.generate_option_docs`で再生成する。
- 変更後はスキーマの整合性テスト、対象機能テスト、`tests/test_regression.py`を実行し、既存手法の値を変えていないことを確認する。
- 最終Residual Adapter＋Switching構成の変更・リファクタリングでは、`tests/test_proposed_regression.py`も実行する。
  基準環境と手順は`docs/experiments/refactoring-baseline.md`に従い、環境差だけを理由にgoldenを更新しない。

## 文書とコミット保留

- 文書の入口は`docs/README.md`。全体説明は`overview/`、個別機能は`components/`、
  実験資料は`experiments/`、設定参照は`reference/`、研究検討資料は`research/`へ置く。
- 2026-10-02のユーザー指示により、以下の3ファイルはgit管理外のままコミット判断を保留している。
  移動したことを理由に自動でstage・commitしない。後続のユーザー指示で扱いを決める。
  - `docs/experiments/experiment-results-audit.md`
  - `docs/overview/fedsda-processing-flow.html`
  - `docs/research/FedSDA Meta-switchingの先行研究・差分・新規性に関する調査報告.pdf`

## 実験実行環境とコマンド

- 長時間実験は、Linux上の`.venv`を`source .venv/bin/activate`してからtmux内で実行する。
- 実行環境は16物理コア・SMTなし・十分なメモリを持つ。OpenMP/MKLは各worker 1スレッドに保つ。
- 1実験なら最大14 workers、2つのtmuxセッションで並行するなら各7 workersを目安とし、
  全セッションの`workers`合計を原則14以下にする。入れ子の並列化は追加しない。
- 長時間コマンドは`tools/run_server_sweep.sh`を使い、`FDE_WORKERS`、一意なラベル、必要なら
  `FDE_RUN_DIR`を指定する。ラッパーがログ・GNU time・Pareto・rawの保存先を構成するため、
  同じ出力指定を`run_pareto_sweep.py`へ重複して渡さない。
- `run_server_sweep.sh`の第1引数`variant`は実験名と既定の`tag`を兼ねる。`FDE_RUN_DIR`を
  省略すると`results/results_<日時>_<variant>/`を作り、明示した場合だけ共有ルート下の
  サブディレクトリ`<FDE_RUN_DIR>/<variant>/`へ保存する。
  `residual-pcgrad-a50`のように、比較対象・主要オプション・重要な固定値を短く判別できる名前にする。
  `--tag`はラッパーが管理するため直接渡さず、出力ファイル名だけ別にしたい場合は`FDE_TAG`を使う。
- ラッパーを使わないコマンドには`--workers`を明示し、並列tmux間で`--out-dir`と`--raw-dir`を
  共有しない。比較対象は同じ実験規模・seed・データセット・固定値で記述する。

## 実験成果物と重複確認

- 長い条件を成果物ファイル名へ埋め込まない。`run_pareto_sweep.py`が短い内容ハッシュ名を生成し、
  完全な条件はCSV・NPZ・`manifest.json`へ保存する。
- エージェントは長時間実験のコマンドを作る前に、内部確認として`--print-plan`でrun構成と
  重複先manifestを確認する。`--print-plan`は人間向けの実行手順・提示コマンドには含めない。
  計画確認は読み取り専用で、結果・raw・ログ用ディレクトリを作成してはならない。
  通常実行では既定の`--duplicate-policy error`を維持し、
  一部でも同一コード・golden・run設定の既存結果があれば、表示されたmanifestを確認して計画を直す。
  意図的な再実験だけ`warn`、照合不要と判断できる場合だけ`ignore`を使う。
- 既存CSVからmanifestを補完するときは
  `python -m tools.experiments.manifests backfill <results-root> --recursive`を使う。
- CSV・Pareto図を失いNPZが残る場合は、先に
  `python -m tools.experiments.artifacts <result-root> --tag <short-tag>`で復元する。
  `.reconstruction.json`が`quality=partial`の旧結果はPareto確認には使えるが、baselineへ採用しない。
- CSVが残る場合の再描画には`run_pareto_sweep.py --plot-csvs ...`を使い、実験を再実行しない。
- クラスタリングで統合されたモデル対と残されたモデル対の予測相補性を比較するときは、
  `python -m tools.experiments.clustering_functional_diagnostics <result-root> --output <csv>`を使う。
  この診断には`cross_evaluation_*`と`clustering_pair_*`を含む新しいrawが必要である。
