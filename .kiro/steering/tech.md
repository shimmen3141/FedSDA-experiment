# 技術と検証

## 現状

既存実装はPython・NumPy・PyTorch。新srcのパッケージ境界とpytestのimport構成を作成済み。
Windows CPUの固定環境を別venvから再構築し、113テスト成功を確認済み。
版とビルド情報は`environments/golden/windows-cpu/`を参照する。

## 原則

- 不変の型付き設定を渡し、実行中にグローバル設定を変更しない。
- 依存を組み立てるruntimeと、機能内の計算・判断・状態を分離する。
- 明確な単位・ID・時系列を使い、実際の差し替え境界だけを抽象化する。
- PyTorch依存はモデル・学習の責務として明示する。

## 検証

新実装のLint・整形はRuff、型検査はPyright。設定の正本は`pyproject.toml`。
導入方針・範囲・実行手順は`docs/research/code-quality.md`を参照する。

最終3ケースと移植対象baselineについて、旧goldenの数値・イベント列を比較する。
旧APIの受理で対応せず、テスト内で新しい記録と対応付ける。
乱数消費、反復順、optimizer更新、共有参照、ID採番を維持する。
環境差でgoldenを上書きしない。

Windows基準の手順は`docs/experiments/refactoring-baseline.md`。
長時間研究実験の実行環境はLinuxであり、開発用golden環境と区別する。

## 研究実験の実行環境とコマンド（2026-10-24更新）

2026-10-24のユーザー訂正を反映した現行の案内。実行環境を明示して並列数を決める。

| 環境 | 並列数の目安 | 用途 |
| --- | --- | --- |
| uranium（研究室サーバ、256コア） | **64 workersを標準**として明示する | 本規模の研究実験 |
| 手元のWindows（16コア） | 従来の14 workers以内という目安 | 短い検証・予備実験。uraniumと混同しない |

- uraniumに「全セッション合計14」「2セッションなら各7」という古い上限を適用しない。64は標準値で、256コア全使用や256 workersへの自動引上げを意味しない。複数実験を並行するときは、各実験のworkersと合計、メモリ・既存負荷を見て配分する。
- 数値ライブラリは各worker 1 thread（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`）。入れ子の並列化は追加しない。
- 長時間実験はLinuxの`.venv`を有効化してtmux内で実行する。サーバの実行・転送はユーザー担当。1時間以上かかる見込みなら進行確認は55分おきとし、完了が分かれば追加確認は不要。
- 研究用CLIは既存の`python -m tools.research_provisional_candidate_expert.run`と同じく、`--output`、実験条件、`--workers 64`を明示する形式を基本にする。`tools/run_server_sweep.sh`は旧Pareto掃引用であり、全ての研究CLIに使うという規則ではない。
- 新しい検証・実験ツールは、必要な条件をCLIから指定でき、出力・失敗・ソース来歴を追える形にする。専用shellは補助として使い、workersを古い手元環境の値へ固定しない。検証・研究ツール段階のspecは不要、本実装が決まったらcc-sddというユーザー指示を維持する。
- 実験前に内部で`--print-plan`等により条件・出力先・重複を確認する。確認だけで結果ディレクトリを作らない。結果・rawはGitへ含めず、関連ツールと実行手順を共有する。

### 旧Pareto用wrapperを使う場合

`tools/run_server_sweep.sh`は`run_pareto_sweep.py`向けの補助であり、新しい研究CLIへ一律に適用しない。

- `FDE_WORKERS`、一意なvariant名、必要なら`FDE_RUN_DIR`を指定する。ログ・GNU time・Pareto・rawの保存先はwrapperが構成するため、同じ出力指定を下位CLIへ重複して渡さない。
- 第1引数variantは実験名と既定tagを兼ねる。`FDE_RUN_DIR`を省略すると`results/results_<日時>_<variant>/`、明示すると`<FDE_RUN_DIR>/<variant>/`へ保存する。
- `residual-pcgrad-a50`のように比較対象・主要オプション・重要な固定値を短く判別できる名前にする。`--tag`は直接渡さず、出力ファイル名だけ変える場合は`FDE_TAG`を使う。
- 直接CLIもwrapperも、並列セッション間で出力先・rawを共有しない。比較は同じ実験規模・seed・データセット・固定値で記述する。

### 古い記述の所在と扱い

次は2026-10-24に確認した所在と整理結果。過去の実測のworker数や承認記録は書き換えない。ユーザーの意図はAGENTS.mdの変更禁止ではなく、環境・コマンドの知識をsteeringへ集約すること。本worktreeのAGENTS.mdは本節と共通引継ぎ手順への参照に置き換えた。

| 場所 | 古い記述・今回の扱い |
| --- | --- |
| 本worktreeの[AGENTS.md](../../AGENTS.md)「実験実行環境とコマンド」 | worker数とコマンドの詳細を本節へ集約し、参照だけにした |
| 元checkoutの`AGENTS.md`（main側） | 別checkoutには16物理コア・合計14・各7・旧wrapper必須の記述が残る。本作業はrefactoring側で整理しており、main側はマージ時の反映対象。uraniumの最新条件として使わない |
| [experiment-sweep.md](../../docs/experiments/experiment-sweep.md) | 研究室サーバ16コア・合計14という説明を本節への参照へ訂正。個別の14 workerコマンド例は標準値を表すものではない |
| [candidate-validation-waiting-diagnostic.md](../../docs/experiments/candidate-validation-waiting-diagnostic.md)、[prediction-initial-mass-diagnostic.md](../../docs/experiments/prediction-initial-mass-diagnostic.md) | Linux一律14/各7という説明を訂正。過去の試行条件は維持 |
| `tools/run_main_ablation_suite.sh` | 既定14は旧16コア向け。helpに区別を明記。uraniumでは`FDE_WORKERS=64`を明示する |
| [S1対照の記録](../../docs/research/fedsda-shadow-learner-s1-controls-20261024.md)、`tools/research_shadow_learner_diagnostic/run_controls.sh` | 今回誤って案内した14を訂正。直接CLIは`--workers 64`、uranium用補助shellの既定も64 |
| [research-handoff-20261023.md](../../docs/research/research-handoff-20261023.md) | 「手元Windows・16コア・14以内」は手元の条件として有効。進行確認の旧57分はユーザーの最新指定55分に読み替える |
