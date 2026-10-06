# Development finding: Ruffのルート探索がアクセス不能な一時ディレクトリで失敗する

- 観測日: 2026-10-07
- 観測した作業: current-training-model-assignment Task3品質検査
- 改善先: project（検査手順）、原因の改善先は不明
- 関連commit・artifact: a4959fb、[対象specの実測](../.kiro/specs/current-training-model-assignment/integration-validation.md)

## 観測した事実

基準Windows/共有venv、sandbox実行で `python -m ruff check . --output-format concise` は探索のアクセス拒否warningを2件出してAll checks passedと表示した。続く `python -m ruff format --check . --output-format concise` は `Expected a ruff source file` というpanicを出した。両方が通常成功することを期待していた。

同じセッションのgit statusも `.pytest_cache/` と `pytest-cache-files-*` のアクセス拒否を表示した。root探索とpanicの因果関係、Ruff内部の原因は確定していない。直前にsandboxのpytest REDで一時cacheの権限warningを観測した。

## 影響とworkaround

- 影響: rootからの品質検査を成功と扱えない。複数コマンドの最後がpip checkだとシェル全体のexit0だけでは途中のpanicを見落とせる。
- その場のworkaround: 対象を `src tests/refactoring` と明示し、lintとformatを別々に実行。共有pyprojectの適用対象をすべて含む。同じ設定でlint成功、format125 files整形済み、各exit0。

## 仮説と改善案

- 仮説: アクセス不能な一時ディレクトリの探索がRuffの診断経路を通りpanicに至る可能性。実装内部を調査していないため断定しない。
- 改善案: このWindows環境で探索拒否を観測したら対象ディレクトリを明示し、各検査のexitと出力を確認する。共有設定の対象を増やすときには明示コマンドも更新する。

## 改善結果

docs/research/code-quality.mdへ回避手順を追加。対象ディレクトリを明示した再実行でlint/formatそれぞれexit0。Ruff自体やファイルACLは変更していない。
