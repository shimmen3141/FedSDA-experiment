# リファクタリング実装の品質検査

## 正本と対象

2026-10-05に導入した開発基盤。実験アルゴリズムや新しい公開APIを追加する作業ではないため、
SDDの新specは作成せず、共有設定と本書に方針・検証を記録する。
機能の要件・設計・命名は引き続き`.kiro/specs/`の各正本に従う。

- Lint・整形の正本: `pyproject.toml`のRuff設定。
- 型検査の正本: 同ファイルのPyright設定。
- ツールの版: `requirements-dev.txt`。Ruff 0.16.10、Pyright 1.1.414、pre-commit 4.6.2。
- CI: `.github/workflows/python-quality.yml`。
- コミット時の検査: `.pre-commit-config.yaml`。

Ruffは`src/`と`tests/refactoring/`、Pyrightは`src/federated_learning_experiments/`を対象とする。
不変の旧参照実装と旧goldenは対象外。旧実装の整形差分を混ぜない。

## 検査方針

Ruffは基本的な誤り、未使用名、import順序、BugBear、Python版に応じた記法を検査する。
`B905`だけは除外する。`zip(strict=True)`の追加は反復の振る舞いを変えるため、
長さの契約は既存の入力検証とテストで確認する。

Pyrightは全体を`standard`とし、設定・宣言、損失統計、学習データ割当を`strict`とする。
対象ディレクトリの正確な一覧は設定を参照する。
型注釈は実行時検証の代わりにはならない。Tensorの形状、ID、単位、非空条件、数値範囲は
既存の検証とpytestで確認する。

コンテナには具体的な要素型を記述する。実行時検証済みの値は必要な箇所だけ`cast`で伝える。
型検査器が表現できない既存の検証・非空条件については、理由を添えた行単位のignoreを使う。
ファイル全体のignoreや`Any`の追加で問題を隠さない。
`sum`の初期値・演算順、乱数消費、runtimeの検証順序を型検査のために変えない。

pytest fixtureの登録用importは`from ... import fixture as fixture`と明示して維持する。
未使用importの自動削除でfixture登録を失わないようにする。

## ローカル手順

基準環境の再構築は[Windows CPU golden環境](../../environments/golden/windows-cpu/README.md)に従う。
そのvenvを選択・有効化して、リファクタリングworktreeのルートで実行する。

```console
python -m pip install -r requirements-dev.txt
python -m pip check
python -m ruff check . --output-format concise
python -m ruff format --check . --output-format concise
python -m pyright
python -m pytest tests -q
```

型検査が別のPythonを参照する場合は`--pythonpath <選択したvenvのpython実行ファイル>`を指定する。
このPCの共有venvは元checkoutの`venv/`にあり、worktreeの直下には存在しない。
機械固有の絶対パスは共有設定へ書かない。

修正は明示的に行う。自動修正後のdiffを確認し、影響するテストを実行する。

```console
python -m ruff check . --fix
python -m ruff format .
```

commit hookを導入するには次を実行する。

```console
python -m pre_commit install --allow-missing-config
python -m pre_commit run --all-files
```

Gitのhookはworktree間で共有される。`--allow-missing-config`により設定のない元checkoutでの
コミットも可能にする。hookはRuffの固定版を専用環境へ導入し、検査だけを行う。
型検査と全テストはCIで実行する。

## エディタとCI

VS Codeの推奨拡張とPythonのformatterを共有する。Python interpreterには開発用venvを選択する。
旧実装まで保存時に整形しないよう、保存時の一括修正は共有設定で有効にしない。
新実装のPythonファイルはGit属性とEditorConfigでLFに揃える。

CIはWindows・Python 3.13.15・固定依存で、依存整合性、Lint、整形、型、全pytestを検査する。
torchはgolden基準と同じPyPI配布版を使用する。MNISTは既存ローダーで初回取得し、
goldenテストでハッシュも確認する。CPU・OSの差で数値がずれた場合もgoldenを自動更新しない。
Linuxの研究実験環境の再現性は、このWindows開発用CIとは別に確認する。

## 導入時のレビューと検証

初回整形は独立commit `63fddfe`へ分離した。対象90 Pythonファイルについて
整形前後のAST（位置情報を除く）が一致したことを確認した。

GPT-6 Lunaの事前命名レビューはPASS。`detector_state_snapshot`でmonitor全体と
単一detectorの状態を区別し、未使用反復変数への`_`接頭辞と同名lambdaの関数化を採用した。
型対応は局所cast・理由付き行ignoreに限定する指摘も採用した。

最終検証（Windows CPU・基準venv、2026-10-05）:

- Ruff Lint成功、整形90ファイル一致。
- Pyright 69ファイル、error 0・warning 0（2.131秒）。
- `pip check`成功、pre-commit設定の検証と全ファイルの2 hookが成功。共有hookを設置済み。
- YAML・JSON設定の読込み成功。CIの公式アクション3件は実在タグに対応するSHAへ固定。
- 全pytest: **3591 passed、3 skipped、既存warning 1、128.14秒、exit 0**。
  旧11ケース・最終構成3ケースのgolden回帰と依存境界検査を含む。
- 旧参照実装・golden・旧goldenテストは固定commit `748c3aa`から差分なし。
- GPT-6 Lunaの独立レビューはPASS。対象2088テストとTorch generator別名の同一性も独立確認。
  最終全pytestは本項の結果を最終ゲートとする。

検証した追跡Pythonファイルと2 goldenのworking tree内容ハッシュ:
`a84f0081726c45de0f90dc7e203a3eca472ad22340328923867ae969a24d9e1c`。
パスを辞書順に並べ、UTF-8パス・NUL・ファイル内容・NULをSHA-256へ順に入力した。
Windowsの改行を含む内容ハッシュであり、別checkoutの改行差にも影響される。

通常権限での全テストはWindows一時ディレクトリのACLで失敗したため、必要な権限で再実行した。
Pyrightもsandbox内で子Pythonの依存解決が阻まれたため、権限を付与して最終検査した。
これらの失敗を成功として数えず、最終成功実行の結果を記録している。
ローカルJUnit・型検査JSON・hook専用cacheは元checkoutの`venv/refactoring-tests/`と
`venv/pre-commit-cache/`に保存し、コミットに含めない。
