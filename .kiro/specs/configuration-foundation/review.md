# 要求・命名の確認記録

## エージェントによる初期確認

- 要求の章は数値IDで、各章に観測可能な受け入れ条件がある。
- preset展開、明示指定、未知項目、不正値、組合せ、保存・復元、登録漏れを含む。
- 手法の計算・学習、CLI、ファイルI/Oはこの要求の対象外。
- 技術・型・関数名の候補は`naming.md`へ分け、要求に実装構造を押し込まない。
- 既存文書・ユーザー指示と照合するinline review。独立した人間/別agentの承認ではない。

## 人間の確認

- 要求: 2026-10-02の「承認します。SDDに従って進めてください」で承認済み。
- ルートパッケージ名`federated_learning_experiments`: 2026-10-02のユーザー返答で承認済み。
- 命名revision 2全体: 同じユーザー返答で承認済み。承認時のLF正規化SHA-256をspec.jsonの履歴へ保存した。
- 命名revision 4: 2026-10-03の「承認します。次に進んでください」で承認済み。
- 設計: 同じユーザー返答で承認済み。レビューの自己判定を人間承認として扱ったものではない。
- task: 初回の実行計画13タスクは2026-10-03の「実装に進んでかまいません」で人間承認済み。後続4〜8は具体化前の計画。
- 命名revision 5: 同日の「データセット名についてはその方針で修正してください」で既存名維持を承認済み。
- 命名revision 6: 第9節へ実装補助関数・内部変数・テスト名を具体化。追加分だけ人間承認待ち。
- 新実装開始: 承認済みの境界を扱うtask 1.1から着手した。

人間の明示的な返答に基づいて承認欄を更新する。未承認の追加名へ承認を拡張しない。

## 2026-10-02の改善記録

- 名前の短さより、意味の明確さ・一意に区別できること・実態との一致を優先する方針を反映した。
- 型・関数・値の対象、入力と解決済み条件、単位、容量と現在件数、指定rankと実rankを区別した。
- FIFO容量、e-SR alpha、前向き検証件数、Adapter rankの意味を既存コードと照合した。
- 最終構成の設定・検証から小規模な実行経路へ接続し、汎用基盤は後続に追加する順序を正本とroadmapへ記録した。
- 全体の要求は上記の後続返答で承認済み。初回と後続の実装範囲は設計・taskでも明示する。

## 設計生成時の確認結果

- Discovery: integration-focused。既存の設定・制約・正本・goldenを照合し、標準ライブラリの契約を公式資料で確認した。
- 受け入れ条件19件のIDを抽出し、設計の対応表が不足・余分なしに一致することを確認した。
- 担当範囲・対象外・許可依存・再検証条件を明示し、14ファイルの役割と命名表の対応を確認した。
- 元入力の変更、深い可変値、数値の暗黙変換、非適用指定、重複登録、旧名の受理に対する契約を確認した。
- 各機能の設定値検証と外側の組合せ検証を分け、例外型を基礎層へ置いて循環参照を避けた。
- 初回だけで全要求を完了扱いにしないこと、全run条件が揃うまで実行層へ接続しないことを明示した。
- 新コード・先取りテスト・taskは未作成。コードとgoldenは変更していないため、数値回帰テストは今回は実行していない。
- `kiro-spec-design`の次段階ゲートに従い、設計承認後に`kiro-spec-tasks`へ進む。

## gpt-6-lunaの独立レビュー

- ユーザーの明示指定でgpt-6-lunaへレビューを依頼した。レビュー担当はコード・文書を編集していない。
- 指摘を主担当が根拠と照合し、[luna-review.md](luna-review.md)に対応判断を保存した。
- reset範囲の名称と既存挙動の不一致、設定の直接構築時の検証、部分型の明示、値域宣言の導出、
  FIFO容量とFixed-Share時間尺度の結合、小規模実行へ接続する条件を改善した。
- 機械的確認は設計の要求対応・文書参照・JSON・承認状態に限る。実装の成功やgolden同値を意味しない。
- gpt-6-lunaの再確認では主要な設計指摘の解消が確認された。最終のrevision番号と未承認状態は主担当が機械的に照合した。
- oracle概念別診断routerは割当先変更時も保持される点を、名称の説明に追記した。

## 2026-10-03のタスク生成

- 設計と命名revision 4への明示承認を記録した。元の承認対象のhashは履歴へ保存し、状態文だけ更新した現在文書のhashも記録した。
- kiro-spec-tasksに従い、実行環境・基礎例外・機能別の設定・組合せ検証・集約型・回帰確認の初回タスクを作成した。
- タスクグラフの独立確認を新しいエージェントへ依頼した。モードはindependent、確認は2回。
- 初回判定はNEEDS_FIXES。追加設定コードの前提不足、基礎例外の先行依存、契約の観測条件の不足を修正した。
- 2回目の最終判定はPASS。判定表現を確認し、初回1〜3だけを実行タスク、4〜8はcheckboxのない後続計画とする限定範囲であることを明確にした。三度目のレビューは行っていない。
- 後続の未確定範囲を実行タスクへ見せず、棚卸し・契約承認後に再分割・レビュー・人間承認する前提を明記した。
- 初回datasetの正式値を具体化した。既存data/specs.pyと最終goldenを照合し、末尾2は概念数、MNISTは10クラスであることを確認した。
- task承認と命名revision 5の追加案確認を待つ。新src・先取りテスト・実験はまだ作成・実行していない。

## 2026-10-03の実装開始・task 1.1

- ユーザーが初回taskの実装開始とデータセット名の維持を承認した。revision 5は`sine2`・`sea2`・`mnist2`を初回の正式値へ修正した。`mnist4`等も後続で扱う際は既存名を維持する。
- `pytest.ini`と`src/federated_learning_experiments/__init__.py`を作成した。旧import経路は変更していない。
- 元checkoutの`venv/Scripts/python.exe`を用い、保存済み環境とPython・libraries・build・全依存freeze hash・golden hash・dtype/threads・OS種別/architectureを照合して一致した。
- 新パッケージの実importと既存schema・旧/最終goldenの30テスト収集が成功した。
- 初回sandboxの全体検証は110 passed・2 setup ERROR・1 FAIL。いずれもWindowsのpytest一時領域へのアクセス制限によるものであり、数値の不一致として扱わない。
- 専用basetempとcacheprovider停止を指定し、権限を拡張して9ファイルを再実行した。**113 passed、exit 0、182.08秒**。旧11ケースと最終3ケースのgoldenを含む。
- 独立レビューは`.agents/skills/kiro-review/SKILL.md`を適用し、task 1.1に限定して**APPROVED**。機械的証拠・境界・golden保全を確認し、完了検証をVERIFIEDとしてtask 1.1だけを完了にした。
- 非behavioralな環境構成のため、新規テストは作成していない。導入前の失敗ログをREDとして捏造せず、実import・収集・既存回帰を証拠とする。
- 設定型・アルゴリズムは未実装。既存経路の回帰成功は、新アルゴリズムの同値性を意味しない。
- task作成時に追記予定だった内部関数・テスト名が未記載だったため、第9節へrevision 6の追加案を具体化した。共通値域検証の基礎層への配置も明記し、その追加分だけ人間へ非同期で確認している。

再検証コマンド（worktreeルート、既存venvとMNISTを共有）:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MPLCONFIGDIR=(Resolve-Path ../../venv/matplotlib-cache).Path
$env:FDE_MNIST_DATA_DIR=(Resolve-Path ../../data/mnist).Path
../../venv/Scripts/python.exe -m pytest --ignore=.venv -p no:cacheprovider `
  --basetemp=../../venv/refactoring-baseline-check/reviewer-tmp-escalated `
  tests/test_option_schema.py tests/test_parameter_schema.py tests/test_metric_schema.py `
  tests/test_shared_backbone.py tests/test_provisional_model.py tests/test_clustering_decision.py `
  tests/test_fedsda_configuration.py tests/test_regression.py tests/test_proposed_regression.py -q
```

basetempはpytestが再作成する専用の一時領域。実験成果物の保存先を指定しない。

## task 1.2

- 独立レビュー: APPROVED。基礎例外の3属性・日本語理由・許容条件を確認した。
- TDD: RED ModuleNotFoundError (exit 1) / GREEN 3 passed (exit 0)。
- 完了検証: VERIFIED。旧経路の統合goldenはtask 3.3で確認する。

## task 2.1: dataset・実験規模・seed・集約間隔を検証する

- 独立レビュー: APPROVED。実差分・命名・境界・契約を確認。
- 対象機能テスト: 88 passed / exit 0。
- 完了検証: VERIFIED。承認済みの初回範囲に限定し、spec全体の完了を意味しない。

## task 2.2: モデル構造と要求rankを検証する

- 独立レビュー: APPROVED。実差分・命名・境界・契約を確認。
- 対象機能テスト: 113 passed / exit 0。
- 完了検証: VERIFIED。承認済みの初回範囲に限定し、spec全体の完了を意味しない。
