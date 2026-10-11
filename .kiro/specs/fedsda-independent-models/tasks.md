# Implementation Plan

- [x] 1. モデルと学習
- [x] 1.1 共有部を持たないモデル構造と、空の共有部のoptimizer
  - 実旧との照合のtestを先に書く。
  - _Requirements: 1.1, 1.2, 1.3_
- [x] 1.2 モデルごとに順に学習する方式と、設定の組
  - _Depends: 1.1_
  - _Requirements: 2.1, 2.2_

- [x] 2. サーバと構成
- [x] 2.1 空の共有部の集約と、クロス評価の通信量
  - _Depends: 1.1_
  - _Requirements: 3.1, 3.2_
- [x] 2.2 構成の名前と、実旧の全体runとの照合（sine2とmnist2）
  - _Depends: 1.2, 2.1_
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 3. 検証
- [x] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - _Depends: 2.2_

## Implementation Notes

### 実装

- 手順: 構成の照合のtestを先に足し、実装を、照合が通るまで、段階に分けて進めた（経過は、research.md）。specの文書は、実装の後に書いた（手順からの逸脱。前の構成と同じ）。調査の文書は、3つのspecに分ける案だったが、1つにまとめた（design.mdの「進め方の変更」）。
- 照合の結果: 分類器は、同じ乱数の状態から作ると、パラメータ・出力・構築の後の乱数の状態が、旧の`SimpleMLP`と一致する（2クラスと多クラス）。構成の全体runは、sine2の7条件で、指標と離散列が、mnist2の2条件で、全状態が、実旧と一致した。固有の経路のtestも通った。
- 旧の、クロス評価の下りの通信量の数え方（共有部を持たないモデルでは、重複を省かない）を、[LEGACY-018](../../../docs/research/implementation-findings/legacy-018-cross-evaluation-download-volume-without-shared-part.md)として記録し、この構成では、旧と同じ数え方にした。

### 独立レビューと検証

- 独立レビュー（2026-10-11）: Claude Haiku 5.5（独立CLI、effort `medium`を指定。数値・乱数・更新順の実装なので）。session `d51ae405-7f8c-4a3d-8129-82251bf9dd72`、対象`653be8a..95ae52e`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし）。レビュー担当は、分類器・optimizerの状態・計算量の計測のtest（72 passed）とRuffを独立に実行し、旧のコードと読み比べて、学習の順と件数の判定、optimizerの状態の扱い（採用した候補、一時モデル、配布で作り直すモデル）、サーバの集約と通信量、設定の組の検査を確かめた。構成の照合は、時間がかかるので、依頼で、実行の対象から外した（主担当の実行による）。
- 指摘の採否（反映は`13f4d95`）:
  1. Minor: `ParameterOptimizerState`が、空のパラメータを、構造によらず受け付ける→形は変えない。この型は、パラメータの列とoptimizerだけを持つ下位の部品で、モデル構造を知らない。空の共有部を、どの構造で許すかは、構造を知る側（候補の構築、候補の学習、参加者の設定の検査）が決め、共有部が空でないのにoptimizerが無い組は、共同更新・候補の学習・保有モデルのつなぎ直しの検査が拒否する。この判断を、design.mdの表の下へ書いた。
  2. Minor: 1モデルの更新の損失（（損失×件数）÷件数）が、旧とビット単位で同じ勾配になるのは、件数が2の冪のとき→採用。research.mdへ、確かめた件数（batch 32）を書いた。
  3. Minor: testの名前の変換が、読み込み済みのmoduleの表から定数を引く→採用。呼ばれた時点で読み込む形へ直した（単独のprocessから呼んで確かめた）。
  4. Minor（記録）: 完了の記録→この記録で更新した。
  5. Minor（命名）: 定数`INDEPENDENT_MODEL_ARCHITECTURE_NAME`→採用。`INDEPENDENT_MODELS_ARCHITECTURE_NAME`へ改めた（naming.mdも）。
  6. 任意: モデルごとに順に学習する分岐の単体test→採用。`tests/refactoring/test_held_model_training_request_handling.py`へ足した（1モデルずつの呼出し、標本列の順、件数が足りないモデルの扱い、乱数の消費の順、学習量の計数）。
  7. 任意: testの補助関数の、置換の重複→採用。1回にまとめた。
- 検証（Windows基準環境）: `13f4d95`の内容で、全pytest（Claudeの実測。並列、`-n 8 --dist loadfile`）12582 passed / 4 skipped / 4 warnings、exit 0。Codexは実行ログとJUnit（`venv/refactoring-tests/fedsda-independent-models-full.xml`）を照合した。反映後の全pytestはClaudeが完了していたため、Codexは全suiteを再実行していない。goldenの3ケースと、全構成の小さい条件の照合を含む。Ruff（check・format）、Pyright 0 errors。構成の照合（`-n`なし、`-k "independent or configuration_names"`）も、反映の後に、やり直して成功。CodexがRuff（check・format）、Pyright、pip checkを再実行して成功（Pyrightはsandbox内でPythonの探索が失敗したため、承認された通常環境で再実行）。固定旧`748c3aa`からの差分は空。Blocker・Majorが無いので、反映の後の、別sessionの確認は、受けていない。
- Codexの完了照合: 全source・testの最終更新はJUnitの生成より前で、その後のsource変更はない。共用`fresh_process_smoke.py`は16の流れと全体run・保存・読込みで成功（旧実装・test moduleのimportなし）。sandbox内では一時ディレクトリの権限で中断し、承認された通常環境で再実行してexit 0。命名チェックは未登録なし。要求1〜4は各下位taskと構成・部品の実旧対照で覆い、設定→分類器→学習→サーバ→保存の接続と依存境界のtestを全pytestで確認した。未完了のtask・重大なレビュー指摘はない。
- WSLでは、実行していない（ユーザーの決定: specごとのWSLは省く）。変異テストは実行していない（既定）。

### 未検証・残る制約

- 照合は、sine2の小さい条件（7つ）と、mnist2の小さい条件（2つ）だけ（ユーザーの決定した基準）。本番の規模では、実行していない。
- 1モデルの更新が、旧とビット単位で一致することは、batchの件数32でだけ、確かめた（research.md）。
- 下りの通信量は、旧と同じ、重複を省かない数え方（LEGACY-018）。ユーザーの決定（2026-10-11）: ほかの構成の数え方へ揃える。修正時期は未確定。主担当の推奨は、全構成のgoldenを作る前に、LEGACYをまとめて直すとき。
- 分類器のクラスと属性の名前（`ResidualAdapterClassifier`、`residual_adapter`）は、この構造の中身と合わない（改善候補IMPROVE-014）。
- この構成のgoldenは、まだ無い（全構成のgoldenを、旧実装を外す前に、まとめて作る）。
