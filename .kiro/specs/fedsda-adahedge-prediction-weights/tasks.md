# Implementation Plan

- [x] 1. 名前
- [x] 1.1 重みの出どころの名前を、方式の名前へ改める（値を変えない）
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. 構成の追加
- [x] 2.1 AdaHedgeの重みでの予測、Fixed-Share側の記録・指標・表の列、構成の名前
  - 実旧との照合のtestを先に書く。
  - _Depends: 1.1_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

- [x] 3. 検証
- [x] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - _Depends: 2.1_

## Implementation Notes

### 実装

- 名前（`c345a88`）: 対応表（naming.md）で、語の境界の置換と、ファイルの移動を行った。置換の後、残りを手で直した: testどうしのimport（改名したtestのmodule名）、Fixed-Shareの保持の状態の名前（testの中の辞書のkey）、説明文とエラーの文の言い方。改名だけの時点で、全pytestは12506 passed / 4 skipped（改名の前は12505。差の1件は、直前のspecで足したtest）。
- 構成（`acff6f3`）: 構成の照合のtestと、実旧との対応の直しを先に入れ、古いsourceで読込みに失敗することを確かめてから、sourceを直した。specの文書は、実装の後に書いた（手順からの逸脱。前の構成と同じ理由）。
- 照合の結果: 小さい条件の7つ全部で、構成の全体runの、指標の全部と離散列が、実旧（`SOFT_ROUTING_CONTEXT = "global"`）と一致した。値は、最初の実装で一致した。直したのは、testの側の対応（旧の、Fixed-Shareの結合と実際の予測の差の3指標を、「いつも0」としていた箇所と、件数の期待）。
- 作業中の誤り: 編集用のscriptが、途中の検査で止まった後、続きのコマンドが、同じscriptをもう一度実行し、2つのtestファイルへ、同じ項目が二重に入った。気づいて、二重の行を消した（commitの前。全pytestで確かめた）。

### 独立レビューと検証

- 独立レビュー（2026-10-11）: Claude Haiku 5.5（独立CLI、effort `medium`を指定）。GPT-6 Lunaは利用上限で使えない期間。session `944bf208-4d68-4165-955c-6b0a94ecb95b`、対象`28ba40a..acff6f3`。判定は`IMPLEMENTATION: APPROVED`（Blocker・Majorなし）。レビュー担当は、構成の照合（42 passed）、予測と記録のtest（90 passed）、Ruffを独立に実行し、旧の`_record_prediction`と突き合わせて、最大重みのモデルの選び方・確信度が同率のときの選び方・実効モデル数・更新に渡す重みが、新旧で同じことを確かめた。レビュー担当は、許可した一覧にないshellのコマンド（`git diff`や`git grep`の出力の絞り込み）を使った、と申告した（読取りだけ）。
- 指摘の採否（反映は`26ddf77`）: (1)Minor: AdaHedgeの重みで予測する構成では、AdaHedgeの結合が実際の予測になるのに、説明文が「予測と学習の判断には使わない」のままの箇所（3か所）→採用。直した。(2)Minor: Fixed-Shareの保持の、説明文の「予測重み」と、内部の名前`_clear_prediction_weight_evidence`→採用（`_clear_weight_evidence`）。(3)Minor: 「いつも0」の確認の条件が、正解の数の一致（データに依る）→採用。構成に基づく条件へ直した（構成が、Fixed-Shareの重みで予測するかどうかを渡す）。重複していた件数の検査を1つ消した。(4)Minor: 取り違えの検査から、3つの利得の指標を除いた→変えていない（提案構成の観測では、3つとも0で、区別できない。この構成の照合が、0でない値で、旧の値と一致することを確かめる）。(5)Minor: 固有の経路のtestの、実旧の側の検査は、clientごとの件数→変えていない（指標と離散列の照合が本体）。(6)Minor: 移行の文書の括弧→直した。
- 検証（Windows基準環境）: 指摘の反映の後（`26ddf77`の内容）で、全pytest（並列、`-n 8 --dist loadfile`）12514 passed / 4 skipped / 4 warnings、exit 0。goldenの3ケースと、提案構成の小さい条件の照合を含む。JUnitはfailure 0、error 0。Ruff（check・format）、Pyright 0 errors。固定旧`748c3aa`からの差分は空。この後は、specの文書・steering・掃引の使い方の文書だけを変えた。
- WSLでは、実行していない（ユーザーの決定: specごとのWSLは省く）。変異テストは実行していない（既定）。

### 未検証・残る制約

- 照合は、sine2の小さい条件だけ（ユーザーの決定した基準）。ほかのdataset、本番の規模での、この構成の新旧の一致は、確かめていない。
- 保存する記録の列の名前と、要約の指標の名前が変わった（表`sample_prediction`は130列→133列）。この変更より前に、新実装で保存した結果（研究室サーバで実行中の、提案構成の新旧比較の出力を含む）は、古い名前の列を持つ。比較の道具（`compare_sweep_results_with_legacy.py`）が読む11指標の名前は、変えていない。
- AdaHedgeの証拠のmoduleは、評価の層に置いたまま（改善候補IMPROVE-013）。
- この構成のgoldenは、まだ無い（全構成のgoldenを、旧実装を外す前に、まとめて作る）。
