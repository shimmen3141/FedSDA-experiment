# Implementation Plan

名前の付替えは、途中の状態（一部だけが新しい名前）では、importが通らない。クラス・module・属性の付替えと、goldenの書き換えを、1つのtask（1つのcommit）で行う（属性名を替えると、パラメータ名が変わり、goldenの照合が、書き換えの前のファイルでは通らないため、分けない）。

- [ ] 1. クラス・module・属性の名前を替え、goldenのキーを書き換える
  - 新しい名前の確認のtest（3つの構造のそれぞれで、間の部品の型と、パラメータ名の接頭辞。前の名前が無いこと）を先に書き、REDを確かめる（前の名前のままの分類器に対して、失敗する）。
  - moduleを改名し、クラス名・属性名を、`src`・`tests/refactoring`・直下のscriptの全箇所で替える（設定の値と項目、残差adapterの部品の名前は、替えない）。分類器の本体は、名前だけを替える。部品のdocstringの、属性の名前を替える。testのファイル名と関数名のうち、分類器のクラス名を含むものを替える。依存境界の登録を替える。
  - goldenの6ファイルの、パラメータ名のキーを、書き換える（設計の「goldenの書き換え」の方法と、確かめること1・2）。
  - 完了の条件: 分類器・部品・パラメータのsnapshot・依存境界・重要動作のgoldenのtestと、軽い組（`-m "not slow"`）が、期待値を変えずに通る。前の名前の`grep`が、0件（設計の「Testing Strategy」の範囲）。設定の値と、残差adapterの部品の名前に、差分が無い。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1, 3.2, 4.1, 4.2_

- [ ] 2. 全体の検証と、記録
  - 全pytest（実旧との照合、基準版1〜3、構成の照合、掃引の識別子の期待値が、期待値を変えずに通る）、Ruff、Pyright。WSLで、goldenのtestと、分類器のtestを実行する。
  - 独立CLIの実装レビューを受け、指摘を反映する。
  - 基準の文書（`docs/experiments/refactoring-baseline.md`。goldenの、キーの名前を書き換えたこと）、IMPROVE-014の記録と一覧、再開案内を更新する。いまの分類器の場所を書いた文書（`docs/research/independent-models-migration-scope.md`の、新実装の分類器のmoduleとクラスの名前）を、新しい名前へ替える。旧実装の選択肢や構造を説明する文書（`docs/components/shared-backbone.md`、`docs/reference/options.md`、`docs/experiments/baselines.md`の`residual_adapter`は、旧実装の設定の値・ディレクトリ名）と、過去のspec・引継ぎの履歴は、当時の記録なので、替えない。
  - _Requirements: 2.4, 3.3, 3.4, 4.3_

## Implementation Notes
- 仕様レビュー1回目（独立CLI、GPT-6 Luna、effort medium——ログのmodel行・reasoning effort行で確認——、session `01a13455-49a3-73f2-aa9b-b5f6d5f711f8`、対象は基点`65011cd`の上の未commitの仕様書群。読取りだけ）: 判定の行は`SPECIFICATION: CHANGES REQUIRED`。指摘は、Minor 1、任意2（Blocker・Majorなし）。goldenの、`global_parameters`のキーのほかに、パラメータ名が入る項目が無いこと、goldenの検査が、生成の記録（`source_commit`）を、いまのsourceと照合しないことは、確認済み、の報告。採否:
  - 1（Minor。更新する文書が、曖昧。`docs/research/independent-models-migration-scope.md`が、いまの分類器の場所を、前のmoduleの名前で書いている）: 採用。task 2へ、替える文書と、替えない文書（旧実装の説明、過去の記録）を、名指しで書いた。
  - 2（任意。クラス名の、別の候補`ConfiguredModelArchitectureClassifier`。現案を採るなら、判断をnamingへ）: 現案のまま。理由をnamingへ書いた（分類器は、どれも、設定で構造が決まる。`Configured`を足しても、区別は増えない）。
  - 3（任意。パラメータ名を持つ・戻す処理——`named_parameters`・`state_dict`・`load_state_dict`——が、同じ版の分類器の間で使われる、という条件と、順序の確かめ方を、設計へ）: 採用。設計の「変えないもの」へ書いた。
- 仕様レビュー2回目（修正の再確認。同じ担当・effort、session `01a13457-a795-7cf1-8739-b2872b33c0f6`、読取りだけ）: `SPECIFICATION: PASS`（Blocker・Major・Minorなし。任意3件は、反映の確認）。
