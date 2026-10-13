# Implementation Plan

名前の付替えは、途中の状態（一部だけが新しい名前）では、importが通らない。クラス・module・属性の付替えと、goldenの書き換えを、1つのtask（1つのcommit）で行う（属性名を替えると、パラメータ名が変わり、goldenの照合が、書き換えの前のファイルでは通らないため、分けない）。

- [x] 1. クラス・module・属性の名前を替え、goldenのキーを書き換える
  - 新しい名前の確認のtest（3つの構造のそれぞれで、間の部品の型と、パラメータ名の接頭辞。前の名前が無いこと）を先に書き、REDを確かめる（前の名前のままの分類器に対して、失敗する）。
  - moduleを改名し、クラス名・属性名を、`src`・`tests/refactoring`・直下のscriptの全箇所で替える（設定の値と項目、残差adapterの部品の名前は、替えない）。分類器の本体は、名前だけを替える。部品のdocstringの、属性の名前を替える。testのファイル名と関数名のうち、分類器のクラス名を含むものを替える。依存境界の登録を替える。
  - goldenの6ファイルの、パラメータ名のキーを、書き換える（設計の「goldenの書き換え」の方法と、確かめること1・2）。
  - 完了の条件: 分類器・部品・パラメータのsnapshot・依存境界・重要動作のgoldenのtestと、軽い組（`-m "not slow"`）が、期待値を変えずに通る。前の名前の`grep`が、0件（設計の「Testing Strategy」の範囲）。設定の値と、残差adapterの部品の名前に、差分が無い。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1, 3.2, 4.1, 4.2_

- [x] 2. 全体の検証と、記録
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
- 実装（task 1。1つのcommit `8fd02e1`）。設計からの変更と、分かったこと:
  - 新しい名前の確認のtest（3つの構造で、間の部品の型、パラメータ名の並び、属性の一覧）を先に足し、前の名前のままの分類器に対して、3件とも失敗すること（`AttributeError`）を確かめてから、名前を替えた。
  - 付替えは、`src`・`tests/refactoring`・直下のscriptの65ファイル。置換の形: moduleの名前、クラスの名前、属性の読み書き（`.residual_adapter`の後に、語の文字が続かないもの）、文字列の中のパラメータ名の接頭辞、説明の中の属性の名前。旧実装のモデルを作るtestのhelper（`build_legacy_residual_adapter_classifier`）は、moduleの名前の置換に巻き込まれたので、戻した。旧実装の状態との対応のhelper（`map_residual_adapter_state_to_legacy_keys`など）、設定の値として不正な値を並べたtestの`"residual_adapter"`は、変えていない。
  - **仕様書の訂正**: goldenの、書き換えたキーの数は、「1ファイルあたり96」ではなく、基準版1が76、基準版2が100、基準版3が96（基準版で、ケースと、保有するモデルの数が違う。同じ基準版の、WindowsとLinuxのファイルでは、同じ数）。
  - goldenの書き換え（1回だけ使うscript。repoへは置かない）: 6ファイルのそれぞれで、(1) commit済みの内容を、生成のときと同じ書式で書き直すと、バイト単位で一致、(2) 書き換えの後のキーを戻すと、もとのJSONと等しい、(3) 替えたキーは、`cases.<ケース>.global_parameters`の直下だけ、(4) 書き換えの後に、`residual_adapter.`が残らない、を確かめてから、書いた。goldenに残る`residual_adapter`は、設定の値と項目の名前（`shared_backbone_residual_adapter`、`shared_backbone_without_residual_adapter`、`residual_adapter_requested_rank`）だけ。生成の記録（`source_commit`など）は、変えていない。
  - commitの前の検査で、importの並びの違反（moduleの名前が変わって、並びの順が変わった。21ファイル）が見つかり、並べ替えてからcommitした。
- 実装レビュー（独立CLI、Claude Haiku 5.5、effort medium、session `d7518548-00cf-4d5b-90b0-af4854965b6a`、対象 `8fd02e1`。選んだ理由: goldenの照合など、testの実行が要る確認なので、testを実行できるHaiku）: `IMPLEMENTATION: APPROVED`（Blocker・Major・Minorなし。任意2）。分類器の本体と`src`の差分が、名前の付替え・importの並べ替え・行の折返しだけであること、変えない名前の無変更、goldenの6ファイル（旧のキーの名前だけを写すと、新と一致。キーの数は76・100・96）、testの関数名の対応（重複なし）、依存境界のtest（名前の置換をかけると、旧と完全に一致）、固定旧の無変更を確認。独立の実行: 対象5ファイル 4479 passed、goldenの照合 127 passed、Ruff、fresh process 16 flows、名前の検査。任意1・2（分類器のmoduleとクラスのdocstringが、共有部を持たない構造に当てはまらない書き方）: 採用（`3d77ce2`）。レビュー担当は、比較用の一時ファイル6つを、`venv/refactoring-tests/claude/review/`へ作った、と報告した（手順の逸脱として記録する。主担当が、中身を確かめて、削除した。追跡ファイルの変更は無い）。
- 最終検証（Windows、`3d77ce2`、`-n 12 --dist loadfile`）: 全pytest 14013 passed / 8 skipped、exit 0（JUnit: `../../venv/refactoring-tests/classifier-structure-independent-naming-full.xml`）。Ruff（check・format）、Pyright 成功。WSL: 重要動作のgolden（Linuxのgolden。書き換えた後のファイル）・分類器・パラメータのsnapshot・fresh processのsmokeのtestで、217 passed（skipなし）。
- 完了の判断: 分類器のクラス・module・属性の名前は、3つの構造のどれにも当てはまる。値は変えていない（実旧との照合、構成の照合、掃引の識別子の期待値が、期待値を変えずに通過。goldenは、キーの名前だけを書き換え、値は同じ）。
- 未検証・残る事項: 研究室サーバ（Linuxの基準環境）での、goldenの照合は、未実施（WSLでは、通過）。
