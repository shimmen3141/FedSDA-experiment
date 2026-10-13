# Research & Design Decisions

## Summary
- Feature: classifier-structure-independent-naming（IMPROVE-014）。2026-10-13、ユーザーが、整理（値を変えない）として採用を了承。同日、範囲の選択肢（属性名も変える／クラスとmoduleだけ／見送る）から、「属性名も変える」を、ユーザーが選んだ（goldenの6ファイルの、パラメータ名のキーを、機械的に書き換えることを含む）。

## Research Log
- [IMPROVE-014の記録](../../../docs/research/improvement-candidates/improve-014-rename-classifier-that-may-have-no-adapter.md): クラスを構造に依らない名前へ、属性を改め、moduleの名前も合わせる。共有部を持たない構造も含めて、名前を決める。決定の表: 実データの列名・state_dictキー・golden hashへの波及を確認する。
- コード（`65011cd`）:
  - 属性`residual_adapter`は、パラメータ名の一部（`residual_adapter.feature_compression.weight`など）。
  - パラメータ名を、文字列で読む処理は、サーバの集約の、接頭辞`feature_extractor.`だけ。`residual_adapter.`を読む処理、パラメータ名の順に依る計算は、`src`に無い。
  - パラメータ名は、runの保存ファイル（`runtime/*artifact*.py`、`sweep/`）には、入らない。
  - 重要動作のgoldenの6ファイルには、`cases.<ケース>.global_parameters`のキーとして、1ファイルあたり96（17ケース）。値のdigestは、名前を含まない（`behavior_golden_cases.py`の、配列ごとのdigest）。goldenの書式は、`_write_exclusive`（`ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False`、末尾に改行、LF）。
  - goldenのクラス名・module名の出現は、0。
  - 記録の表の列名・指標の名前に、`residual_adapter`は、無い。保存するrun設定には、設定の値（`shared_backbone_residual_adapter`など）と、項目（`residual_adapter_requested_rank`）がある（変えない）。
- 旧の成果物（固定旧のgolden、旧の実験結果）は、旧のモデルのパラメータ名（`adapter.down.`など）を使い、新の名前を含まない。

## Design Decisions
- 属性名も変える（ユーザーの選択）。クラス名だけを変える案は、名前と中身のずれが、属性の側に残る。
- goldenは、実行し直さずに、キーの名前だけを書き換える。理由: 値は変わらない（名前だけの変更）。Linuxのgoldenは、Linuxの基準環境が無いと作り直せない。書き換えが、キーの名前だけであることは、前後の比較で確かめられる。
- 残差adapterの部品（`NonlinearResidualAdapter`）と、設定の値・項目の名前は、変えない。前者は、実際に残差adapterを指す。後者を変えると、保存するrun設定と、runの識別子が変わる（実行中の、サーバの比較の結果と、対にできなくなる）。
- クラス名は、構造を決める設定の型（`ModelArchitectureSettings`）と語をそろえて、`ModelArchitectureClassifier`。属性名は、「モデルごとに持つ」「特徴を変える」の2点を言う`model_owned_feature_transform`（既にある部品名`ModelOwnedHiddenLayers`と、語がそろう）。

## Risks & Mitigations
- 置換が、変えない名前まで替える → 置換の形を限り、設定の値・部品の名前の無変更を、`git diff`と、run設定・識別子のtestで確かめる。
- goldenの書き換えの誤り → 書式の一致（もとのファイルを、同じ書式で書き直すと、バイト単位で一致）、内容の一致（キーを戻すと、もとと等しい）、照合のtest。

## References
- [IMPROVE-014の記録](../../../docs/research/improvement-candidates/improve-014-rename-classifier-that-may-have-no-adapter.md)、[基準の文書](../../../docs/experiments/refactoring-baseline.md)、[構造と命名](../../steering/structure.md)
- 利用skill: kiro-spec-init/requirements/design/tasks、kiro-impl（--review off）、kiro-validate-impl。共通引継ぎ手順に従い、独立CLIの仕様レビューと実装レビューを受ける。
