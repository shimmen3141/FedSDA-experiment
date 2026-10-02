# 命名revision 6のgpt-6-lunaレビュー

2026-10-03。ユーザーは、第9節のレビューと有用な指摘の反映をもって承認とし、以降も同じ方式で命名を承認するよう明示的に委任した。
レビュー担当はgpt-6-luna、編集は主担当。要求・設計・taskの承認を一般に委任したものではない。

| 指摘 | 主担当の判断 | 理由・反映 |
|---|---|---|
| `allowed_parameter_values`を`allowed_values`等へ変更 | 不採用 | 設定のparameterは文字列も含む。対象が設定項目であることを保つ現行名が明確。型検査と許容値の厳密一致は別に検証する |
| metadataテストが型注釈も扱うことを名前へ含める | 採用 | `test_field_annotations_and_metadata_declare_parameter_constraints`へ変更した |
| 候補検証の奇数件数テストを明確にする | 一部採用 | `test_candidate_future_validation_accepts_odd_sample_counts`へ変更。検証するのは将来標本数であり、候補モデル数への変更は実態と異なるため採用しない |
| 共通検証のcore配置・循環依存・他の名前 | 維持 | 設定フィールドの値だけを読む基礎層とし、機能型・集約型・方式カタログへ依存しない |

公開型・公開フィールド・正式値は変更しない。共通検証の許可依存を設計へ反映し、revision 6を承認済みとして実装へ進む。

## revision 7: 単独実験条件型の配置

実装準備で、集約型と実験条件型を同じファイルへ置くと、集約型→組合せ検証→実験条件型のimportで循環することを確認した。
gpt-6-lunaは`configuration/experiment_run_conditions.py`への分離を妥当と判断した。
実験条件型はcoreの値検証だけを呼び、外側へ依存しない点を明記する指摘を採用した。
型・項目・契約は維持し、配置表と命名表を更新してrevision 7を承認した。

## revision 8: 共通値検証の直接テスト

直接テスト3件とテスト専用dataclass名をgpt-6-lunaへ提示した。
名前と役割が一致し、変更必須の指摘なし。`invalid_numeric_values`への限定案は、bool・型違いも扱うため推奨されなかった。
主担当は全5名を維持し、命名表第11節へ記録して承認した。
