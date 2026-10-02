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
