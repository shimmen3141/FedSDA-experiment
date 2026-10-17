# 命名表

| 名前 | 種別 | 役割 |
|---|---|---|
| analyze_fedsda_component_ablation | module | 現行FedSDAの構成要素比較専用CLI |
| main | function | 引数を検証し分析と出力を実行する入口 |

公開APIはmainのみ。補助関数・型はprivate。手法名のfedsdaと、既存runのcomponent ablationの意味を用い、新しい研究手法名を作らない。test名・引数・局所変数は命名表の対象外。
