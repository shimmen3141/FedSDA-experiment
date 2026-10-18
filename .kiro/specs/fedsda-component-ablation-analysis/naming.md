# 命名表

| 名前 | 種別 | 役割 |
|---|---|---|
| analyze_fedsda_component_ablation | module | 現行FedSDAの構成要素比較専用CLI |
| main | function | 引数を検証し分析と出力を実行する入口 |

公開APIはmainのみ。補助関数・型はprivate。手法名のfedsdaと、既存runのcomponent ablationの意味を用い、新しい研究手法名を作らない。test名・引数・局所変数は命名表の対象外。

### 移設後の公開名（2026-10-18）
|名前|種別|役割|旧名との対応|
|---|---|---|---|
|tools/research_component_ablation/analyze.py|CLI module|研究用の構成要素比較解析|旧analyze_fedsda_component_ablationを配置移動。公開mainの役割は同じ|
