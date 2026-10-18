# 命名

| 公開名 | 役割と区別 |
| --- | --- |
| tools/research_memory_reclamation/run.py | 学習や保持量を変えず、実行側で解放済み領域の回収を試す独立研究CLI。製品runtimeや構成登録先ではない |
| main | 計画、対照の逐次起動と照合、非成功終了を管理するCLI入口。数値計算・採否判定を行う関数ではない |

その他のhelperは非公開名。専用checksのtest名は命名表の対象外。共通srcに新しい公開名を追加しない。
