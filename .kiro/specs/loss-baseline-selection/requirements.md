# 要件: 用途別の損失基準値選択

## Boundary Context
既存の不変な損失集計から、監視・警報区間の既存モデル再利用・警報後の参照履歴に使う平均を選ぶ。統計がない場合も明示する。
モデル/class別の統計所有、更新・seed算出・統計merge、参照ID選択、採否・学習・候補進行・snapshot時機は対象外。
既存構成の基準値方針を保存する移植であり、閾値を新しいオプションにしない。

## 入力契約
集計なし、またはbounded-loss-momentsの承認済み集計値を受ける。同specの型・値域・有限性・空状態契約を再検査する。不正入力を項目名/理由付きで拒否し入力を変更しない。

## Requirements
### Requirement 1: 監視の基準平均
1. When 集計なしまたは0件の集計を受ける, the Baseline Selection System shall 監視基準として0.01を返す。
2. When 1件以上の集計を受ける, the Baseline Selection System shall 保存平均をmin(1−1e−6,max(0.01,mean))の順で制限して監視基準を返す。
3. The Baseline Selection System shall クラス統計への置換・平均や分散の再計算を行わず、渡された集計だけから選択する。
### Requirement 2: 再利用と警報後履歴
1. When 集計なしまたは2件未満である, the Baseline Selection System shall 警報区間再利用と警報後履歴の双方について基準なしを返す。
2. When 2件以上の集計を受ける, the Baseline Selection System shall 警報区間再利用には保存平均が0なら基準なし、それ以外は保存平均を制限せず返す。
3. When 2件以上の集計を受ける, the Baseline Selection System shall 警報後履歴にはゼロを含む保存平均を制限せず返し、再利用用の欠落判定と区別する。
### Requirement 3: 検証と境界
1. If 入力が集計契約に反する, the Baseline Selection System shall 拒否し、入力と共有数値状態を変更しない。
2. The Baseline Selection System shall 旧監視・再利用・警報後履歴の直接照合と、既存監視/候補評価への明示入力で値と判断を検証できる。
3. The Baseline Selection System shall 許可された数値集計と標準機能のみを参照し、旧alias/importなし・既存golden不変・独立起動・完成範囲を記録する。
