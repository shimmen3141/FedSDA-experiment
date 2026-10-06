# 要件: 損失統計のモデルID付替え

## 境界

既存のモデル全体・クラス別損失統計storeに、単一の元IDから変更先IDへの付替えを追加する。
正式登録の通知と呼出順、ID採番・負IDから非負IDへの制約は上位処理が担う。
モデル実体、標本、optimizer、カウンタ、送信保留、通信、client全体の登録は含めない。
サーバ統合時の対応表適用・件数による統計選択とは別の操作である。

## Requirements

### Requirement 1: 値と順序を保つ単一付替え

1. When 登録済み元IDと変更先IDを受ける, the Loss Statistics Store shall 元IDの全体・全クラス統計を丸ごと変更先IDへ移し、元IDの項目を除く。件数・平均・偏差平方和・クラス順を変えない。
2. When 変更先IDが既存で元IDと異なる, the Loss Statistics Store shall 元統計で変更先統計を上書きし、変更先の既存位置と他IDの相対順を保つ。統計の加算や件数による選択は行わない。
3. When 変更先IDが未登録、または元IDと同一である, the Loss Statistics Store shall 元IDを取り除いた後の末尾へ統計を置く。他IDの相対順を保つ。
4. When 元IDが未登録, the Loss Statistics Store shall 変更先の有無にかかわらず状態を変えず、新しい空統計を生成しない。

### Requirement 2: 検査・所有・上位接続

1. The Loss Statistics Store shall 両IDに負値を含むexact builtin intを要求し、boolとint派生型を拒否する。元IDが欠落する場合も両IDを事前検査し、不正入力を項目名と理由を示して拒否し、状態を変えない。
2. The Loss Statistics Store shall 公開取得値の独立性を維持し、付替え前に取得した統計を変えない。付替え後の帰属損失は変更先の現在統計へ更新できる。
3. The Loss Statistics Store shall ID付替えで自身の統計だけを更新し、送信保留の解除・対応IDの更新・登録時parameter snapshotの変更を引き起こさない。それらの操作は上位から明示的に実行する。
4. The Loss Statistics Store shall 付替えで共有乱数と既定数値環境を変更しない。
