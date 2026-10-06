# 要件: モデル学習標本のID付替え

## 境界

既存学習標本storeの単一モデルIDを付け替える。
採番・負IDから非負IDへの制限・正式通知・他ownerの更新順は上位責務。
サーバ評価用stored_dataと容量管理、サーバ対応表による標本連結、NN/optimizer/統計/counter/保留は含めない。

## Requirements

### Requirement 1: 列・一覧順序

1. When 元IDが登録済みである, the Training Sample Store shall 元の標本列を丸ごと変更先IDで保持し、異なる元IDの項目を除く。標本順・空列・重複・各record/Tensor参照を保持し、cloneしない。
2. When 異なる既存先IDへ付け替える, the Training Sample Store shall 元の列で変更先を上書きし、先の既存位置と他IDの相対順を保つ。旧先の標本を連結しない。
3. When 先IDが未登録または元と同じIDである, the Training Sample Store shall 元項目を取り除いた後の末尾へ置き、他IDの相対順を保つ。
4. When 元IDが未登録, the Training Sample Store shall 状態を変えず、変更先の標本列も生成しない。

### Requirement 2: 所有・検査・後続利用

1. The Training Sample Store shall 両IDを先にexact builtin intとして検査し、bool/int派生型を拒否する。負値も許す。元欠落の場合も不正項目と理由を示して拒否し、状態と標本を変えない。
2. The Training Sample Store shall 取得済み一覧snapshotのIDと列構造を変えず、付替え後の追加は変更先の列へ行える。新snapshotでは変更先IDと同じ標本参照を返す。
3. The Training Sample Store shall ID付替えで標本payloadを検査・変更せず、標本の数値/shape/device等の契約を既存samplerへ委ねる。
4. The Training Sample Store shall 自身の一覧以外のモデル・統計・評価標本・counter・保留と、共有乱数・既定数値環境を変更しない。
