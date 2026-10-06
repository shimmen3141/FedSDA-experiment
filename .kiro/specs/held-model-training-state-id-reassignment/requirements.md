# 要件: 保有モデル学習状態のID付替え

## 境界

既存registryの一つの保有モデル学習状態を別IDへ移す。
正式ID採番、負IDから非負IDへの制限、通知・現在帰属ID・他ownerの更新順は上位責務。
モデル欠落時のsnapshotからの再生成、標本/統計/counter/送信保留/通信は含めない。

## Requirements

### Requirement 1: 単一IDと一覧順序

1. When 登録済み元IDと変更先IDを受ける, the Training State Registry shall 同じ分類器と個別optimizer管理器を変更先IDで保持し、異なる元IDの項目を除く。
2. When 異なる既存先IDへ付け替える, the Training State Registry shall 元の学習状態で変更先を上書きし、先の既存位置と他IDの相対順を保つ。モデルの平均・合成はしない。
3. When 未登録先または元と同一のIDへ付け替える, the Training State Registry shall 元項目を取り除いた後の末尾へ置き、他IDの相対順を保つ。
4. When 元IDが未登録, the Training State Registry shall 一覧を変えず、モデルやoptimizerを生成しない。

### Requirement 2: 参照寿命・検査・学習継続

1. The Training State Registry shall 両IDを先にexact builtin intとして検査し、bool/int派生型を拒否する。負値も許す。欠落元の場合も不正項目と理由を示して拒否し、一覧と学習実体を変えない。
2. The Training State Registry shall 付替え前の取得済み状態record・binding・一覧snapshotのIDと参照を変えず、新しい取得では変更先IDと同じ分類器・管理器を返す。
3. The Training State Registry shall 分類器・共有特徴抽出器・parameter・既存gradient・現在optimizerの参照と値/蓄積stateを維持し、clone・reset・学習を実行しない。付替え後のbindingは管理器の現在optimizerを参照し、後続学習を同じ状態から継続できる。
4. The Training State Registry shall 自身の一覧以外の統計・標本・counter・保留を変更せず、共有乱数と既定数値環境を変更しない。
