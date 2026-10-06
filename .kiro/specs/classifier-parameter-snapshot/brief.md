# 分類器のparameter snapshot

研究実装のリファクタリング担当者は、旧登録処理の送信保留・候補初期化へ新モデル値を渡せるようにする必要がある。既存の候補初期化は完成済みsnapshotを受け取るが、分類器から独立snapshotを生成する境界は未移植である。
今回、準備済み最終Residual Adapter分類器の全parameter値をその時点でコピーし、モデルと独立したsnapshotを返す。

## Boundary Candidates
分類器の契約検査と値の取得・独立コピー。学習/optimizer、初期化元選択・平均、snapshotのモデルへの適用、登録、ID、送信保留状態、通信の実行は隣接責務である。旧metadata/旧keyの互換読込みを新APIへ追加しない。
