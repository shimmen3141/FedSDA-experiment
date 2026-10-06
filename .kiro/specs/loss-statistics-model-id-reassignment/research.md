# 調査根拠

- 固定旧`clients/base.py:425`の`confirm_model_registration`は現在帰属IDが負のとき`model_stats[new_global_id] = model_stats.pop(temp_id)`を行う。元欠落は何もしない。既存先は上書きして位置維持、未登録先は末尾、同IDではpop後末尾。実旧メソッドをoracleにし、他ownerを空にして統計部分を観測する。
- 旧メソッド全体にはモデル欠落時の再生成、標本の付替え、加算カウンタ、保留解除がある。本specはそれらを移植しない。非負現在IDの早期returnも上位条件であり、storeの汎用signed ID操作には持ち込まない。
- `model-and-class-loss-statistics`はID変更を明示的に後続へ分離している。既存storeの所有辞書のAPIを拡張する。新しい汎用registry/rename utilityは不要。
- `model-id-mapped-loss-statistics-selection`はサーバ対応表を一回適用し、件数最大で衝突を解決する。本specの上書きとは異なるため流用しない。
- `pending-model-upload`はparameter snapshotと対応IDのみを保持する。上位が旧IDの現在統計を取得し、付替え後に現在帰属IDを切り替え、明示clearする接続をtest-onlyで検証する。
- 同ID・正の元IDはstore一般契約として直接検証し、旧負IDの正式登録対照と区別する。新たな旧実装の不具合は現時点では観測していない。
