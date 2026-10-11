# Research & Design Decisions

- 旧の`perform_hierarchical_clustering`: 4つの統計の件数が下限以上の対に、損失の距離を計算する。`distance`では、その値を、判定の値にする。提案構成（`class_functional_confidence`）では、正誤の集計がある対だけが、判定の値を持つ。クラスタは、判定の値と、閾値（`_clustering_cutoff`。どちらも`distance_threshold`）と、linkageから求める。
- 新の`cluster_and_consolidate_global_models`は、損失の距離を、診断の観測のために、すでに計算している。判定の値の出どころを替えるだけで済む。
- 値の名前: 旧の`distance`は、何の距離かが分からない。新は、計算の中身（平均損失の増加量）で書く: `mean_loss_increase`。構成の名前（`fedsda-loss-distance-clustering`）は、ユーザーが了承した案のまま。
- linkageは、旧の`distance-average`も、提案構成と同じaverageなので、変えない。
