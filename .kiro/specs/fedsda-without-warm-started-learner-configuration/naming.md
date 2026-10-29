# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `WITHOUT_WARM_STARTED_LEARNER_CONFIGURATION_NAME`（値`fedsda-without-warm-started-learner`） | `sweep/experiment_configurations`の定数 | 最終構成の既定から、「新しいモデル」の仮説のwarm startの学習者だけを外した構成（初期化し直した学習者だけで測る） | 構成要素の比較の`fedsda-without-…`の命名にそろえる。設定の値`reinitialized_learner`は基準版10までの既定の学習者の組 |
