# Research Log

## 調べたこと（light discovery。既存の製品の拡張）
- 責任度の提供の関数`build_released_sample_responsibility_provider`（`runtime/released_sample_training_responsibilities.py`）は、`fedsda_run_client.py`の2か所（独立したFixed-Shareの重みを使う構成と、柔らかい学習の既定）で組み立てられ、`ModelTrainingSampleStore.training_responsibility_provider`に置かれる。`append_model_training_samples`が標本ごとに`provider(model_id, sample)`を呼び、その後で固い割り当ての列へ足す。
- `calculate_training_responsibilities`は、重みと確率の辞書が同じ非空の集合で、`fallback_model_id`がそれに含まれることを要求し、その集合の中で正規化する。辞書を絞れば、絞った集合の中の責任度になる。
- 責任度が0のモデルは`ResponsibilityWeightedTrainingStore`の`support`が増えず、`sample_batches`の参加（正の責任度の標本数がbatch以上）に入らない。学習量も0なので、集約にも参加しない。
- 柔らかい学習では、ほぼすべての保有モデルが毎回フルサイズのbatchで学習される（`responsibility-spread`の診断: 連合学習・間隔50で、1回の更新に参加したモデル数3.49、保有モデル数3.50）。配られただけのモデルも、手元の標本で毎回更新されている。
- 既存の構成の固定は、`test_soft_training_single_model_prediction.py`のfixtureの照合が担う。新しいfieldを既定値の省略の宣言にすれば、既存の写像は変わらない。

## 判断
- 重みと確率は全保有モデルで求めてから絞る: 重みの同期（Fixed-Shareの集合の更新）とforwardの回数を既定と同じにし、予測の重みの状態と計算量の計数を変えないため。
- 「自分で固く割り当てた」の判定に、固い割り当ての標本の列（`_training_samples_by_model_id`）を使う: 柔らかい学習でも固い学習と同じ規則で保持されていることを、前のspecで確認済み。列のcopyを避けるため、IDの集合だけを返すメソッドを足す。
- `fallback_model_id`を必ず含める: 初めて固く割り当てるモデル（配られたモデルが選ばれた直後、採用した候補）を対象に入れ、責任度の集合が空にならないようにする。
