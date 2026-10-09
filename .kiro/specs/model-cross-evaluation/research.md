# Research & Design Decisions

## Summary

- **Feature**: `model-cross-evaluation`
- **Discovery Scope**: New Feature（clientの評価とサーバの集計は新規。損失と予測の計算は既存の部品）
- **Key Findings**:
  - 旧の最終構成のサーバは、新規モデルが登録されたラウンドで、集約と配布の間に、クロス評価→クラスタリング→統合を行う。大きいので、クロス評価（本spec）と、クラスタリングと統合（次のspec）に分けた。
  - クロス評価は、clientの状態を変えないが、乱数を消費する（clientの抜出しと標本の抜出しはPythonの乱数、評価用のモデルの生成はtorchの乱数）。
  - 損失の和は、旧ではfloat32のNumPyの配列の和である。torchの和とは足す順が違うので、同じ演算で求める。

## Research Log

### 旧のクロス評価

- **Sources Consulted**: 旧`servers/fedsda.py`（49〜70行の`run_round`、89〜130行の`_cluster_and_consolidate`、コンストラクタ）、`servers/clustering.py`（`_cross_evaluate`、`_record_functional_pair_stats`、転送の記録）、`servers/shared_backbone.py`（32〜52行）、`clients/base.py`（468〜636行）、`models.py`（`predict`、`per_sample_error`、`per_sample_error_and_prediction`）、`clustering.py`（`FunctionalPairStats`）、`experiment.py`（100〜112行のクラスタリングの有効化、1824〜1860行の診断の保存）、`tests/test_proposed_regression.py`（最終構成の設定）。
- **Findings**:
  - 最終構成: クラスタリングの開始は「新規モデルがあるラウンド」（`on_new_model`。実験の枠が、登録の前に、送信できるモデルを持つclientがいるかを見る）、判定は`class_functional_confidence`、linkageは`average`、後処理は`merge`、信頼水準0.95、対の診断の収集あり。
  - `_cluster_and_consolidate`は、対象のモデルが1つ以下なら、クロス評価をしない。対象は、全clientが保有する非負のID（集約の対象と同じ列。昇順）。
  - `_cross_evaluate`の引数は、最終構成では、モデルIDの列とラウンドだけ（モデルを送る、キャッシュを使わない、損失差を集めない）。
  - 判定が`class_functional_confidence`なので、評価する側と対象が違う組は、常に`evaluate_model_diagnostics(params_i, target_model_id=id_j, include_class_correctness=True)`、同じ組は`evaluate_model(params_i, target_model_id=id_j)`。
  - clientの`evaluate_model`の和は`float(errors.sum())`（`errors`はfloat32のNumPyの配列）、2乗和は`float((errors ** 2).sum())`。
  - `evaluate_model_diagnostics`の対象のモデルの予測は、clientが保有するモデル（配布の前の、ローカルの値）の`predict`。渡されたモデルの予測は、損失と同じ1回の出力から求める。
  - クラス別は、`torch.unique(labels)`の順（昇順）。
  - `get_held_model_ids`は集合を返す。サーバは、clientの順に、モデルIDごとの保有clientの列を作る。
  - 共有部を持つサーバの転送の記録: 共有部はclientごとに1回、概念固有部は（モデル、client）ごとに1回で、モデル転送数もそのとき1足す。
  - 診断の保存（`experiment.py`）は、3つの記録を、列の配列にして保存する。本specは、記録を持つところまで。
- **Implications**: clientの評価を1つの関数（正誤の比較の有無を引数）にし、サーバのクロス評価が、組を走査して、通信量・記録・集計を行う。

### oracle

- 再較正つきのラウンドの対照（tests/refactoring/test_post_aggregation_prediction_recalibration.py）は、旧の`run_round`全体を呼ぶ。クロス評価は、`run_round`の途中（集約と配布の間）にあるので、本specの対照は、登録→集約→（クロス評価）→配布→再較正を、旧の部品を直接呼ぶ形にする（旧は`_register_new_models`、`update_global_models`、`_cross_evaluate`、`broadcast_models`、全clientの`recalibrate_routing_after_aggregation`）。クロス評価の結果は、どちらの実装も使わないで先へ進む。
- 実旧の`_cross_evaluate`は、返す表のほかに、`_last_pair_functional_stats`・`pair_prediction_diagnostics`・`cross_evaluation_diagnostics`・`cross_evaluation_class_diagnostics`を更新する。これらと照合する。
- 旧の設定の差し替えへ、評価標本の追加の件数、評価の標本の上限（`EVAL_MAX_SAMPLES`）、clientの上限（`CROSS_EVAL_MAX_CLIENTS`）を足す（既定は、いまの値のまま）。

## Design Decisions

### Decision: クロス評価と、クラスタリング・統合を、2つのspecに分ける

- **Rationale**: クロス評価は、clientでのモデルの生成と標本の抜出し（乱数）、通信量の重複しない計上、3種類の診断の記録を含み、それだけで1つのまとまりである。クラスタリングと統合は、表と集計を入力にする純粋な計算と、グローバルモデルの書換え・ID対応・ラウンドへの接続である。境界（表と、対の集計）がはっきりしている。

### Decision: 診断の記録を1種類にする

- **Selected Approach**: clientの評価1回につき、1件の記録（ラウンド、client、評価する側、対象、損失の統計、正誤の数、クラス別の数）。旧の3つの記録は、これから導ける（testが、導いたものを実旧と照合する）。
- **Rationale**: 旧の3つは、同じ評価を、粒度を変えて重ねて持っている。

### Decision: 損失の和をNumPyの配列の演算で求める

- **Rationale**: 旧と同じ値（末尾の桁まで）にするため。新実装は、ほかの場所でもNumPyを使っている。

### Synthesis

- **Build vs. Adopt**: 損失と予測クラスの計算は既存の部品。標本の選択、正誤の数え方、組の走査、通信量、集計は新しく書く。
- **Simplification**: Cached方式、非劣性の検証のための損失差、判定の種類による評価の切替は、移植しない。

## Risks & Mitigations

- 評価標本が少ない条件では、件数0の評価ばかりになる — 対照の条件で、評価標本の追加の件数を増やし、件数が正の評価を通す。経路の網羅を、testで確かめる。
- 旧の学習データの辞書の、読取りによるキーの追加 — 対照で差が出たら、旧の挙動として記録し、扱いを決める。

## References

- [server-model-registration-and-aggregation](../server-model-registration-and-aggregation/)、[global-model-distribution](../global-model-distribution/)、[post-aggregation-prediction-recalibration](../post-aggregation-prediction-recalibration/)。
