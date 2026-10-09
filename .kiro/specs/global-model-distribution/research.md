# Research & Design Decisions

## Summary

- **Feature**: `global-model-distribution`
- **Discovery Scope**: Extension（移植済みの部品をつなぎ、足りないownerの操作を足す）
- **Key Findings**:
  - 旧の受取りの7つの段のうち、5つには移植済みの部品がある。足りないのは、保有モデルの作り直しと、ownerの「全部を置き換える」操作である。
  - 旧は、配布のたびに、非負のIDの保有モデルを新しく作る。torchの乱数を消費し、optimizerの状態が初期化される。
  - 旧は、学習率の設定を2つ使い分ける。新の束には1つしかない。
  - つなぎ先の共有部とそのoptimizerは、配布のたびに別のオブジェクトになる。clientは、共有部のoptimizerの状態を固定の参照で持っているので、保持の形を変える必要がある。

## Research Log

### 旧の受取りと、対応する新部品

- **Sources Consulted**: 旧`clients/base.py`（662〜727行）、`clients/fedsda.py`（452〜460行）、`clients/shared_backbone.py`（52〜80行）、`models.py`（180〜200行、246〜270行、291〜324行）、`servers/shared_backbone.py`（132〜152行）。
- **Findings**:

| 旧の段 | 新部品 |
| --- | --- |
| (1)現行モデルのIDが変わるなら、`mapping_change_positions`と`server_merge`の適応イベント | 適応記録のownerへ、結果種別と位置の列を足す（本spec） |
| (2)統計の付け替えと、サーバの統計での補完 | `select_loss_statistics_after_model_id_mapping`。結果を置く「全置換え」の操作を、損失統計のownerへ足す（本spec） |
| (3)評価標本の付け替えと抜出し | `ModelEvaluationSampleStore.remap_model_evaluation_sample_collections` |
| (4)学習データと計数の付け替え | `ModelTrainingSampleStore.remap_model_training_sample_collections`、`ModelTrainingAndAssignmentCountsStore.remap_model_training_and_assignment_counts` |
| (5)保有モデルの作り直し | 分類器の生成、`ParameterOptimizerState`の生成。保有モデルの「全置換え」の操作を、registryへ足す（本spec） |
| (6)共有部へのつなぎ直し | `reconnect_held_models_to_shared_feature_extractor` |
| (7)現行モデルのIDの付け替え | `CurrentTrainingModelAssignment.remap_current_training_model_id` |

  - (5)の旧: `m = self._new_model()`（`model_cls()`。共有部→アダプタ→分類層を初期化し、2つのoptimizerを`BASE_LR`で作る）、`m.set_params(params)`（値の上書き）。グローバルモデルの辞書の順。
  - (6)の旧: つなぎ先は、非負のIDが最小のモデル。ほかのモデルは`attach_backbone`で、共有部を替え、概念固有部のoptimizerを`NEW_MODEL_LR`で作り直す。一時IDのモデルも同じ。
  - 作り直したモデルのうち、つなぎ先以外の共有部（とそのoptimizer）は、つなぎ直しで捨てられる。値として使われるのは、非負のIDが最小のグローバルモデルの共有部だけである。
  - Cached方式のための控え（`cached_global_model_params`）は、最終構成のサーバ（NoCached）が読まないので、移植しない。
- **Implications**: 受取りは、既存の部品を旧の順に呼ぶ関数にし、足りない操作をownerへ足す。

### 共有部のoptimizerの状態の持ち方

- **Context**: clientのownerの記録は、共有部のoptimizerの状態を、固定の参照で持つ。clientは、共有部を「初期モデルの写しの分類器の共有部」として読んでいる。
- **Findings**: 配布の後、つなぎ先の共有部と、そのoptimizerの状態は、新しいオブジェクトになる。古い参照のまま学習すると、保有モデルがつながっていない共有部を更新する。
- **Implications**: 共有部のoptimizerの状態を、置き換えられる保持者で持つ。共有部と、候補の構造の参照は、呼出しのたびに、現在の学習帰属のモデルの分類器から読む（旧の`_shared_backbone()`と同じ）。

### 学習率の設定

- **Findings**: 旧のモデルの生成は`BASE_LR`、`reset_optimizer`と`attach_backbone`は`NEW_MODEL_LR`を使う（datasetの定義に学習率があれば、どちらもそれを使う）。新の束の`parameter_optimizer_settings`は、候補の生成に使われている（`NEW_MODEL_LR`に当たる）。初期モデルのoptimizerは、事前学習が自分の引数の設定で作る（`BASE_LR`に当たる）。
- **Implications**: 束へ、作り直すモデルのoptimizerの設定（`BASE_LR`に当たる）を足す。対照testは、2つの学習率を違う値にした条件を含める。

### oracle

- 登録と集約の対照（`build_server_round_oracle`）に、実旧の`broadcast_models(id_mapping)`を足す。ID対応を与える場合は、実旧のclientの`apply_server_mapping(id_mapping, global_models, global_stats)`を直接呼ぶ（統合のspecを待たずに、受取りの全分岐を照合できる）。

## Design Decisions

### Decision: 集約後の再較正を、次のspecに分ける

- **Context**: 旧の`run_round`は、配布の直後に、全clientの予測重みと診断証拠を再較正する。
- **Selected Approach**: 本specは配布と受取りまで。再較正は次のspec。
- **Rationale**: 再較正は、予測側の状態（Fixed-Shareの重み、globalと真の概念別の診断証拠）の、損失の列の再生で、受取りが変える状態（統計、標本、保有モデル）と重ならない。診断証拠のownerへ、集約後の再生と再始動の操作を足す必要があり、数値の照合の対象も別である。

### Decision: 受取りは、計算を先に済ませてから、ownerを更新する

- **Selected Approach**: 検査→統計の選択→新しい分類器とoptimizerの状態の生成（torchの乱数を消費）→ownerの更新（適応記録、統計、評価標本、学習データ、計数、保有モデル、共有部のoptimizerの状態、現在の学習帰属）。
- **Rationale**: 配布されたパラメータが分類器の構造に合わない、などの失敗で、ownerを変えない。旧は、統計・標本・計数を更新してからモデルを作るが、モデルの生成はそれらを読まず、それらの更新はtorchの乱数を使わないので、成功時の値と乱数の消費は変わらない（評価標本の抜出しは、借りたPythonの乱数を使う）。

### Decision: 受取りで、現在の学習帰属が保有されなくなる入力を拒否する

- **Context**: 旧は、配布にないIDが現行のままでも受け取り、次の標本処理で失敗する。
- **Selected Approach**: 受取りの後の現在の学習帰属（ID対応を適用したもの）が、配布されたIDか、残る一時IDでなければ、何も変えずに拒否する。

### Synthesis

- **Build vs. Adopt**: 段(2)(3)(4)(6)(7)は既存の部品。新しく書くのは、保有モデルの作り直しと、全体の順序。
- **Simplification**: 配布の「差分だけを送る」などの最適化はしない（旧と同じく、毎回全部を配る）。

## Risks & Mitigations

- 既存の完了済みのmodule（registry、損失統計のowner、適応記録のowner、clientとその束）を変更する — 追加だけにし、既存の操作と契約を変えない。既存のtestが全部通ることを確かめる。束への必須のfieldの追加で、束を作っているtestと共用scriptを直す。
- 候補検証を保持している間の配布で、sessionが古い分類器を参照する — 実旧との対照に、候補検証を保持したままラウンド境界を越える条件を含める。

## References

- [server-model-registration-and-aggregation](../server-model-registration-and-aggregation/) — サーバのowner。[held-model-shared-feature-reconnection](../held-model-shared-feature-reconnection/)、[model-id-mapped-loss-statistics-selection](../model-id-mapped-loss-statistics-selection/)。
