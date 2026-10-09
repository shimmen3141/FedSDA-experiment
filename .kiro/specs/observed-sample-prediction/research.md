# Research & Design Decisions

## Summary

- **Feature**: `observed-sample-prediction`
- **Discovery Scope**: Extension（移植済みの部品をつなぎ、既存の標本1件の処理へ接続する）
- **Key Findings**:
  - 最終構成（有効化方針`always`）では、旧の警報側の通知`_on_drift_alarm`・`_on_drift_resolution`は状態を変えない。移植する処理がない。
  - 旧`_record_prediction`は、3つの重み（global、Fixed-Share、真の概念別）を正規化して結合に使うが、更新へ渡す重みは、真の概念別だけ正規化前の値である。
  - 実旧clientのクラスを最終構成のクラスへ切り替えると、既存の標本処理のoracleの上で、実旧の`_record_prediction`を含む`process_one_step`を実行できる（下の「oracleの実行確認」）。

## Research Log

### 最終構成で通る分岐

- **Context**: 旧のmixinは、最終構成で使わない方式（クラス別の文脈、Meta、評価するモデルを絞る方式、回復期だけの有効化）も持つ。
- **Sources Consulted**: `tests/test_proposed_regression.py`（`ALGORITHM`）、`federated_drift_experiment/config.py`（110〜133行）、`clients/fedsda.py`（1287〜2048行）、`expert_routing.py`（521〜578行）、`clients/shared_backbone.py`（170〜185行）。
- **Findings**:
  - 最終3goldenは`soft_routing_context="switching"`、`soft_routing_activation_policy="always"`。`ROUTING_ACTIVE_SET_POLICY`は既定の`"all"`（`routing_active_set`はNone）、`ROUTING_ARCHIVE_SHADOW_DIAGNOSTICS`は既定のFalse。
  - `always`では`_use_soft_routing`は常に真（`active`が真で、`exit_due`は`drift_recovery`のときだけ真）。`_record_hard_prediction`は通らない。
  - `_on_drift_alarm`は`always`で即戻る。`_on_drift_resolution`は、再生の条件が`drift_recovery`を要求するので再生せず、空のtupleを空のtupleで上書きし、`resolve`は`always`で即戻る。
  - `switching`では`_prediction_probabilities`を呼ばないので、`history_routing_gate_open`は増えない（下書きの実行で0件を確認）。
  - 共有部を持つ構成の`_routing_scores`は、ID昇順で最初のモデルで特徴を1回計算し、各モデルの`forward_from_features`を呼ぶ。
- **Implications**: 警報側の通知は移植しない（UNPORTED-003）。feature名から通知を外した。予測は「保有する全モデル・Fixed-Shareの重み・常時」の1経路だけを移植する。

### 旧`_record_prediction`の、重みの扱い

- **Context**: 新の結合の部品は、総和が1の重みを要求する。旧は重みを「active集合へ制限して再正規化」してから使う。
- **Findings**:
  - 旧は、global・Fixed-Share・真の概念別の3つとも、`_restrict_routing_probabilities`（ID昇順の通常加算の総和で各値を割る）を通した値で結合する。active集合は保有する全モデルなので、制限は起きず、正規化だけが行われる。総和が0以下のときの均等割りは、3つとも総和がほぼ1なので到達しない。
  - 更新へ渡す重み: `expert_router.update(model_losses, proposal_probabilities)`は正規化後、`switching_expert_router.update(model_losses, switching_probabilities)`は正規化後、`oracle_concept_router.update(model_losses, oracle_concept_repository_probabilities)`は**正規化前**。
  - 旧の観測前の重みの取得順は、global→Fixed-Share→真の概念別。更新順は、global→真の概念別→Fixed-Share。3つは独立したownerなので、順序は数値に影響しない。
- **Implications**: 新は`normalize_model_prediction_weights`で3つを正規化して結合し、更新には、globalとFixed-Shareは正規化後、真の概念別は正規化前の重みを渡す。実旧との対照で、3つのownerの状態を標本ごとに照合する。

### 記録の項目と旧の列・計数の対応

- **Findings**（最終構成では、旧の「実際の予測」と「Fixed-Shareの予測」は同じ値である）:

| 旧 | 新の記録の項目 |
| --- | --- |
| `history_accuracy`、`history_routing_switching_correct`、`routing_diagnostics["mixture_correct_count"]`、`routing_switching_diagnostics["correct_count"]`・`["actual_correct_count"]` | 結合した予測が正しいか |
| `history_concept` | 真の概念ID |
| `history_model_id`、`history_routing_switching_leader_id` | 予測重みが最大のモデルのID |
| `history_routing_max_weight` | 最大の予測重み |
| `history_routing_effective_experts`、`history_routing_switching_effective_experts`、`routing_switching_diagnostics["effective_experts_sum"]` | 実効モデル数 |
| `history_routing_oracle_correct`、`["oracle_correct_count"]`、`["missed_oracle_count"]` | 結合した予測またはいずれかのモデルが正しいか |
| `history_routing_leader_correct`、`["leader_correct_count"]` | 予測重みが最大のモデルが正しいか |
| `routing_switching_diagnostics["global_correct_count"]` | globalの診断重みで結合した予測が正しいか |
| `history_routing_oracle_concept_correct`、`routing_oracle_concept_diagnostics` | 真の概念別の診断重みで結合した予測が正しいか |
| `["confidence_leader_correct_count"]`、`["confidence_leader_missed_oracle_count"]` | 確信度が最大のモデルが正しいか |
| `routing_class_diagnostics[クラス]` | 観測クラスごとに上の項目を数える |
| `history_routing_soft_active`、`soft_routing_activation.soft_sample_count` | 記録の件数（常時有効なので全標本） |

- **Implications**: 集計の計数のownerは作らず、記録から導く。対応づけはtest側に置く（tech.mdの「旧APIの受理で対応せず、テスト内で新しい記録と対応付ける」）。

### oracleの実行確認（REDより前）

- **Context**: 実旧の`_record_prediction`は、これまでのoracleでは空の関数に差し替えていた。実旧clientは`SharedBackboneClassConditionalESRFedSDAClient`を`__new__`で作っている（現行モデルだけで予測するクラス）。
- **Findings**: 2026-10-10、Windows基準環境で下書きのtest（commitしていない）を実行した。`build_sample_processing_oracle`の実旧clientの`__class__`を`ResidualAdapterRestartingSoftRoutingFedSDAClient`へ切り替え、差し替えた`_record_prediction`を外し、`config.SOFT_ROUTING_CONTEXT`を`"switching"`にして、予測が読む属性（`soft_routing_activation`、`switching_expert_router`、`oracle_concept_expert_routers`、`routing_leave_one_out_diagnostics`、標本ごとの列、集計の計数、再生用の2属性）を与えると、2値・多クラスの両方で、実旧の`process_one_step`を60標本続けて実行できた（保有モデル3つ、うち1つは一時ID。予測重みが最大のモデルが標本ごとに入れ替わる）。
- **Implications**: 対照testは、実旧の最終構成のクラスの実メソッドを使う。式をtestへ複製しない。真の概念IDがNoneの標本は、実旧が`int(None)`で失敗するので、対照の標本列では整数だけを使い、Noneは新実装だけのtestで確かめる。

## Design Decisions

### Decision: 予測を、標本1件の処理の最初の段としてつなぐ

- **Context**: 旧`process_one_step`は、位置の加算の直後、候補検証の観測より前に`_record_prediction`を呼ぶ。
- **Alternatives Considered**:
  1. `process_observed_sample`の中の最初の段にする。
  2. 予測と`process_observed_sample`を順に呼ぶ、別の関数を作る。
- **Selected Approach**: 1。`process_observed_sample`が、旧`process_one_step`の全体（計算量と所要時間を除く）に対応する。
- **Rationale**: 既存の対照test（実旧の標本処理と標本ごとに照合する60標本の軌跡）が、そのまま予測の対照にもなる。呼出し側（新client）が順序を誤る余地がない。
- **Trade-offs**: `process_observed_sample`の引数が2つ増え、既存のtestのoracleと順序のtestを更新する。

### Decision: モデル別の除外寄与の診断（旧`RoutingLeaveOneOutDiagnostics`）を後続のspecにする

- **Context**: 旧は、予測の直後に、モデルを1つ除いた結合の損失の差を、（モデル集合の世代、通信区間、モデル）ごとに集計する。予測と学習の判断へは戻らない（`routing_active_set`がNoneのため）。
- **Selected Approach**: 本specでは移植しない。予測の結果に、この診断の入力（モデル別の確率、予測重み、globalの診断重み）を含めて返すので、後続のspecは、同じ位置に呼出しを足せる。
- **Rationale**: 差分で求める除外後の結合と、世代・区間ごとの集計は、独自の状態と数値の繊細さを持つ（共通引継ぎ手順の「数値の一致が繊細な部分は小さいspecへ切り出してよい」）。最終3goldenは、この診断を比較しない。
- **Follow-up**: 再開案内の「次の候補」へ書く。

### Decision: 真の概念IDがない標本では、概念別の診断を行わない

- **Context**: 標本1件の処理は、概念IDが整数または「なし」の標本を受け取る（承認済みの契約）。旧の予測は、`int(concept_id)`で、Noneを例外にする。
- **Alternatives Considered**: 1. Noneを拒否する。2. 概念別の診断だけを省く。
- **Selected Approach**: 2。概念別の証拠を作らず、記録の該当項目を「なし」にする。
- **Rationale**: 真の概念IDは診断にだけ使い、予測・学習へ渡さない。旧の他の箇所（`_record_model_concept`）もNoneを「数えない」として扱う。標本1件の処理の契約を狭めない。
- **Trade-offs**: 旧にない経路なので、実旧との対照はできない。「概念IDがある場合と、概念別の項目以外は同じ結果になること」を新実装だけのtestで確かめる。

### Decision: 数値の計算を、最初の状態更新より前に置く

- **Context**: 観測前の重みの取得は、モデル集合が変わっていれば重み・証拠を初期化する（状態の更新）。旧は、取得の後でモデルを評価する。
- **Selected Approach**: 全モデルの出力・確率・損失・モデル別の正否・確信度を先に計算し、その後で重みを取得する。
- **Rationale**: 標本の中身（特徴の数、ラベルの範囲）とモデルの出力の不正が、どの状態も変える前に拒否される。モデルの評価は重みを読まず、重みの取得はモデルを読まないので、成功時の値は旧と変わらない（実旧との対照で示す）。

### Synthesis

- **Generalization**: 3つの重み（global、Fixed-Share、真の概念別）による結合と正否の判定は同じ形なので、module内の1つの補助へまとめる。
- **Build vs. Adopt**: 確率化・正規化・結合・クラスの決定・損失は`class_probability_calculations`、最大重みのモデルの選択は`FixedSharePredictionWeightController.select_maximum_weight_model_id`を使う。新しく書く数値は、実効モデル数、確信度、確信度が最大のモデルの選択だけ。
- **Simplification**: 集計の計数のowner、混合予測の有効・無効を持つowner、予測方式を差し替える抽象は作らない（最終構成は1方式）。

## Risks & Mitigations

- 既存の標本処理のtest（順序、拒否の文言）が、予測が先に拒否することで変わる — 実装のtaskで、既存のtestを要求に照らして更新する。期待値を緩めない。
- 実旧clientのクラスの切替えで、既存のoracleの他の挙動が変わる — 切替えで加わるoverrideは、予測、警報側の2つの通知（最終構成では状態を変えない）、切替の通知（既存のoracleが同じ実メソッドを呼んでいる）だけ。既存の対照の全項目が一致し続けることで確かめる。

## References

- [observed-sample-processingのdesign.md](../observed-sample-processing/design.md) 2節 — 旧`process_one_step`との対応。
- [UNPORTED-003](../../../docs/research/implementation-findings/unported-003-drift-recovery-prediction-mixture-activation.md) — 回復期だけの有効化を移植しない判断。
