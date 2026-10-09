# Research & Design Decisions

## Summary

- **Feature**: `server-model-registration-and-aggregation`
- **Discovery Scope**: Extension（サーバ側の最初のspec。clientの既存のownerを読む）
- **Key Findings**:
  - 旧の登録と集約は、他の段（クロス評価、配布）と独立したメソッドで、配布なしでも続けて実行できる。
  - 旧のグローバルモデルは、モデルIDごとの完全なパラメータで、集約の対象外のモデルは古い共有部のまま残る。新も、モデルIDごとの完全なパラメータで持つ。
  - 採用の後に現行モデルが非負のIDへ戻ると、採番だけが行われる（LEGACY-016）。

## Research Log

### 旧のサーバのラウンドと、specの分け方

- **Sources Consulted**: 旧`servers/base.py`、`servers/fedsda.py`（12〜131行）、`servers/shared_backbone.py`、`experiment.py`（94〜118行、337〜358行）、`model_lineage.py`、`models.py`（272〜288行、402〜408行）。
- **Findings**: 再開案内の「次の一手」に、4つのspecへの分け方を書いた。本specは(1)。
- 旧の登録: `request_new_model_id`（1から。初期モデルは0）、`record_model_registration`（来歴の上書き）、clientの`confirm_model_registration`。通信量は足さない。
- 旧の集約（共有部を持つサーバ）の演算: clientを登録順に見る。参加モデルは、対象のIDの昇順で、学習データが1件以上のもの。共有部の和は、最初のclientで「値×重み」から始め、以後「和＋値×重み」。概念固有部も同じ。最後に「和÷重みの合計」。重みはPythonのint。
- 統計: `stats['mean'] * stats['n']`の和と`n`の和（client順）。移植済みの`aggregate_participating_client_loss_means`と同じ演算である。
- 通信量: `comm_models_up += len(assigned)`、共有部と概念固有部は`record_parameter_transfer("up", …)`（値の数は`numel`、バイト数は`numel × element_size`）。

### 新側の既存部品

| 用途 | 部品 |
| --- | --- |
| clientの正式IDの確認 | `confirm_held_model_registration`（held-model-registration-confirmation） |
| 分類器のパラメータの写し | `snapshot_classifier_parameters`（名前→独立したtensor） |
| 損失平均の集約 | `aggregate_participating_client_loss_means`（server-loss-mean-aggregation） |
| clientの状態 | `FedsdaRunClient.owners`（保有モデル、学習データ、損失統計、送信保留） |

- 新のパラメータ名: 共有部は`feature_extractor.`で始まり、概念固有部はそれ以外（アダプタ、分類層）。分類器の`state_dict`の順は、共有部→アダプタ→分類層で、旧の「共有部＋概念固有部」の順と同じ。

### oracleの実行確認（REDより前）

- 2026-10-10、Windows基準環境で下書きのtest（commitしていない）を実行した。実旧の事前学習の結果を、`SharedBackboneFedSDANoCachedServer`へ`register_model_params(0, …)`・`register_model_stats(0, …)`で登録し、実`__init__`で作ったclient 3つを`register_client`して、12ラウンド、各ラウンドで「全clientの標本処理（標本ごとにclient順）→保留中の学習→状態の報告→`_register_new_models(t)`→`update_global_models(対象のID)`→送信待ちの進行」を実行できた。配布は行わない。一時IDのモデルの登録（ラウンド7）と、非負のIDへ戻った後の登録（ラウンド8。採番だけ）を観測した。

## Design Decisions

### Decision: グローバルモデルは、モデルIDごとの完全なパラメータで持つ

- **Alternatives Considered**: 1. 共有部を1つと、モデルIDごとの概念固有部に分けて持つ。2. 旧と同じく、モデルIDごとに完全なパラメータを持つ。
- **Selected Approach**: 2。
- **Rationale**: 旧は、集約の対象外のモデルに古い共有部が残り、通信量の計上と配布が「最初のグローバルモデルの共有部」を読む。1にすると、この違いを別に再現する必要がある。構造の整理は、挙動を変えない移植の後で判断する（改善候補として記録する）。

### Decision: 集約は、全部の計算の後で、まとめて反映する

- **Context**: 旧は、clientを見ながら通信量を足す。
- **Selected Approach**: 和・平均・統計・通信量の増分を先に計算し、最後に、通信量→グローバルモデル→統計の順に反映する。
- **Rationale**: 計算の途中の失敗（clientどうしでパラメータの形が違うなど）で、何も変えない。成功時の値は旧と同じ（通信量は整数の和で、順序に依らない）。

### Decision: 登録と集約は、clientの列を受け取る関数にする

- **Selected Approach**: runtimeの関数2つ。サーバの操作をまとめるclass（実行の枠の`RunServerOperations`の実体）は、配布・統合がそろう、全体runを接続するspecで作る。
- **Rationale**: 今は、呼ぶ順を決める実体がまだ要らない。対照testは、旧のメソッドを1つずつ呼ぶのと同じ粒度で照合する。

### Synthesis

- **Build vs. Adopt**: 正式IDの確認、パラメータの写し、損失平均の集約は既存の部品を使う。新しく書く数値は、パラメータの重み付き平均だけ。
- **Simplification**: クラスタリングの診断の記録は、来歴のownerへ入れない（後続のspecで足す）。

## Risks & Mitigations

- 重み付き平均の演算の順が旧と違うと、末尾の桁が違う — 旧と同じ式の形（値×重み、和＋値×重み、和÷合計）と、同じclientの順で計算し、全パラメータを実旧と照合する。
- 正式IDの確認の旧の挙動（LEGACY-016）が、後続の配布で別の食い違いを生む — 本specは挙動を維持し、記録する。配布のspecで、一時IDのまま残るモデルの扱いを確かめる。

## References

- [LEGACY-016](../../../docs/research/implementation-findings/legacy-016-registration-id-consumed-without-model.md)
- [held-model-registration-confirmation](../held-model-registration-confirmation/)、[server-loss-mean-aggregation](../server-loss-mean-aggregation/)
