# 設計: Fixed-Share予測重み

## Overview

最終構成の予測重みをモデル学習から独立して検証できる数値部品を提供する。旧SwitchingExpertRouterの正常入力に対する状態遷移を保持する。

### Goals

予測前重み、観測後更新、再生、診断計数を一所有者に集め、旧oracleと各操作後に照合する。

### Non-Goals

torch出力混合、AdaHedge、activation、ClassESR、候補、学習、FIFO再計算、run接続・保存。

## Boundary Commitments

### This Spec Owns

一実体のモデル別重み、累積分散、集合変更・leader変更・再較正計数、入力検査とcopy返却。

### Out of Boundary

モデル・optimizer・ラベル・真の概念・割当先・FIFO・server・乱数は所有しない。同集合なら学習割当変更のhookを設けず証拠を保持する。

### Allowed Dependencies

stdlibと同機能の`prediction_combination_settings.PredictionCombinationSettings`だけを参照する。設定側のcore依存は保持。旧実装はテストだけ。

### Revalidation Triggers

時間尺度、入力・同点・再生・計数の意味、演算順、状態所有・依存方向を変えたら本specと後続接続を再検証する。

## Architecture

機能別設定に依存する単一FixedSharePredictionWeightController。上位RunSettingsや実行・モデルへ依存しない。interface/adapterは追加しない。

| 層 | 技術 | 役割 |
|---|---|---|
| 数値状態 | Python 3.13、stdlib math | 旧演算順保持 |
| 固定条件 | 既承認PredictionCombinationSettings | 明示時間尺度 |
| 検証 | pytest、旧SwitchingExpertRouter | テスト内の直接照合 |

## File Structure Plan

- 新規`src/federated_learning_experiments/methods/fedsda/prediction_combination/fixed_share_prediction_weights.py`: 状態・検査・更新。
- 新規`tests/refactoring/test_fixed_share_prediction_weights.py`: 公開契約・旧基準照合。
- 変更`tests/refactoring/test_single_run_dependency_boundaries.py`: controllerの設定依存だけ許可し、禁止/許可注入例を追加。
- 変更`.kiro/steering/roadmap.md`: 移植順・完成範囲。
- package再export、旧src・golden・既承認設定は変更しない。

## Requirements Traceability

| Requirement | 契約・検証 |
|---|---|
| 1.1, 1.2, 1.3 | 必須keyword設定、先行検証、copy・独立実体 |
| 2.1, 2.2, 2.3, 2.4 | ID昇順同期、一様化、負ID、同集合保持、不正集合 |
| 3.1, 3.2, 3.3, 3.4 | 旧演算順・制限・単一モデル、ID集合・確率検査 |
| 4.1, 4.2, 4.3 | 最大重み選択、最小ID計数、公開選択は状態不変 |
| 5.1, 5.2, 5.3, 5.4, 5.5 | 列順再生、集約計数、空列、明示reset、外部損失 |
| 6.1, 6.2, 6.3 | 全操作旧oracle、AST境界、部分移植の明示 |

## Components and Interfaces

### FixedSharePredictionWeightController

Intent: 損失による予測重みの遷移。Outbound: 既存設定（P0）。契約: Service / State。
公開名・引数・診断propertyの全一覧はnaming.mdを正本とする。

| API（引数keyword-only） | 入出力 |
|---|---|
| constructor | prediction_combination_settings: PredictionCombinationSettings |
| get_prediction_weights_before_label_observation | model_ids: Iterable[int] → dict[int,float] |
| update_weights_after_loss_observation | observed_losses_by_model_id / prediction_weights_by_model_id: Mapping[int,float] → None |
| select_maximum_weight_model_id（static） | prediction_weights_by_model_id, preferred_model_id: intまたはNone → int |
| reset_weights_after_aggregation | 入力なし → None |
| replay_observed_losses / replay_observed_losses_after_aggregation | observed_loss_sequence: Iterable[Mapping[int,float]] → None |

#### Preconditions / Postconditions

- IDはbuiltin int（bool除外）、負ID可。非空・重複なし。損失はbuiltin int/floatの有限値、bool除外。float変換失敗も先に拒否する。
- 確率は有限0～1。math.fsum総和の1との差1e-12以内。受理値は再正規化しない。
- constructorは設定型と全fieldの既存制約を再確認し、時間尺度の有限float変換を確認する。既定値は追加しない。
- updateは全検証・ID集合一致を確認してから集合同期。損失制限→期待損失→分散→累積分散→学習率→指数重み→正規化→shareの旧順序、通常sum、昇順、分散floor1e-12を保持する。
- 単一モデルは同期だけで終了。公開同率選択は優先IDが候補なら優先、なければ最小ID。優先ID不正型は拒否。
- 全再生列を検証copy後、非空なら重み・分散だけ消去して順番に取得・更新。行間の集合差は許し、計数は消去しない。
- 集約後再生は非空だけ回数+1・標本数+列長。明示resetは空状態でも回数+1。空再生は全状態不変。
- 各run/clientが独立実体を作る。入力・返却dictは内部へ結合しない。診断propertyは読取専用。thread間共有は想定しない。

## Error Handling

理由付きValueError/TypeError、既存設定違反はRunSettingsValidationErrorで拒否し、重み・分散・全計数を保持する。iterable列挙の例外も状態変更前に伝播する。資源枯渇へのtransaction保証は追加しない。

## Testing Strategy

- 各操作後に旧oracleと重み・分散・4計数の完全一致。損失制限・最良モデル交代・ゼロ分散・単一モデル（3.1–3.3、4.2、6.1）。
- 初回・負ID・列挙順・集合差・同率・copy・独立実体・乱数非変更（1.1–1.3、2.1–2.4、4.1、4.3）。
- 通常/集約再生、途中集合差、空列、明示reset、不正後段全状態不変（5.1–5.5、1.2、3.4）。
- 禁止import注入、新src全走査、既存schema・既存golden・最終構成golden。新全体runは未完成と記録する（6.2、6.3）。
