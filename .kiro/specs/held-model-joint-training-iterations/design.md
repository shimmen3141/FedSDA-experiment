# 保有モデルの共同学習反復の設計

## Overview
最終FedSDAの明示モデル別標本から、抽出→ID対応→一回共同更新を指定回数繰り返す。
完成済み二部品を接続し、モデル・optimizer・Randomの所有を外側に残す。
新しい選択肢・設定・汎用callback/registryを追加しない。

## Boundary Commitments
### This Spec Owns
保有ID→classifier/個別optimizerの借用記録、指定回数の反復と成功更新の損失tuple。
対応表の構造/IDと非負回数の検査。
### Out of Boundary
標本の追加/ストア/帰属/FIFO、保有モデル変更、optimizer生成/reset、回数算出とpending/interval、
候補訓練、通信、時間・counter・勾配診断、PCGrad、新clientと新全体run。
### Allowed Dependencies
記録moduleはdataclasses.dataclass、torch.optim.Optimizer、モデル公開ResidualAdapterClassifierだけ。
反復moduleはrandom.Random、torch.optim.Optimizer、モデル公開SharedFeatureExtractor、
兄弟公開HeldModelTrainingBinding、ModelTrainingSampleCollection、ParticipatingModelTrainingBatch、
LocalTrainingSettings、sample_training_batches_for_held_models、perform_joint_model_parameter_updateだけ。
両moduleで__future__.annotationsも許可。一般stdlib許可に先行するexact AST gateを追加する。
Torch演算・private helper・global random・methods/上位runtime/config・legacyは許可しない。
### Revalidation Triggers
反復順、ID対応、上流入力/出力、optimizer寿命・RNG所有の変更時は本specの旧反復照合と上流二specを再確認する。

## Architecture
外側の算出済み回数/保有対応/標本列→反復executor→毎回sampler→ID対応→joint update。
反復executorは標本順を保ち、対応表はlookupだけに使う。抽出結果を次の回へ再利用しない。
オブジェクトをコピーしないためoptimizerのmomentum/Adam stepとモデル共有参照が継続する。

## File Structure Plan
- 新規src/federated_learning_experiments/learning/training/held_model_training_binding.py: IDと学習実体の不変借用記録。
- 新規同held_model_joint_training_iterations.py: 対応表検査と反復。
- 新規tests/refactoring/test_held_model_joint_training_iterations.py: 実旧の反復oracle、拒否/空/0/状態/順序。
- 変更tests/refactoring/test_single_run_dependency_boundaries.py: exact依存と許可/拒否テスト。
- 対象specの記録と.kiro/steering/roadmap.mdのみ。旧production/goldenと既存設定/部品は変更しない。

## Components and Interfaces
```python
@dataclass(frozen=True, kw_only=True)
class HeldModelTrainingBinding:
    model_id: int
    classifier: ResidualAdapterClassifier
    concept_specific_parameter_optimizer: Optimizer

def perform_held_model_joint_training_iterations(
    *, requested_joint_update_iteration_count: int,
    held_model_training_bindings: tuple[HeldModelTrainingBinding, ...],
    ordered_model_training_samples: tuple[ModelTrainingSampleCollection, ...],
    batch_sample_count: int, python_random_generator: Random,
    local_training_settings: LocalTrainingSettings,
    shared_feature_extractor: SharedFeatureExtractor,
    shared_parameter_optimizer: Optimizer | None,
    update_shared_features: bool,
) -> tuple[float, ...]: ...
```
constructorは参照を束ねるだけでdefaultや検査を持たない。
回数はexact builtin int>=0（bool/subclass拒否）、0なら他引数を読む前に()。
正回数では対応表をexact tuple、各要素exact HeldModelTrainingBinding、ID exact int/一意として事前検査する。
負/巨大IDも許可し、IDでsortしない。_index_held_model_training_bindingsが順序付きdictを返す。
classifier/optimizerの詳細は参加時にjoint updateが検査し、未参加のpayloadを検査しない。

各回、対応dictのkeyからfrozenset保有IDをsamplerへ渡す。samplerの全候補preflightとskipを再利用する。
空ならcontinue。非空なら抽出順にIDをlookupしParticipatingModelTrainingBatchを作る。
一回joint updateへ渡し、返ったfloatをcompleted_joint_update_lossesへ追加する。
非空参加でNoneを返す場合は上流契約違反としてRuntimeError。通常の公開上流では起きない。
終端でtupleへ変換する。状態の保存/復元、generator/optimizer生成、独自Tensor演算を追加しない。

## Error Handling / 所有
回数/対応表の契約違反は最初の抽出より前にValueError。どの引数・位置/IDかを説明する。
上流の抽出検査・更新検査・演算例外はそのまま伝える。
モデルID→分類器の共有参照/optimizer group適合やgrad modeはjoint updateの既存責務。
抽出後の失敗でRNGや完了済み更新を巻き戻さない。実行途中の外部改変も保証しない。
empty時はjoint updateを呼ばないので、使わない共有部/設定/optimizerの検査も行わない。

## Requirements Traceability
| 条件 | 設計/検証 |
|---|---|
| 1.1 | count回ループ・毎回新抽出と実旧複数反復 |
| 1.2 | 標本順ID lookup・対応表逆順でも一致 |
| 1.3 | empty continue・RNG/model/optimizer無変更 |
| 1.4 | 0即return・他入力未アクセス |
| 1.5 | 実旧loss hookを反復単位へ集計・順序tuple |
| 2.1 | exact非負回数preflight・bool/負/小数/subclass拒否 |
| 2.2 | exact対応表/record/IDと重複preflight |
| 2.3 | 既存optimizer/Randomを全回借用し旧終端state完全一致 |
| 2.4 | sample入力snapshot、失敗時先行更新/RNG非rollback |
| 2.5 | frozen/empty共有/環境状態と全上流契約 |
| 3.1 | 実旧_train_heads_togetherと全parameter/grad/state/RNG照合 |
| 3.2 | exactAST/freshCPU旧非import |

## Testing Strategy
Task1はmissingmoduleの実RED→二module GREEN。既存test helper build_joint_update_oracle_pairで同一初期NN/optimizerを構築し、
旧実samplerと旧_train_heads_togetherを一回の複数反復呼出しとして使う。実旧forward/loss/backward/optimizerは置換しない。
旧samplerの戻り値とloss hookを観測し、各回の全sample/ID/損失と終端parameter/grad/optimizer/RNGを完全照合する。
二値/多クラス、Adam/AMSGrad/SGD、共有更新/凍結、反復0/1/4、batch1/3、非昇順ID・逆対応表・skip/不足を含む。
Python global RNGを借用Random開始stateへ合わせて実旧だけ実行しfinallyで復元する。抽出の余分なoracle呼出しをしない。

Task2は拒否/0/empty/未参加payload/ambient保持/借用frozen記録のtest。
後段のjoint update検査失敗で抽出RNGが進み、前に完了した更新が残ることを明示テストする。
空共有は新featureの単独smoke/既存NN構成で検証する（旧LEGACY010は修正しない）。

Task3はexact AST gateを禁止importの実REDから追加、fresh CPUの新packageだけによる反復smoke、
対象pytest/全pytest（基準ローカル旧11・最終3goldenを含む）・Ruff check/format・Pyright・pip check、
固定旧差分/hash/UTF8/配置/要件表を確認する。CIのホストgolden診断をローカル一致の代替にしない。

## 主担当design保存前gate
PASS。軽量discovery、既存部品採用、具体API/配置/型/所有/全12要件/実旧oracle/失敗境界を確認。
新ライブラリ・汎用層・設定追加は不要。レビュー承認はLunaの独立確認後に保存する。
