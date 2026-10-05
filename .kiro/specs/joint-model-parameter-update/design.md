# 設計: 一回の共同モデルパラメータ更新

## Overview
確定済みの参加バッチ列から共有forward・共同損失・backward・stepを一回だけ実行する。
共有/個別optimizerは借用する。標準Torch損失/optimizerを使い、独自optimizerやregistryを作らない。

## Boundary Commitments
### This Spec Owns
参加記録の宣言、全入力の更新前検査、一回の標本数加重共同更新、更新前共同損失の返却。
### Out of Boundary
参加抽出・epoch/反復・optimizer生成/所有/reset/復元・モデル登録・候補進行・通信・PCGrad・診断/counter・新全体run。
### Allowed Dependencies
- participating_model_training_batch.py: dataclasses.dataclass、torch.Tensor、torch.optim.Optimizer、learning.models.residual_adapter_classifier.ResidualAdapterClassifier。
- joint_model_parameter_update.py: torchのTensor/cat/isfinite/no_grad/is_grad_enabled/float32/strided、torch.nnのParameter/BCELoss/CrossEntropyLoss、torch.optimのOptimizer/Adam/SGD、同trainingのLocalTrainingSettings/ParticipatingModelTrainingBatch、learning.modelsのSharedFeatureExtractor/ResidualAdapterClassifier。
exact module/public symbolだけを一般stdlib許可の前にAST検査する。__future__.annotationsは許可。
生成builderへの依存は不要。math/random/NumPy/legacy、runtime/CLI/保存/上位method、Torch内部や別optimizerへの依存を追加しない。
### Revalidation Triggers
参加記録・loss形式・入力型/shape・パラメータ参照契約・操作順・共有凍結を変えると後続学習/候補進行と数値照合を再検証する。

## Architecture
分類器/optimizer/特徴/ラベルを一つの参加記録に束ねる。モデルID・バッチ抽出器・可変client contextは渡さない。
NNはforwardを提供し、損失とstepはtrainingが担当する。上位が共有optimizerを一つ所有し、空共有部なら生成しない。
既存LocalTrainingSettingsの正式二値を再検証するが、その名前・設定値は変更しない。

## File Structure Plan
新規src/federated_learning_experiments/learning/training/participating_model_training_batch.py: frozenな借用入力記録。
新規同joint_model_parameter_update.py: 事前検査と一回更新。
新規tests/refactoring/test_joint_model_parameter_update.py: 実旧経路・拒否・正常拡張の比較。
変更tests/refactoring/test_single_run_dependency_boundaries.py: exact二moduleのguard/禁止許可test。
証拠は対象specと.kiro/steering/roadmap.md。旧production/golden/既存モデル/設定は変更しない。

## Components and Interfaces
```python
@dataclass(frozen=True, kw_only=True)
class ParticipatingModelTrainingBatch:
    classifier: ResidualAdapterClassifier
    concept_specific_parameter_optimizer: Optimizer
    input_features: Tensor
    observed_class_labels: Tensor

perform_joint_model_parameter_update(
    *, local_training_settings: LocalTrainingSettings,
    shared_feature_extractor: SharedFeatureExtractor,
    shared_parameter_optimizer: Optimizer | None,
    participating_training_batches: tuple[ParticipatingModelTrainingBatch, ...],
    update_shared_features: bool,
) -> float | None
```
記録のconstructorは参照を束ねるだけ。更新関数が公開fieldを毎回検査する。default値を持たず、テンソル/モデル/optimizerはコピーしない。

### 更新前検査
1. exact LocalTrainingSettingsを公開fieldから再構築してforged値を拒否。exact SharedFeatureExtractor/exacttuple/exactboolと記録exact型を検査。
2. 空tupleならNone。モデル/optimizerの詳細検査や勾配状態の要求なし。
3. 非空なら外側のgrad有効を要求。共有extractorの既存validate_structureを同dimsで使用。
4. 各記録の分類器はexact ResidualAdapterClassifier、同一extractor参照、class_countはexactint>=2。分類器と個別optimizerのidを重複させない。
5. 入力TensorはCPUfloat32strided/notnestedのrank2[N,D]、N>0、D=extractor.input_feature_count、有限。ラベルは同条件[N,1]で有限。二値[0,1]、多クラス整数[0,class_count)。入力Tensorのcontiguousは要求しない。
6. 共有tupleはextractor.parameters()、個別tupleはadapter.parameters()→classification_layer.parameters()。全Parameterはexact Parameter/CPUfloat32strided/notnested。個別はrequires_grad=True、共有は更新有効時のみTrue必須。個別集合の重複/共有集合との重複を拒否。
7. optimizerはexactAdam/SGD。全param_groupsのparamsを入力順に平坦化し、期待tupleと長さ/id/順序が同じことを検査する。複数groupは列が一致すれば受理し、設定やstateを再作成しない。共有tuple空なら共有optimizerはNoneのみ、非空ならNoneを拒否。共有凍結時も同じ参照列契約。
全検査をzero_gradより前に完了する。Parameter値/optimizer内部stateの有限性や外部によるNN内部の任意改造を完全検査する責務は追加しない。

### 更新操作
存在する共有optimizer.zero_grad()→各個別optimizer.zero_grad()、cat(input_features)。
共有更新有効は共有extractorを一回呼ぶ。無効はこの呼出だけno_gradで囲む。
入力順にNでsliceし、classifier.forward_from_shared_features、二値BCELoss(prediction,labels)、多クラスCrossEntropyLoss(logits,labels.view(-1).long())。
sum(loss_i*N_i)/total_sample_countを入力順に計算し、joint_loss.backward()。
共有更新有効かつoptimizer存在ならstep、その後個別stepを入力順に実行。float(joint_loss.item())を返す。
平均勾配ベクトル再計算やIDsortはしない。旧mean診断は更新値に関わらないため移植しない。

## Requirements Traceability
| 条件 | 設計/検証 |
|---|---|
| 1.1, 1.3 | 明示API、設定public copy、exact型 |
| 1.2 | 空列None/値gradstate不変 |
| 2.1, 2.2 | batchshape/finite/labels/共有参照検査 |
| 2.3 | 全Parameter/optimizer参照・重複・requiresgrad |
| 2.4 | 空共有はNoneのみ・個別更新 |
| 3.1 | zero順/concat/共有forward一回の実観測 |
| 3.2 | BCEsofttarget/CE整数と入力順の加重loss |
| 3.3 | backward一回/step順/実旧stategrad/返却loss |
| 3.4 | no_grad特徴・共有state不変/個別step |
| 4.1 | 後段不正でもzero前拒否・deep state保持 |
| 4.2 | 入力値/構造/設定/RNG/外側grad保持 |
| 4.3 | post-start失敗rollback非保証を明示 |
| 5.1 | 実旧共同経路複数stepOracle |
| 5.2 | 空共有の独立期待値、旧生成失敗とは区別 |
| 5.3 | exactAST/freshCPU/fulltests旧11最終3 |

## Error Handling / 状態
事前契約違反はfield/参加indexの分かるValueError。zero_grad/stepをspyし、後段不正でも全参照値/grad/state不変を検証する。
正当な更新は借用Parameterのgrad/valueとoptimizer stateを書き換える。入力テンソル値を変更しないが、requires_gradを持つ入力にはbackwardによるgradが生じ得る。
演算開始後の例外はrollbackしない。非有限パラメータや演算overflowの安全保証は本spec外。
no_grad contextは共有特徴計算だけで必ず復元。RNG/defaultdtype/deviceを設定しない。

## Testing Strategy
Task1はmissing-module実RED→GREEN。実旧ResidualAdapterMLPと共有backbone、実optimizer、固定batchを返す最小clientで_train_heads_togetherをcount_multiplier=1呼出。
loss/backward/stepを置換せず、batch抽出/計算記録だけtest側固定。新旧state名はtest-only対応。初期同値/複数stepの全値・grad・optimizerstateをtorch.equalで比較する。
二値/多クラス、単一/複数、不均等N、参加ID非昇順、Adam標準/AMSGrad/SGD、共有有効/凍結を検証。共有forwardとzero/stepの実hook観測、共同lossは旧実lossのhookで取得。
Task2は不正type/forged/後段shape/finite/labels/参照逆順/重複/grad無効の更新前拒否、空列、softtarget、非contiguous、空共有の独立勾配/個別更新、RNG/ambientcontext/borrowed参照を検証する。
Task3はexactAST注入実RED→GREEN、旧importなしfreshCPU joint update、全testsと旧golden無変更、hash/UTF8/diffを確認する。
環境はdocs/experiments/refactoring-baseline.md。新全体run移植の完成証拠とは区別する。

## 主担当design保存前gate
PASS。全numeric要件、具体所有/依存/配置/空列/拒否順/演算順/実旧oracle/正常拡張/検証順を確認した。独自losswrapperやoptimizerlifecycle abstractionを追加しない。
