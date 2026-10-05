# 設計: 保有モデルの学習バッチ抽出

## Overview
確定済みのモデル別標本列から一回共同更新に使うID付きbatchを抽出する。
標準Random.sampleとTorch.catを採用し、抽出器がRNGや標本ストアを新しく所有することはない。

## Boundary Commitments
### This Spec Owns
新入力/出力記録、全参加予定列の事前検査、保有/件数による参加選別、入力順の復元なし抽出/連結。
### Out of Boundary
FIFO/観測ストア所有、sample追加/ID統合/帰属判断、分類器/optimizer管理、学習/反復/診断/counter/通信、新全体run。
### Allowed Dependencies
- model_training_sample_records.py: dataclasses.dataclass、torch.Tensorだけ。
- held_model_training_batch_sampling.py: random.Random、torch.Tensor/cat/isfinite/float32/strided、同records moduleのObservedTrainingSample/ModelTrainingSampleCollection/SampledModelTrainingBatchだけ。
exact module/public symbolを一般stdlib許可より前にASTで固定。__future__.annotationsは許可。
global randomのsample/seed/getstateなど、Torch内部・nn/optim、NumPy、モデル、methods/FIFO/帰属、runtime/CLI/保存、legacyへ依存しない。
### Revalidation Triggers
population表現/参加順/抽出方式/shape/skip検査範囲/RNG所有を変えると後続学習・モデルID統合と終端RNGを再検証する。

## Architecture
records→sampler、上位がsampler出力IDからclassifier/optimizerを対応付け、既存ParticipatingModelTrainingBatchへ変換して共同更新へ渡す。
PendingTrainingAssignmentBufferの位置→観測解決→model別列構成も上位。新sampleへ旧concept metadataを持ち込まない。
ID登録表/バッチgenerator/プロトコル/新設定schemaを追加しない。batch_sample_countは具体値として渡す。

## File Structure Plan
新規src/federated_learning_experiments/learning/training/model_training_sample_records.py: 三つの借用入力/出力記録。
新規同held_model_training_batch_sampling.py: 一回の参加選別/抽出。
新規tests/refactoring/test_held_model_training_batch_sampling.py: actual旧oracleと拒否・明示上位接続。
変更tests/refactoring/test_single_run_dependency_boundaries.py: exact二module/symbolと注入test。
証拠は対象spec/.kiro/steering/roadmap.md。既存モデル/共同更新/FIFO/設定、旧production/goldenを変更しない。

## Components and Interfaces
```python
# すべてfrozen=True, kw_only=True、defaultなし。constructorは参照を束ねるだけ。
ObservedTrainingSample(input_features: Tensor, observed_class_labels: Tensor)
ModelTrainingSampleCollection(model_id: int, training_samples: tuple[ObservedTrainingSample, ...])
SampledModelTrainingBatch(model_id: int, input_features: Tensor, observed_class_labels: Tensor)

sample_training_batches_for_held_models(
    *, held_model_ids: frozenset[int],
    ordered_model_training_samples: tuple[ModelTrainingSampleCollection, ...],
    batch_sample_count: int,
    python_random_generator: Random,
) -> tuple[SampledModelTrainingBatch, ...]
```
標本/標本列を深いcopyで再構築しない。sampleの同値や同参照重複は禁止しない。

### 全検査を最初の抽出より前に終える
held_model_idsはexactfrozenset、ordered_model_training_samplesはexacttuple、batch_sample_countはexactint>0、generatorはexactRandom（SystemRandom/subclass/Noneを拒否）。
全保有IDと全collectionのmodel_idはexactbuiltinint（boolを拒否、負/大きいintを許可）。全collectionはexactModelTrainingSampleCollectionでID重複を拒否する。
未保有collectionのtraining_samplesは読まない。保有collectionのtraining_samplesはexacttupleを要求し、len<batch_sample_countなら内容を読まずskipする。
参加予定collectionの全elementがexactObservedTrainingSampleであることを確認する。各featureはCPUfloat32strided/notnested、rank2[1,D]、D>0、有限。列内Dが揃うことを確認。
labelは同CPU32/layoutの[1,1]で有限。クラス数/クラス値範囲/整数性は知らないため検査しない（二値softtarget維持）。参加collection間のD一致や共有モデルの期待Dも後続共同更新の責務。
Tensorはisinstanceで受理し、非contiguous/requires_gradの有無で拒否しない。サンプルTensor値/既存gradに書かない。Torch.catの標準autogradに従い、外側gradmodeを上書きしない。
検査helperは参加collection tupleを返し、samplingを呼ばない。

### 抽出
参加collection順にpython_random_generator.sample(collection.training_samples,batch_sample_count)を一回。
抽出されたsampleのinput_features/observed_class_labelsをそれぞれcatしSampledModelTrainingBatchを作り、参加順tupleで返す。
sort、set集合の反復で参加順を作る、sample列からの削除、独自sampler、global RNG scope、RNG copy/reseed/restoreは行わない。
空なら()。catは独立storageを返す（batchsize1も）。入力Tensorのautograd参照を持ち得るが入力値/gradは変更しない。

## Requirements Traceability
| 条件 | 設計/検証 |
|---|---|
| 1.1 | collection入力順とheld/正Nのfilter |
| 1.2 | 標準sample一回・位置復元なし・抽出順cat/同参照別位置 |
| 1.3 | 未保有payload未参照・不足内容未検査/no draw |
| 1.4 | emptytupleとRNG不変・heldonlyID非生成 |
| 2.1 | exact型/ID/重複/正countをdraw前検査 |
| 2.2 | 全参加populationの後段invalidもdraw前拒否 |
| 2.3 | 未保有/不足/参加で異なる検査範囲を明示 |
| 2.4 | borrowedtuple/Tensorrefs/value/grad保持・catfreshstorage |
| 2.5 | borrowedRandomだけ進む・PythonNP/Torch/ambient不変 |
| 3.1 | actual旧oracle pool/set algorithmとmulti-call終端RNG |
| 3.2 | test-onlyFIFO位置解決とNN/optimizer/joint旧学習接続 |
| 3.3 | exactAST/freshCPU/fulltests/旧golden不変 |

## Error Handling / 所有
契約外入力はfield/collection index/sample indexを含むValueErrorで最初のRandom.sample前に拒否。
出力recordは不変だがTensorはmutabilityを維持し、入力とは独立storage。samplerは永続stateを持たない。
抽出開始後の外部改変、標準generatorの内部破損、catの資源不足などに原子的rollbackを保証しない。
入力契約検査は全NN/ラベルの適合性や学習の数値安定性の保証ではない。

## Testing Strategy
Task1対象test missingmodule実RED→二productionGREEN。SimpleNamespace(models,train_data_store,batch_size)で旧_sample_training_batchesを直接呼ぶ。
globalPython RNGと借用Randomの開始stateを一致させ、実旧全ID/値/抽出順/終端stateをtorch.equalで比較しglobalstateはfinally復元。
5/100などpool/set両抽出経路、非昇順/負ID、skip/空/不足、3call、同参照重複/B1/B=Nを検証。標準sampleのwrapspyで参加者だけ一回を観測する。
Task2 test-only: 後段不正でもsample未呼出/借用RNG/全input保持、型/ID/tuple/finite/device/layout/shape/dims/後段不選択sampleも検査、skip不正payloadを無視、記録frozen/explicit、noncontiguous/grad/出力storage/ambient/PythonNP/Torchを確認。
Task3はtest-only上位接続。既存FIFOrelease/drain位置を観測dictへ解決してモデル別tupleを構成し、出力ID→NN/個別optimizer→共同更新をexplicit接続する。実旧batch抽出＋共同更新（loss/backward/optimizer実物）と複数step全値/grad/state/終端RNGを比較。モデルIDなどの対応はtestだけで旧に合わせる。
Task4 exactAST禁止/許可先行RED→GREEN、freshCPU抽出→共同更新smoke/nolegacyimport、全tests旧11/最終3golden、baseline差分/hash/UTF8/配置を確認。環境はdocs/experiments/refactoring-baseline.md。

## 主担当design保存前gate
PASS。全12条件/具体API/所有/検査範囲/数値とRNG順/実旧oracle/上位testseam/依存と配置を確認。新外部依存なし、一般registryやデータストア管理を取り込まない。
