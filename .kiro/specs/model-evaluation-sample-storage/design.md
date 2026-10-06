# 設計: モデル別評価標本の保持

revision: 1

## 所有・依存

新evaluation層の一storeがモデル別list構造と2固定件数だけを所有する。immutable評価record/collectionはpayloadを借用する。Randomは呼出し単位の借用、モデル/学習store/forward/設定globalを参照しない。
recordsはdataclasses.dataclassとtorch.Tensorだけ、storeはrandom.Randomと同階層の公開2型だけ（__future__.annotationsは許可）。generic stdlib許可より前のexact AST guardを追加する。package initは説明だけ。
新runtimeや設定集約へ組み込むのは後続。ここでは必須constructor引数で固定値を明示する。

## Interfaceと遷移

- frozen kw-only `ObservedEvaluationSample(input_features:Tensor,observed_class_labels:Tensor)`。
- frozen kw-only `ModelEvaluationSampleCollection(model_id:int,evaluation_samples:tuple[ObservedEvaluationSample,...])`。
- `ModelEvaluationSampleStore(*,maximum_stored_sample_count_per_model:int,added_batch_sample_count:int)`。前者exact int>0、後者exact int>=0。不正はValueError。
- `sample_and_append_model_evaluation_samples(*,model_id:int,evaluation_samples:tuple[ObservedEvaluationSample,...],python_random_generator:Random)->None`。ID/tuple/records/Randomを先に全検査（TypeError）。負IDなら戻る。非負はsetdefault空列、k=min(len,固定抽出件数)、k0戻る。それ以外はRandom.sampleを必ず実行しextend、超過末尾slice。
- `reassign_model_evaluation_samples_id(*,original_model_id:int,reassigned_model_id:int)->None`。両exact ID事前検査、あればpop代入。学習storeの同名責務と別owner、連結しない。
- `remap_model_evaluation_sample_collections(*,model_id_mapping:dict[int,int],python_random_generator:Random)->None`。全mapping/Random事前検査、一回get/元順extendして一時dictを作り、先順に超過列だけsample。最後に辞書交換。
- `snapshot_ordered_model_evaluation_samples()->tuple[ModelEvaluationSampleCollection,...]`。構造分離、payload/record共有。

非正常なMemoryError・private改変・並行変更・Random内部異常のrollbackは通常契約外。不正公開入力は状態/RNG変更前に拒否。負IDへの直接追加は無視するが、付替え/対応表のsigned IDは許す。

## File Structure Plan

|パス|責務|
|---|---|
|src/federated_learning_experiments/evaluation/__init__.py|層の説明のみ|
|evaluation/model_evaluation_sample_records.py|評価標本/一覧宣言|
|evaluation/model_evaluation_sample_store.py|保持/抽出/ID対応|
|tests/refactoring/test_model_evaluation_sample_storage.py|実旧対照/拒否/参照/損失接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|2module exact guardのRED/GREEN|
|対象spec・steering resume/roadmap|承認/実測/現在地|

## 検証trace

1.1–1.4: 固定値拒否、負/0/正ID・0/全件/部分抽出・超過を旧_store_evaluation_dataに同Random状態で対照。1件重複・同payload・empty初出順を含む。
2.1: 実旧confirmによる先既存/欠落/同ID/元欠落・empty上書き/順序/借用照合。
2.2–2.3: 実旧apply_server_mappingの空/連鎖/循環/多元先・負ID・先初出順、超過/非超過を対照しRandom終端一致。
3.1–3.2: constructor/操作全fieldの不正exact型・全mapping末尾不正を拒否、参照一覧/Tensor/RNG不変。過去snapshot/借用payload/opaque/float64/meta非検査。
3.3: exact AST禁止/許可テストRED→GREEN、fresh新CPU・旧非import、3globalRNG/torch defaults保持。
3.4: class2/4の実NNで保持→付替え→再編→cat→既存新bounded lossを旧NN per_sample_errorへexact照合。上位だけがTensorの有効性/NNを扱う。
全pytest旧11/最終3golden、Ruff/format/Pyright/pip、固定旧差分空、承認hash/sourcehash/JUnitを記録する。
