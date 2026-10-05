# 設計: モデル別学習標本の保持

## Overview / Goals
割り当て済み標本の構造だけを所有し、既存samplerへ順序付きsnapshotを提供する。
旧append/extend/一回ID対応の正常系を保存し、検証エラーで部分的な追加を残さない。

## Boundary Commitments
### This Spec Owns
モデル初出順、標本追加順、空列、重複、snapshotの構造分離、一回ID対応の連結。
### Out of Boundary
Tensorの内容/clone、統計、概念診断、容量、評価標本、NN/optimizer、乱数、正式登録pop/上書き、クライアント/全体run。
### Allowed Dependencies
productionは__future__.annotationsと同階層model_training_sample_recordsの公開ObservedTrainingSample/ModelTrainingSampleCollectionのみ。
record経由の借用Tensor型に依存するがtorchへ直接import/演算しない。
上位が保持器を呼び、保持器はsamplerや上位を呼ばない。新依存ライブラリなし。
### Revalidation Triggers
追加/空列/ID対応/型契約/参照所有/順序変更でsampler・反復executor・将来clientの照合を再実施する。

## Architecture / Contracts
ModelTrainingSampleStoreは引数なしで空dictを所有する。一つの可変状態のみ。
- append_model_training_samples(*,model_id:int,training_samples:tuple[ObservedTrainingSample,...])->None:
  ID/exact tuple/全record型を確認後、setdefault(...,[]).extend。空追加も空列を作る。
- snapshot_ordered_model_training_samples()->tuple[ModelTrainingSampleCollection,...]:
  元dict順で新record/新tupleを作る。モデル一覧や標本列の構造は後続変更から独立。標本recordとTensorは共有。
  個別モデルgetterは設けず、読み取りによるキー作成を防ぐ。
- remap_model_training_sample_collections(*,model_id_mapping:dict[int,int])->None:
  exact dictと全キー/値のexact intを検証後、新dictへ元dict順に一回get/extendして最後に交換。
  循環/連鎖も一回のみ。未指定保持、初出宛先順、空列保存。空対応も同じ内容/順序。
IDの範囲制約なし。型違いはTypeError、日本語message。Tensor異常は追加時に触らずsamplerへ委譲。
検証エラーについて原子的であり、MemoryError/非同期操作/外部のobject.__setattr__悪用のrollbackは保証しない。
appendは旧_absorb_into_storeの標本追加と対照。空extendは実旧defaultdictへの直接extendと対照。
remapは実旧apply_server_mappingに評価store/statistics/models空namespaceを渡して対照。
旧client全体の統計やモデル再構築同値性を主張しない。

## File Structure Plan
|パス|変更|責務|
|---|---|---|
|src/federated_learning_experiments/learning/training/model_training_sample_store.py|新規|構造所有・追加・snapshot・一回ID対応|
|tests/refactoring/test_model_training_sample_storage.py|新規|実旧append/remap対照・検証契約・snapshot借用・sampler接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|変更|保持器のexact禁止/許可import guard|
|.kiro/specs/model-training-sample-storage/*|新規|承認/仕様/証拠|
|.kiro/steering/roadmap.md|変更|現在地|

## Requirements Traceability / Testing Strategy
|要件|検証|
|---|---|
|1.1–1.4|空初期/空追加/重複/負ID/交互追加、旧実absorb、snapshot保存とpayload identity|
|1.5|ID/tuple/record型違い、後尾不正標本でもstate不変、Tensor異常はsamplerで拒否|
|2.1–2.3|実旧サーバ対応の単独/衝突/連鎖/循環/空/未使用map、後尾不正mappingでstate不変|
|3.1|独立旧clientの旧samplerとnew snapshot→samplerの全batch/RNGを多条件で比較|
|3.2|依存注入RED→GREEN、新のみfresh CPU smoke、全回帰/golden/Ruff/Pyright/pip/hash/旧固定差分|
最初は新module未存在の実REDを取り、実装後GREEN。接続testは既存実NN更新の再移植を行わず母集団/抽出境界を重点確認。
