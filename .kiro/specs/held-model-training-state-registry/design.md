# 設計: 保有モデルの学習状態管理

## Overview
モデルIDからNNと個別optimizer管理器を参照する保有一覧を一つのregistryで扱う。学習呼出し用の既存HeldModelTrainingBindingは、必要時に現在optimizerから作る。
Goals: 旧dictの初出順/置換順、参照保持、reset後の現在参照を維持する。
Non-Goals: 正式候補登録全体・サーバID対応・学習実行。

## Boundary Commitments
### This Spec Owns
保有一覧のdict構造、登録/同ID置換、readonly状態記録、取得/順序付きsnapshot、借用学習記録生成。
### Out of Boundary
NN/optimizer生成・clone、shared owner、shared接続/値反映/reset、採否/採番/ID対応/削除、標本/統計/送信/counter、新client/全体run。
モデル/ownerの寿命は上位が管理し、registryが登録参照を保持する。二つのregistryへの同じ実体の登録を禁止せず、排他的所有や自動cloneを保証しない。候補移管の時機は上位の責務。
### Allowed Dependencies
新moduleはdataclasses.dataclass、torch.float32/strided、torch.nn.Parameter、torch.optim.Adam/SGD、ResidualAdapterClassifier、ParameterOptimizerState、HeldModelTrainingBindingの公開型/属性だけ。旧package・private helper・factory・sampler・学習executor・config・random・I/Oへ依存しない。
### Revalidation Triggers
登録/置換/順序/parameter対応・参照寿命を変える場合、共同学習と将来client登録/同期を再照合する。

## Architecture
上位→registry→既存NN/optimizer管理器・借用記録。登録準備と学習実行の間に状態保有だけを置く。新ライブラリやprotocolは追加せず、標準dictとfrozen dataclassを使う。
現在optimizerをcacheしない。状態recordは管理器を参照し、bindingは生成時のoptimizerを参照する。この違いを名称とdocstringで示す。

## File Structure Plan
|パス|変更|責務|
|---|---|---|
|src/federated_learning_experiments/learning/training/held_model_training_state_registry.py|新規|状態recordと順序付き保有一覧|
|tests/refactoring/test_held_model_training_state_registry.py|新規|登録/拒否/借用/実旧順序/共同学習対照|
|tests/refactoring/test_single_run_dependency_boundaries.py|変更|新module exact依存guard|
|対象spec、.kiro/steering/roadmap.md、resume.md|新規/変更|承認・実測・現在地|

## Components & Interfaces
### HeldModelTrainingState
frozen=True, kw_only=True dataclass。model_id:int、classifier:ResidualAdapterClassifier、concept_specific_parameter_optimizer_state:ParameterOptimizerState。record自体はreadonlyだがNN/管理器はliveな可変参照。構築だけでは入力検証せず、registryの登録境界で検証する。

### HeldModelTrainingStateRegistry
__init__()で空のdict[int,HeldModelTrainingState]を所有する。
- register_held_model_training_state(*,model_id:int,classifier:ResidualAdapterClassifier,concept_specific_parameter_optimizer_state:ParameterOptimizerState)->None。
  exact builtin int、exact classifier/owner、owner現在exact Adam/SGD、モデルのadapter→classification順parameterが非空・一意なexact Parameterで、CPU float32 strided/notnestedかつ共有parameterから独立、現在optimizer flatten groupsのidentity/順序が一致することを確認してからrecordを生成しdictへ代入する。ID範囲制約なし、同ID置換は初出位置を維持。既存classifier内部層の外側改ざんは通常契約外。
- get_held_model_training_state(*,model_id:int)->HeldModelTrainingState。
  ID型検証後dict参照。未登録はKeyError(model_id)。型/対応違いはValueErrorの日本語message。取得による空登録なし。
- snapshot_ordered_held_model_training_states()->tuple[HeldModelTrainingState,...]。
  現dict順の新tuple。record/実体をcloneしない。後続登録/置換でtupleとrecordのフィールドは変わらず、owner resetはliveに観測される。
- snapshot_ordered_held_model_training_bindings()->tuple[HeldModelTrainingBinding,...]。
  現dict順で既存bindingを新規生成し、owner.parameter_optimizerを借用する。登録後の通常操作はowner reset・shared接続だけで概念parameter不変。再検証・reset・cloneをこの取得へ追加しない。古いbindingのoptimizer参照は自動更新しない。

_validate_model_idはID検証、_validate_held_model_training_state_inputsは公開参照から登録対応を検証するprivate helper。外部部品のprivate関数を再利用しない。
通常検証失敗は変更前。MemoryError、並行変更、private/frozen回避のrollback保証は対象外。異なるIDが同実体を指すことは旧dict同様に許容し、ID内の対応と順序だけを保証する。

## Requirements Traceability / Testing Strategy
|要件|検証|
|---|---|
|1.1|空状態・空binding一覧|
|1.2|負/ゼロ/正/大整数、実体identity、初出順|
|1.3|同ID別モデル置換、旧record/snapshot保持、実旧dict順序|
|1.4|readonlyrecord、snapshot構造分離、取得で登録増加なし|
|1.5|bool/int派生/NumPy整数・型違い・owner不一致・逆parameter順・不正dtype/空/共有重複で全状態保持|
|1.6|未登録KeyError、取得の型拒否|
|2.1|現在optimizerのbinding生成、空/複数順序|
|2.2|学習state蓄積→reset→旧binding保持/新binding反映|
|2.3|値/grad/state/RNG保持、共有optimizer外側保持|
|3.1|実旧registerのmodel dict境界を明示対照、class2/4×Adam standard/AMSGrad/SGD×共有更新/凍結の12条件で登録/置換/reset→実旧sampler/反復と新registry→既存反復を3step exact照合|
|3.2|exact AST先行RED、fresh新CPU smoke、全pytest旧11/最終3golden、Ruff/format/Pyright/pip、固定旧差分/sourcehash|
旧registerはloss/stats/pendingも実行するが、その数値を本registryの完了範囲へ含めない。test-onlyの上位対応を明記し、旧/新model内部は既存公開APIで対応させる。
