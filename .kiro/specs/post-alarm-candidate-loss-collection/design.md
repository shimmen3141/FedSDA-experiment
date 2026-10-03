# 設計: 警報後候補損失収集

## Overview

候補/開始時参照の外部算出済み損失列だけを所有する状態部品。提案次位置から規定件数へ到達する旧正常clientの時系列を保持し、完了結果を既存数値評価へ明示入力する。

## Boundary Commitments

### This Spec Owns
収集位置・候補/参照のloss系列・規定件数到達状態・immutable copyとatomic入力拒否。

### Out of Boundary
モデル/参照snapshot生成、学習、payload/held_data、履歴baseline、採否起動、episode、正式登録/切替、終端回収。

### Allowed Dependencies
stdlibと同機能CandidateModelTrainingAndAcceptanceSettingsだけ。torch/NumPy、既存採否関数やFIFOへのproduction importなし。数値算出は呼出側が行う。

### Revalidation Triggers
位置順・参照順・target到達時機・float化・不正入力状態・結果表現の変更では、採否接続と後続clientのforward→判定→monitor/FIFO順を再検証する。

## Architecture

一つの収集状態と同所有のfrozen結果。独自モデルID型、model Protocol、汎用session frameworkは作らない。stdlibのlist/dict/tupleを使い、数値平均はしない。

## File Structure Plan

- 新規src/federated_learning_experiments/methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_collection.py: 収集クラス・同所有snapshot・全入力検査。
- 新規tests/refactoring/test_post_alarm_candidate_loss_collection.py: 旧session/client直接oracle、atomic拒否、位置/copy/RNG/keyword、既存採否への明示接続。
- 変更tests/refactoring/test_single_run_dependency_boundaries.py: exact同機能設定だけ許可・禁止注入。
- 変更docs/research/implementation-findings: 旧session不正入力の部分更新を個別追跡し正常経路への影響を未確認とする。
- 変更.kiro/steering/roadmap.md: 完成境界と後続責務。

## Requirements Traceability

| 条件 | 実装・証拠 |
|---|---|
| 1.1, 1.2, 1.3 | ctor・固定参照ID tuple、設定/位置/ID検査、旧session参照順 |
| 2.1, 2.2, 2.3, 2.4 | observe・全検査後commit、次位置/同一系列長/float化・拒否state不変 |
| 3.1, 3.2, 3.3 | ready/count、target直後拒否、旧client到達位置capture、非自動採否/終端消去 |
| 4.1, 4.2 | frozen snapshot、tuple系列、別実体/input dict/RNG/default dtype/device不変 |
| 5.1, 5.2, 5.3 | 旧session/client直接oracle、既存採否へのtest接続、全AST/golden/smokeと共有発見 |

## Components and Interfaces

PostAlarmCandidateLossCollection(*, candidate_model_training_and_acceptance_settings:CandidateModelTrainingAndAcceptanceSettings, proposal_sample_index:int, reference_model_ids:tuple[int,...])。
exact既存設定型と__post_init__再検査、exact proposal int>=0、exacttuple・非空・全ID exactint・重複なしを構築時検査。
内部は設定、提案位置、参照tuple、候補list[float]、参照dict[int,list[float]]、最後位置None。
proposalは診断contextで、最初の追加はproposal+1、次からlast+1。

- observe_losses_after_label_observation(*, sample_index:int, candidate_loss:float, reference_losses_by_model_id:dict[int,float])->None: readyならRuntimeError。位置連続性、候補loss、exactdict、全keyのexactintと開始集合一致、全参照lossを更新前に検査する。builtin int/float finite[0,1]のみ。float化した全値を局所で用意してから全系列へ追加しlast更新。dict順ではなく開始順に読む。
- validation_sample_count:int property: len(candidate list)。
- ready_for_acceptance_evaluation:bool property: count>=設定件数。超過追加は拒否するので到達後は不変。
- get_state_snapshot()->PostAlarmCandidateLossCollectionState: frozen kw_only dataclass。proposal_sample_index、last_validation_sample_index、candidate_losses tuple、reference_losses_by_model_id tuple[(id,tupleloss)]、validation_sample_count、required_validation_sample_count、ready_for_acceptance_evaluation。途中/完了とも可。

snapshotの参照tupleを呼出側がdictへ明示変換して、既存evaluate_candidate_using_post_alarm_lossesへ渡す。historicalmean/available/currentID/閾値/settingsは上位から明示入力。収集クラスは採否部品をimportしない。

## Error Handling

型違反TypeError、値域/連続性/集合違反ValueError、規定件数後の追加RuntimeError。項目名を含む。候補/参照全値の検証・float化をstate更新前に終え、拒否でready/count/last/全系列を変えない。
旧session不正入力の部分更新とtarget超過許可は正常clientで起きない呼出を新契約で拒否する差として記録する。旧productionは変更しない。

## Testing Strategy

task1は旧sessionへ同じ値を直接appendし、target2/3/5、参照順(9,-2,3)・逆順inputdict、float化、proposal次位置、atomic invalidとconstructorを比較する。
task2は旧client::_observe_forward_validationをtest-only candidate/reference model stubとfinalize captureで直接実行し、target到達位置・消えたlive参照でも固定snapshot全件を観測することを比較する。immutable/copy/別実体/RNG/defaultdtype/device/keywordを確認し、snapshotlossを既存採否へ明示接続する。
task3はexact ASTと禁止注入、全tests旧11/最終3golden、旧import/torch/NumPyなし独立smoke、15条件の証拠とpartial範囲を統合する。

