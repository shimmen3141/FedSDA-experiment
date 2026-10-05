# ローカル学習要求スケジュールの設計

## Overview
旧clientのtrain_step/flush_pending_updatesから、要求counter・間隔・回数算出だけを分離する。
正常成功後の明示確認によりcounterをclearし、失敗なら外側が確認しない。学習実体は所有しない。

## Boundary Commitments
### This Spec Owns
機能別固定設定と単一所有の保留要求counter、追加/照会/完了確認。
### Out of Boundary
NN/optimizer/標本、実更新、自動callback/再試行、並行/再入可能な実行、警報/ラウンドflush位置、
CLI/preset/完全run設定、診断、同期、PCGrad、新全体run。
初回ValidatedExperimentRunSettingsSubsetは完全run型ではないため変更しない。
### Allowed Dependencies
local_training_schedule_settings.pyはdataclasses.dataclass/field、公開core.validate_settings_field_values。
local_training_request_schedule.pyは同feature公開LocalTrainingScheduleSettingsのみ。
両者で__future__.annotationsは許可。exact AST guardを一般stdlib許可より前へ置く。
runtime/configuration/NN/Torch/Random/legacy/上流privateは禁止。
### Revalidation Triggers
要求の定義、間隔/予算、消化の順序、同期所有、公開設定を変えた場合は本specと反復specを再確認する。

## Architecture
要求一件→counter+1→間隔判定→算出int返却。
外側が保留件数snapshot→反復executorへintを渡す→正常return後だけsnapshot件数で完了確認。
明示flushは間隔を無視して現在の全pendingから算出する。0更新予算でも保留要求が正なら完了確認する。
汎用callback、計画ticket、学習phase wrapperは追加しない。

## File Structure Plan
- 新規src/federated_learning_experiments/learning/training/local_training_schedule_settings.py: frozen/kwonly/defaultなし固定2数値。
- 新規同local_training_request_schedule.py: 単一counter所有と4公開操作。
- 新規tests/refactoring/test_local_training_request_scheduling.py: 旧実要求列/拒否/成功・失敗境界と反復接続。
- 変更tests/refactoring/test_single_run_dependency_boundaries.py: exact二moduleと許可/禁止注入。
- 対象specと.kiro/steering/roadmap.mdだけ。旧source/golden/既存RunSettingsSubsetと完成済み部品は変更しない。

## Components and Interfaces
```python
@dataclass(frozen=True, kw_only=True)
class LocalTrainingScheduleSettings:
    training_requests_per_update_interval: int  # >=1
    joint_update_iterations_per_training_request: int  # >=0
    def __post_init__(self) -> None: ...

class LocalTrainingRequestSchedule:
    def __init__(self, *, local_training_schedule_settings: LocalTrainingScheduleSettings) -> None: ...
    @property
    def pending_training_request_count(self) -> int: ...
    def record_training_request(self) -> int: ...
    def calculate_pending_joint_update_iteration_count(self) -> int: ...
    def acknowledge_completed_training_requests(self, *, completed_training_request_count: int) -> None: ...
```

設定はmetadataに単位/最小値とinclusiveを宣言して共通値検査を使用する。
共通検査が整数派生型を受理するため、設定側で2fieldのexact builtin int確認を先行し、不適合はValueError。
正当なbuiltin intの値域等の項目/理由の報告は既存公開検査へ委譲する。
constructorはexact LocalTrainingScheduleSettingsを検査して__post_init__再検証後に借用し、counter=0。
型違い/改変された不正設定はValueErrorで初期化を拒否する。設定はfrozen、名前と値を変更しない。
frozenは通常の利用契約とし、構築後にobject.__setattr__で設定を改変する利用は保証対象外。

recordはcounterを+1し、interval未満なら0、以上ならcalculateの全pending×Lを返す。返すだけではclearしない。
calculateは任意のpendingに対してL×pendingを返し、状態を変更しない。
acknowledgeは正のexact intかつ現在pendingと一致する件数だけ受け入れ、検査後0へ戻す。
不正型/0/負/不一致/空への重複確認はValueError。途中追加した要求を以前のsnapshot件数で消さない。
同件数の過去batchの誤確認を区別する並行ticketは持たない。同期単一所有の外側は成功した現在batchへ確認する。

## Requirements Traceability
| 条件 | 設計/検証 |
|---|---|
| 1.1 | record +1/閾値判定、旧BaseClient実要求列 |
| 1.2 | pending×Lのみ、実反復接続と不参加試行 |
| 1.3 | calculateは全pending/0と無変更、端数/空flush |
| 1.4 | L0でもpending増加/正件数ack、旧range0と一致 |
| 1.5 | ack成功時のみclear、旧callback正常return後 |
| 2.1 | exact整数/metadata範囲、偽造設定再検証 |
| 2.2 | 不正ack拒否前後snapshot一致 |
| 2.3 | 失敗時ack未呼出、旧失敗後pending/再試行 |
| 2.4 | frozen explicit設定、constructor0 |
| 3.1 | 実旧train_step/flush_pending_updatesの列とcounter/呼出し順/回数 |
| 3.2 | exactAST/stdlib fresh smoke/実旧NN反復へのtest-only明示接続 |

## Testing Strategy
Task1のmissingmodule実REDから二moduleと対照testをGREENへ。
実旧BaseClient.train_step/flush_pending_updatesをSimpleNamespaceへbindし、train_all_held_modelsの受けたmultiplierだけを観測する。
旧学習callbackの代用はschedule検証に限り、回数の計算は旧と同じupdates_per_sample×multiplierを観測。
interval1/2/5×L0/1/3、要求/flush/端数/二重flush、失敗callbackと不正ackで各event終端pending/呼出しを比較。
core設定/記録/照会の拒否・readonly・0・frozen/default/kwonly・超巨大整数（NNループには渡さない）を検証する。

Task2はtest-only明示接続。前回test helperで同初期旧NN/optimizerを構築し、旧callbackを本物の_train_heads_togetherへ接続。
新はschedule返却countで実perform_held_model_joint_training_iterationsを実行して成功後ack。
class2/4×Adam/SGD×interval1/3×L0/2を基本16条件とし、共有update/frozenを追加して32条件。
要求/flushの混合列で各境界の全NN/grad/optimizer/RNG/呼出し予算を比較する。旧数値本体はmockしない。
参加不足の正常skipも別caseで消化を確認。上位client/streamのproduction統合とは主張しない。

Task3はexact AST実RED→GREEN、新moduleだけのstdlib fresh process schedule smokeと新CPU反復接続smoke、
対象/全pytest（既存旧11/最終3golden）・Ruff/format/Pyright/pip・旧固定差分・内容hash・UTF8/配置を確認。
11要件の証拠と後続境界を記録し、Luna実装APPROVED/最終GOで完成とする。

## 主担当design gate
PASS。全11要件・4操作・所有/例外/数値・2source/2test配置が対応する。
標準dataclassと既存値検査を採用し、設定framework/汎用callbackを増やさない。小さい同期stateを維持する。

