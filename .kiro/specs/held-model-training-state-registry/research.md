# 調査根拠

旧固定基準748c3aa。新worktree開始HEAD86c3a99、直近実装83cddc5。
- clients/base.py::_register_trained_new_model: prepare→models[temp_id]=new_model→実損失統計→pending params/stats/ready。dictは同ID置換でも位置を保持する。
- clients/fedsda.py の採用時: 登録→学習counter帰属→upload遅延→標本extend→帰属変更。これは後続の上位統合。
- clients/base.py::confirm_model_registration: tempモデルのpop/再登録と統計/標本等のID更新は別責務。今回ID対応を先取りしない。
- 新HeldModelTrainingBindingはID/classifier/optimizerへの借用記録。個別optimizer管理器のreset後に既存bindingは自動交換されない。
- 新ParameterOptimizerStateは現在optimizerを所有し、resetで成功後に交換。新ModelTrainingSampleStoreは標本構造を別途所有。
- 新integrate_adopted_candidate_shared_featuresは候補準備だけでID登録を行わない。

正式登録全体は候補生成/学習/登録損失計算/parameter snapshot/送信状態等の依存が未整備。一つへ詰めず、保有一覧の必要な所有境界を先に移植する。今回の調査で旧正常系の新たな不具合は観測していない。

## Design Decisions
Extensionのlight discovery。既存ParameterOptimizerStateとHeldModelTrainingBindingの公開契約を照合した。dataclass/dictを採用し、汎用repository/protocol/自動cloneを追加しない。
一般化は「候補だけの登録」を「保有一覧の状態登録」へ留める。ID対応/削除/送信状態を将来用に先取りしない。
dict構造はregistry、個別optimizerの現在参照は既存owner、学習は既存executorが担う。bindingの現在参照をcacheしないことでreset後の学習接続を明示する。
適用skill: fable-method（現物調査/証拠/責務単位）、kiro-spec-init/requirements/design/tasks（EARS・Boundary・trace・task graph）、kiro-impl/review/verify-completion（RED/GREENと独立承認）。新依存・外部APIは追加せずWeb調査は不要。
