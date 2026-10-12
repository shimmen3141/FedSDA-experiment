# LEGACY-019: 候補検証の途中でIDが付け替わった参照モデルは、その検証で、再利用の対象から外れる

- 記録日: 2026-10-13。対象: 旧`federated_drift_experiment/clients/fedsda.py::_finalize_forward_validation`（`available_losses`を、`if model_id in self.models`で絞る。207〜214行付近）と、新`src/federated_learning_experiments/runtime/post_alarm_candidate_validation_progress.py`（確定の時点の保有モデルのIDを、利用できる参照として評価へ渡す）、`methods/fedsda/candidate_model_selection/post_alarm_candidate_loss_evaluation.py`の`_select_validated_reusable_reference`。固定基準748c3aa。
- 発見spec: [pending-model-registration-by-own-id](../../../.kiro/specs/pending-model-registration-by-own-id/design.md)（仕様レビューの指摘）。種別: 正常な経路で起きる、扱いの確認。状態: 記録だけ（直していない。意図した仕様かどうかは、未確認）。

## 内容

候補検証は、警報の時点で、保有する全モデル（一時IDのモデルを含む）を、IDをキーにして、参照として固定する。検証は、規定の件数の標本が届くまで、ラウンドの境界をまたいで続く。その間に、参照に含まれるモデルのIDが付け替わることがある。

- サーバの登録で、一時IDが正式IDへ付け替わる。
- サーバの統合で、IDが、統合先のIDへ付け替わる（または、無くなる）。

検証の確定は、その時点の保有モデルのIDの集合にある参照だけを、再利用の対象にする。付け替わったモデルは、検証が持つID（古いID）では保有されていないので、損失の列が残っていても、再利用の対象から、何も記録せずに外れる。比較の相手（損失の合計が最小の参照）は、固定した全参照から選ぶので、判定の記録の、比較の相手のIDに、保有していない古いIDが残ることがある。例外にはならない。

新実装は、旧と同じ扱いを移植している（`tests/refactoring/test_post_alarm_candidate_validation_progress.py`の`test_validation_progress_uses_current_reference_availability`が、実旧と照合している）。

## 影響

- 検証の途中で登録・統合されたモデルが、その検証の確定で、再利用されない。候補が棄却されると、標本は、警報の前のモデルへ取り込まれる（付け替わったモデルの方が合っていても）。
- LEGACY-016の修正（送信保留のモデルを、それ自身のIDで登録する）で、現行でないモデルの登録でも、付替えが起きるようになったので、この扱いに当たる機会は、増える方向である。起きる頻度と、精度への影響は、未確認。

## 直す場合

検証が持つ参照のIDと、開始時の学習帰属のIDを、登録・統合のID対応で付け替える（検証の保持の部品に、付替えの操作を足す）。旧に無い動作なので、挙動を直す変更として、頻度を測ってから判断する。
