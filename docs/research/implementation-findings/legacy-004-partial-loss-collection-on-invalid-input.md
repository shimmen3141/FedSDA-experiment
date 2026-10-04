# LEGACY-004: 不正な参照損失入力で旧収集sessionが部分更新する

## 状態と対象

2026-10-04、旧748c3aaのfederated_drift_experiment/provisional_model.py::ForwardValidationSession.append_losses。状態は再現済み・未修正。種別は直接APIの不正入力への堅牢性。正常clientでの発生は未確認。

## 再現と観測

固定参照順=(2,1)、target_count=2の空sessionへappend_losses(.2,{2:.4})を直接渡すとKeyError(1)。例外後もcandidate_losses=[.2]、reference_losses={2:[.4],1:[]}が残る。調査agentと主担当が旧メソッドを直接実行して確認した。
旧APIは余剰IDを無視し、規定件数到達後の直接追加も許す。通常clientは全固定参照を毎回供給してtarget到達直後にfinalizeするため、これらの不正/過剰呼出が通常実験で起こるとは断定しない。

再現は[収集部品のテスト](../../../tests/refactoring/test_post_alarm_candidate_loss_collection.py)のtest_candidate_loss_collection_invalid_observation_is_atomic_for_legacy_partial_update。worktreeで`../../venv/Scripts/python.exe -m pytest tests/refactoring/test_post_alarm_candidate_loss_collection.py -k legacy_partial_update -q -p no:cacheprovider`を実行する。旧メソッドの部分更新と新APIの全状態不変を同じ不足参照で比較する。

## 影響・今回の扱い

直接APIで例外を捕捉して続行した場合、系列長の不一致・不正な検証件数となる可能性がある。過去成果への影響は未確認。今回の新収集部品は全値を更新前に検査して拒否時状態を維持し、規定件数後を拒否する。これは新契約の入力境界であり旧production修正ではない。

## 将来の修正と検証

旧APIを別途修正するなら、全参照の検証/float化後にcommitし、余剰ID・規定件数後を拒否するかの契約を明示する。正常clientへの直接照合・旧11/最終3goldenと例外後状態を検証する。担当は候補損失収集、修正spec/commitは未定。新spec post-alarm-candidate-loss-collectionに対応する拒否テストと再現証拠を置く。

