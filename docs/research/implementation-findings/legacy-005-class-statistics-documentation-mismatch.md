# LEGACY-005: クラス統計の保持について旧説明と実装が一致しない

## 状態と対象

2026-10-04、基準748c3aa。ClassConditionalESRFedSDAClientのdocstring（clients/fedsda.py:1174）とBaseClientの統計保持。状態は再現済み・未修正。種別は説明の不整合。発見specはbounded-loss-moments。

## 観測と根拠

docstringは「クラス別の事前統計を保持していないため」全体統計を基準にすると説明する。一方BaseClient._update_model_statsはclass_statsを保持し、初期pretrainと新モデル登録でもクラス統計を保存する。監視baselineが全体統計を使用する動作自体はコードと一致する。主担当と調査agentが該当ソースを確認した。

## 影響・今回の扱い

説明からクラス統計が存在しないと誤認し、監視の基準をクラス統計へ置換する危険がある。数値不具合の証拠ではなく、過去成果への数値影響は確認していない。今回の移植は保存統計と監視基準の選択を分離し、旧productionは変更しない。

## 将来の修正と検証

担当はクラス損失監視の説明。クラス統計を保持する事実と、各クラス検出器の初回観測時点の全体モデル平均を使う方針を説明し直す。説明だけの変更として実コードと照合する。統計基準そのものを変える場合は別の手法変更specと回帰・実験が必要。修正spec/commitは未定。

## 判断（2026-10-13、Claude Codeの提案をユーザーが了承）

説明の整理として採用（優先度は低い）。新実装の文書に、「クラス別の統計は保持するが、監視の基準は全体の統計」と書く。固定旧のsourceは変更しない。

## 対応（2026-10-13）

新実装の監視の部品（`methods/fedsda/loss_change_detection/overall_and_true_class_loss_monitoring.py`の`OverallAndTrueClassLossMonitor`のdocstring）と、[検出の文書](../../components/drift-detection.md)の「全体・正解クラス別e-SR混合」へ、上の説明を書いた。実コードとの照合: クラス別の検出器は、そのクラスの最初の観測で、`current_model_baseline_loss_mean`を基準に作る。クラス別の統計は、`learning/loss_statistics/model_and_class_loss_statistics.py`が保持する。動作は変えていない（説明だけ）。固定旧のdocstringは、そのまま。
