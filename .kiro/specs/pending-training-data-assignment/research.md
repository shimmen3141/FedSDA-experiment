# 調査・設計判断

## Summary

旧基準748c3aaのFedSDA FIFOを調査し、位置の所有とモデル帰属を分離する。stdlib dequeと既存の容量設定を採用し、payload用汎用型や新規依存は導入しない。

## 調査結果

- federated_drift_experiment/clients/fedsda.py のprocess_one_stepは今回標本を追加後に警報判定経路へ進む。平時だけ容量超過分を解放するため、警報のFIFOは容量＋1件になり得る。
- _estimated_drift_startはspanをFIFO長で切り詰め、_detector_candidate_startは検出器本来のspanを使う。位置100、span80、FIFO30なら開始位置は71と21。
- _resolve_driftの候補将来検証中警報、_resolve_episode_duplicateは全件消費・clearする。短い警報区間ではold区間を割当済みでもFIFOを残す。分割参照と消費は分ける。
- flush_pending_updatesは保留学習回数を処理する操作でFIFOを消費しない。実験終端のFIFO自動消費は導入しない。

## 発見記録の正本

- [丸め境界の判定・理由不一致](../../../docs/research/implementation-findings/legacy-001-candidate-decision-reason-rounding.md)
- [短い警報後の割当済み標本の残留](../../../docs/research/implementation-findings/legacy-002-retained-assigned-samples-after-short-alarm.md)
- [終端FIFO末尾の未帰属](../../../docs/research/implementation-findings/legacy-003-unassigned-fifo-tail-at-run-end.md)

上記リンクはworktree内の共有記録を指す。今回の旧数値保持移植は不具合修正と分離する。

## 設計ゲート

主担当は全14条件のtraceability、所有/対象外/依存/再検証境界、具体ファイル、入力拒否前の状態更新禁止、3単位の実行可能性を保存前draftで確認した。外部APIを導入しない既存処理の抽出なので追加Web調査は不要。fable-methodとkiro-spec-design/tasks/implを利用する。

