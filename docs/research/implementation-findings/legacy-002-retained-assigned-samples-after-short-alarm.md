# LEGACY-002: 警報区間が短いと割当済み旧区間もFIFOに残る

## 状態と対象

2026-10-04、旧commit748c3aaのclients/fedsda.py::_resolve_driftで発見。状態は再現済み・未修正。担当はclientの警報解決とデータ帰属。修正specは未作成。

## 観測

推定変化区間より前のold_dataは_store_evaluation_dataと_absorb_into_storeへ渡される。その後drift_dataがMIN_DRIFT_DATA未満ならinsufficient_dataを記録し検出器をresetしてreturnし、buffer.clearを実行しない。
研究調査agentが旧unboundメソッドを直接実行し、FIFO=(0,1,2,3,4,5)、span=2、最低件数5で、old=(0,1,2,3)が割当callbackへ渡り、FIFOは全6件保持されることを確認した。

主担当も[pending-training-data-assignmentの再現テスト](../../../tests/refactoring/test_pending_training_assignment_buffer.py)で旧`federated_drift_experiment.clients.fedsda.FedSDAClient._resolve_drift`を直接実行した。span=1/2/8、最低件数10で、短い警報区間と割当済み前区間の全件保持を確認した。候補将来検証中・episode重複の全件消費も同じtoken列で比較する。

worktreeで`../../venv/Scripts/python.exe -m pytest tests/refactoring/test_pending_training_assignment_buffer.py -k legacy_alarm_branches_preserve_consumption -q -p no:cacheprovider`を実行すれば再現できる。test内のMIN_DRIFT_DATAはmonkeypatchで復元され、旧コード・結果・goldenを変更しない。task 2の全対象は62 passed / exit 0。

## 影響と未確認事項

割当済みold区間が後の平時解放/警報で再び割当される可能性がある。対象分岐への到達と残留は確認済み。過去実験での件数・学習/指標への影響は未確認。
期待する改善は、既に確定した標本の重複帰属を避けること。ただし短い変化区間を次の検出へ保持する意図も考慮する必要がある。

## 今回の扱いと修正案

FIFO部品の分割参照と消費を別操作にし、現在のclient側挙動を勝手に修正しない。旧production/goldenは固定したまま。
修正案は、old区間だけ消費してdrift区間を保持する、全件を現行へ確定して消去する、帰属済み標本を追跡して重複を防ぐ方式。選択は未定。
修正時は短い警報→平時解放/再警報の標本ID列・割当回数・学習統計・旧11/最終3goldenを比較し、既存実験への影響と再実験の要否を判断する。独立再現テストと修正commit/結果を後から追記する。

## 判断（2026-10-13、Claude Codeの提案をユーザーが了承）

採用（2番目に進める）。前区間（現行モデルへ取込み済み）だけを保留から消費し、短い変化区間は保留に残す。計測（2026-10-13、新実装`ab10ad4`へ読取りだけの計数を差し込んだ。提案構成、client 10、標本2500、集約間隔50、sine2・sea4・circle2・mnist2のseed 0と1の計8 run、Windows。8 run・1つの規模だけなので、頻度は目安）: 同じ標本の2回目の取込みは、sine2で1671件と2136件（取込み全体の約6〜8%。短い警報は65回と79回で、同じ区間で繰り返す）、ほかの6 runは0〜7件。別のモデルへの重複は0件。sine2で、学習標本・損失統計・評価標本が変わる見込み。検出の証拠の扱い（IMPROVE-006）は、この修正の結果を見てから別に判断する。旧との照合を保つため、旧と同じ扱いを設定で選べる形で入れる。
