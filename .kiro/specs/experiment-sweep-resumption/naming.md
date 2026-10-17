# 公開名

| 名前 | 役割・区別 |
|---|---|
| sweep/experiment_sweep_resumption.py | 保存済み掃引の探索・再開補助。学習や既存artifactの形式を所有しない |
| SweepResumptionCandidate | 過去掃引の保存先と完了・要確認・復旧可能件数 |
| describe_planned_run_set | 構成名と完全設定の順序非依存な集合をJSON互換値で表現 |
| describe_resumption_environment | 復旧一致に用いる数値環境 |
| hold_sweep_output_lock | 出力先に対するOS排他をcontext managerで保持 |
| write_sweep_resumption_plan | 新しい補助計画を原子的に保存し、既存との矛盾は拒否 |
| write_run_start_record | 個別run計算前の条件・来歴を原子的に保存 |
| can_recover_interrupted_run | NPZ単独と同一の開始証拠があるかを無変更で確認 |
| recover_interrupted_run | 復旧可能NPZを退避して未実行状態へ戻す |
| find_sweep_resumption_candidates | 同条件の過去掃引を探索して進捗を返す |
