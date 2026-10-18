# 設計

## 方針・境界
既存のanalyze_behavior_fix_comparison.pyの単独CLI方式を踏襲する。新しい分析基盤や依存は追加しない。既存の集計CSVは完全な設定を持たないため、run JSONを正本として使う。旧tools/・学習処理・設定既定・goldenは編集しない。

## File Structure Plan
- analyze_fedsda_component_ablation.py: 読取り、照合、対比較、Markdown/CSV、CLI。
- tests/refactoring/test_fedsda_component_ablation_analysis.py: 合成成果物による契約・CLI検証。
- pyproject.toml: 新CLIをRuff対象に追加。
- docs/experiments/fedsda-component-ablation-analysis.md: 判断基準・実行手順。
- docs/experiments/ablation-plan.md、docs/research/method-positioning.md、docs/README.md: 現行計画との接続、主張と証拠の対応。
- .kiro/steering/resume.mdと本spec: 引継ぎと証拠。

## 入力・照合（1.1–1.4）
--baseline-directory は runs/ を持つ単一構成ディレクトリ。--ablation-directory は6構成のディレクトリを持つ実験ルート。
--configurations は既知の6構成の部分集合のみ、--datasets、--seeds、--aggregation-intervals、--client-count、--per-client-sample-count を受け取る。重複した格子引数は拒否する。
JSONの実行条件からキー(dataset, interval, seed, clients, samples)を取り出す。格子外は診断の上で除外する。キーが同じrunが複数あれば、有効そうな一方を選ばずそのキー全体を無効にする。
基準はbuilderのfedsda、変更構成は対応するbuilder名で設定を生成し、dataclassをsettings_type付きの保存形式へ変換して辞書を完全一致させる。T240など格子内で別設定の基準runは設定不一致として除外するが、同条件の正しいT120と競合する重複とはしない。設定が一致したrunが複数ある場合は重複である。
保存来歴はcommit非空、dirty=false、動作基準版正整数、revision_unavailable_reason=nullを要求する。environmentはsystem/machine/python/numpy/torchの非空文字列を要求し、対間で完全一致。この5項目は保存済み環境の照合範囲であり、CPU型番・BLAS実装・backend設定まで同等と保証するものではない。この限界を報告にも明示する。--allow-provenance-difference を指定した時だけ対間のcommit/基準版差を許可する。異なる変更構成同士の混在も報告で可視化する。
記録されたNPZ名は通常のbasenameに限定し、symlinkを含め解決先がruns/外なら拒否。JSONも通常ファイルを読み、入力の記述やコードは実行しない。

## 数値（2.1–2.4）
主要指標を明示した固定の指標表（保存パス、名称、単位、倍率、改善方向）で抽出する。null等はNone。精度の倍率100、資源は1。有限の実数でboolは除外する。
paired delta=変更−基準。高い方が良い指標はdelta>0が改善、資源等はdelta<0が改善。モデル数には優劣を付けず増/同/減と表記する。対がある値だけで双方の平均/標本SDを計算し、SE=paired SD/sqrt(n)。n<2の分散統計は未定義。信頼区間・検定・非劣性・自動採否は出さない。
欠けたrunキーを診断し、指標欠測にはキーと指標名を残す。実験失敗の理由はJSONだけでは分からない。過去の失敗sweepレコードは集計しない。

## 成果物（3.1–3.4）
mainはargparseで入力検査後に分析し、--stdoutならMarkdownだけを出す。通常は変更側ルートにanalysis_<UTC時刻マイクロ秒>ディレクトリをmkdir(exist_ok=False)し、report.mdとpaired.csvをexclusiveで保存する。CSVは1対×指標、キー・基準/変更値・差・指標単位を保存する。報告は入力runのパス、設定SHA256、環境、commit/基準版を監査表に残す。分析自身のgit revision/dirtyも取得する（秘密を含むremote情報は取得しない）。
入力ディレクトリがない等は書込み前にexit2。無効なrunは除外理由として報告し、期待格子を満たさなければexit1。格子外・別時間尺度runの除外だけはexit1にしない。部分報告の誤読を防ぐため先頭で完全/不足を明記する。

## 検証
手計算可能な合成値でseed差・SD/SE・精度単位を検証し、格子不足、重複、環境/来歴不一致、設定差、null、不正ファイル/参照先、stdout無書込みを確認。subprocessでCLIを検証。実在の既存T120120runを読み取り照合するがサーバの結果は未取得のままとする。独立CLIの仕様レビューを実装前、実装レビューを変更後にLuna highで行う。全pytest、既存fresh smoke、Ruff、Pyright、pip check、spec_checksを実行する。

## 対象構成の確定表（要求1.2）
変更側のディレクトリはすべて `<ablation-directory>/<構成名>/runs/*.json`。
- fedsda-without-candidate-validation
- fedsda-without-prediction-weight-recalibration
- fedsda-loss-distance-clustering
- fedsda-overall-loss-esr-detection
- fedsda-without-concept-specific-adapter
- fedsda-independent-models

基準側は `<baseline-directory>/runs/*.json`。既定datasetはsine2、sea2、sea4、circle2、mnist2、mnist4。

## 指標の確定表（要求2.2）
パスはrun_metricsを根とする。比率のSD/SE/差も倍率を掛けた単位で示す。通信量を標本当たりにする行ではprocessed_sample_countで割る。全runはprocessed_sample_count=client_count×per_client_sample_countを要求し、未処理のrunを除外する。

| JSONパス | 名称 | 単位 | 倍率/処理 | 改善方向 |
|---|---|---|---|---|
| prediction_accuracy | 全体精度 | %（差はpp） | 100 | 大 |
| stable_period_prediction_accuracy | 安定期精度 | %（差はpp） | 100 | 大 |
| prediction_diagnostic_metrics.combined_prediction_period_accuracies.recovery_period_accuracy | 回復期精度 | %（差はpp） | 100 | 大 |
| communication_volume.uploaded_byte_count | 上り通信総量 | byte | 1 | 小 |
| communication_volume.downloaded_byte_count | 下り通信総量 | byte | 1 | 小 |
| communication_volume.uploaded_byte_count | 標本当たり上り通信 | byte/sample | /processed_sample_count | 小 |
| communication_volume.downloaded_byte_count | 標本当たり下り通信 | byte/sample | /processed_sample_count | 小 |
| computation_cost_summary.client_forward_multiply_accumulate_count | client前向き計算総量 | MAC | 1 | 小 |
| computation_cost_summary.client_estimated_backward_multiply_accumulate_count | client推定逆向き計算総量 | MAC | 1 | 小 |
| computation_cost_summary.client_forward_and_backward_multiply_accumulate_count_per_processed_sample | 標本当たりclient前向き+推定逆向き計算 | MAC/sample | 1 | 小 |
| computation_cost_summary.server_multiply_accumulate_count | server計算総量 | MAC | 1 | 小 |
| computation_cost_summary.mean_held_model_count | 平均保有モデル数 | model | 1 | 優劣なし |
| final_global_model_count | 最終globalモデル数 | model | 1 | 優劣なし |
| change_response_metrics.loss_change_alarm_detection_metrics.detection_precision | 警報適合率 | %（差はpp） | 100 | 大 |
| change_response_metrics.loss_change_alarm_detection_metrics.detection_recall | 警報再現率 | %（差はpp） | 100 | 大 |
| change_response_metrics.loss_change_alarm_detection_metrics.detection_f1 | 警報F1 | %（差はpp） | 100 | 大 |
| change_response_metrics.training_model_switch_timing_metrics.mean_matched_switch_delay_sample_count | 対応スイッチ平均遅延 | sample | 1 | 小 |

回復期精度は保存済みの評価区間の精度であり、回復に要した時間ではない。MACは演算数の見積りで実測時間ではない。モデル数の件数は増/同/減として示す。

## 研究用解析の分離（2026-10-18追加）
- CLIをtools/research_component_ablation/analyze.py、専用26件を同ディレクトリchecks.pyへ移す。CLIのsrc参照とGit来歴取得のルートを__file__.parents[2]へ修正し、testの2か所のCLI参照を新パスへ向ける。計算・保存schemaは変更しない。移設に伴い分析来歴のscript hashと記録コマンドは変わるが、過去成果物を再生成しない。
- pyproject.tomlの解析CLI登録を取り除く。共通側に除外規則や互換aliasは追加しない。専用testは通常収集の命名にせず、明示pytestだけにする。共用fresh・CI・製品srcは変更しない。
- 現行の実行手順を新CLIへ更新する。過去の研究報告の旧コマンド名は履歴として保持する。専用lintは既存と同じ選択をisolatedで明示実行する。
- 共通登録/通常収集から除外、専用26件成功、既存全pytest/goldenとfresh成功、src/CI/共用fresh差分空を受入証拠とする。再開補助と旧新版比較CLIは、今回の一時研究介入ではなく既存の正式機能/移植検証のため対象外とする。

移設後もchecks.pyのparents[2]はリポジトリルートのまま（tests/refactoring/file.pyとtools/research_component_ablation/file.pyは同じ深さ）。この計算は変更せず、CLI参照2か所を新パスへ変更する。現在の手順更新はdocs/experiments/fedsda-component-ablation-analysis.mdだけ。docs/research/fedsda-server-results-20261018.md内の旧CLI名は実施時の履歴として保持する。
