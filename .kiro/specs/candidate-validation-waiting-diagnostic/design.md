# Design Document

## Overview / Boundary Commitments
`run_candidate_validation_waiting_diagnostic.py`を実験専用の入口とし、製品srcは一切編集しない。V/Oは通常runの直接呼出し、Dだけprocess内の名前解決点をcontext managerで差し替える。既存掃引へ構成を登録せず、独立の診断成果物として分離する。

## Architecture and Contracts
- CLI: `--cases dataset:interval ...`、`--seeds`、`--modes validation immediate delayed-forced`、`--sample-count`（既定5000）、`--workers`（1〜14）、`--output-directory`、`--purpose`、`--print-plan`。
- 標本数は各caseの間隔で割り切れる正整数とし、末尾未処理の条件は本診断では拒否する。datasetは既存6種、seed非負、case/seed/mode重複を拒否する。
- 各workerは1runのみ。multiprocessingのspawn Pool、maxtasksperchild=1。各processでtorch/OMP/MKL/OpenBLASを1threadへ設定。入れ子の並列化はしない。
- Vはbuilderのfedsda、Oはwithout-candidate-validation、Dはfedsdaから構築。V/Oには差替えを適用しない。
- Dは`runtime.post_alarm_candidate_validation_progress.evaluate_candidate_using_post_alarm_losses`だけを差替え、元関数を同じ引数で1回評価する。dataclasses.replaceでcandidate_accepted=True、reusable_reference_model_id=None、decision_reason=`diagnostic_forced_adoption_after_validation_wait`へ変更。その他の値を保持し、元評価と適用後を別記録へ残す。元関数に乱数操作はなく、差替え自体も乱数を使わない。
- この差替えは規定件数到達後にしか呼ばれない。開始・観測・同期・終端の経路はそのまま。理由はstrで列挙制約がなく、record tableへ正直な新理由を保存できることを確認済み。
- 介入履歴の順番は完了検証の順番。NPZのcandidate_validation判定数/強制理由と照合できるようにする。元判定はそのD軌道上でのwould-decisionであり、V軌道の反実仮想の最終精度ではない。

## Output and Errors
run名はdataset/mode/interval/seed/sample数。各runの専用ディレクトリをmkdir(exist_ok=False)で原子的に予約してから学習する。予約ディレクトリ内へrecords.npzとresult.jsonを保存し、JSONを最後にexclusive createする。同名ディレクトリがあれば空でも全格子事前検査で拒否し、同時起動の競合は原子的mkdirで拒否する。実行失敗時はcontext managerで復元し、予約ディレクトリを保持する。JSONがなければ未完了であり、自動削除・再開・修復はしない。
JSON schema: format=`candidate-validation-waiting-diagnostic-v1`、dataset/interval/seed/sample_count/mode/purpose、source_commit/has_uncommitted_changes、source_tree_sha256/script_sha256、environment（system/python/numpy/torch/torch_threads/OMP/MKL/OpenBLAS）、processed_sample_count、run_settings（元の完全設定）、run_metrics（実指標）、record_table_file_name=`records.npz`、record_table_sha256、intervention_events（元評価と適用後評価の各mapping）、rng_before_sha256/rng_after_sha256。通常format2 JSONと混ぜない。既存保存形式は変えない。
各runの実行直前にPython/NumPy/torchの外側RNGをseedで初期化する。実験関数は外側RNGを保存復元する契約を持つので、その前後状態を同一環境でのhash（pickle protocol=5でPython random.getstate()の全tuple、NumPy legacy global RandomStateのnp.random.get_state()全5要素＝bit-generator名/MT19937状態配列/位置/has_gauss/cached_gaussian、およびtorch CPU get_rng_state()の全byteを直列化しSHA256）で記録する。V/Oと直接runの対照testは同じprocessで初期化状態を復元して実行し、全配列・JSON化指標と3種の終了RNG状態を直接比較する。processごとの未初期化RNGのhash同士は比較しない。
source全Pythonの正規化hashとscript hashを開始時・run前後で照合し、実行中のコード変更を拒否する。出力をsrcに置くことは禁止する。print-planは学習・出力作成なし。

## Behavioral Specification / Verification
|入力・遷移|期待結果|証拠|
|V/Oと直接実行、同seed|全配列・指標・RNG一致|小規模実run対照|
|D収集9→10|9件以前は差替え未呼出し、到達時1回だけ強制採用|既存進行helperによる境界テスト|
|元reject/reuse/accept|全てaccepted、元評価保持、独自理由|評価値の入力例テスト|
|例外発生|元の関数へ復元|context失敗テスト|
|未完了候補|D強制イベントなし、V同等の終端記録|既存終端helper/実run|
|既存ファイル/不正case|学習前拒否|CLIテスト|
|print-plan|run数一致、出力未作成|CLI subprocess|

## Interpretation
D−Oは待機だけでなく凍結・標本帰属・参照作成なども含む。D−Vも方策全体の差であり単一イベントの因果寄与率ではない。既存サーバ結果と異なる環境の片側比較にしない。採否は自動化しない。

NumPyのdefault_rng/Generatorはこの外側RNG検査の対象ではなく、本実装は外側にGeneratorを生成しない。既存run内の局所RandomStateは元の実行が所有する。fingerprintはPython/NumPy/torchの版が一致する同一環境内だけで比較し、Gaussianキャッシュだけが違う状態でもhashが変わることをテストする。
