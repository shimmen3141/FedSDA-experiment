# Research & Design Decisions

## 共通評価の調査
FedSDAはrun_metric_calculationsの精度/定常精度/検出計算を使用し、変更応答・モデル診断・予測診断・計算量は別specで拡張した。比較計算量specの単位はclient/serverの積和演算、標本あたり、モデル×標本あたりで、FedDrift計測は移植後と明記されている。今回も基本指標は共通計算/名前/単位へ合わせ、計測と追加診断は次の責務とする。候補検証/Fixed-Share再較正はFedDriftへ当てはめない。

## 実行の調査
既存single_run_executionはFedSDAのRunParticipants/5client操作/3server操作に依存する。FedDriftをcastして流用せず、bootstrapの具体部品と結果/イベントを再利用する。旧run_random_drift_experimentは事前学習→全概念列/全標本生成→floor(chunk)のclient block/R同期→flush/条件付き最終回収を行う。計測/保存前の全体状態は既存server helperで照合できる。chunk未満ではreadyが無いので旧の未定義tは評価されない。

## Linux調査
既存合意に従いWSL基準venvで固定旧の全体runを2回実行し、基本指標と最終parameter hashの再現性を確認する。調査scriptはgit管理外venv/refactoring-testsへ置く。開始時のsandbox WSLアクセスは拒否されたため、承認済みの検証範囲で通常の昇格実行を使用。Python 3.14.4 / torch 2.12.1+cpuでsine2/sea4/mnist2を独立processから各1回実行し、全基本指標と最終parameter hashを完全照合して一致した。証拠はvenv/refactoring-tests/feddrift-whole-linux-{1,2}.json。これは同条件での基本指標/最終値の再現性であり、全内部状態のprocess間照合を実施したとの意味ではない。

## 実旧oracleと記録対応
入口stubによる実旧全体run対照RED（NotImplementedError）を実測。全体loopと生成順はactual入口をobserverで取得し、旧進行の複製はしない。基本指標も入口stub RED後に接続。FedDriftのローカル切替位置はcompleted_detection_batch_transitionsのretained以外のlast_sample_indexに保存され、共有adaptation ownerの切替列は使わない（同ownerはserver remap記録用）。検出結果/位置は導出前に予測記録と対応を再検査する。

実装レビューM1への補強として、sine2/count17/threshold0.03/R3/seed71/change_probability1.0を別2processで実行した。基本指標と最終hash完全一致、切替4/precision1.0/recall0.25。追加Linux goldenを新runへ照合し全4case passed（23.61秒）。証拠はvenv/refactoring-tests/feddrift-whole-linux-drift-{1,2}.json。

生成器の構築でMNISTを読み込むため、初期準備は生成器構築からfactory完了までを1成功境界とした。読込み例外もinitial_preparation/causeへ統一し、準備済み集合/呼出元3RNGを保持する。実旧成功時の生成順は変えない。
