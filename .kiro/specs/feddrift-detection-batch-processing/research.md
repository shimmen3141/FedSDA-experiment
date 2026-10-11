# Research & Design Decisions

## Summary
- Feature: feddrift-detection-batch-processing。Extension（既存数値判定と状態ownerの接続）。
- 旧`BaseFedDriftClient._detect_and_assign`と`BaseClient._spawn_new_model/_register_trained_new_model/_absorb_into_store/_store_evaluation_data`を調査した。
- 検出時の初期化元は最小損失モデルではなく現在モデル。`FedDriftClient.new_model_initial_epochs`は0。新規モデルの統計はバッチ平均・不偏分散で、1件なら分散0.1。非検出では標本ごとのWelford更新である。

## Research Log
- 既存の分類器は独立モデル構造で旧SimpleMLPと同順の乱数を消費する。snapshotは旧名へ変換せず、test側でparameter順と値を対応付ける。
- learningのモデル一覧・学習標本・現在帰属・採番・損失統計、evaluationの標本保持を再利用できる。
- `PendingModelUploadState`は遅延1以上しか受理しない。queue→advanceで即時扱いにする代わりに、同ownerへ明示的な即時操作を足す。配置の`methods/fedsda`は現状の共有ownerの配置で、新しいFedDrift専用の保留枠を複製しない。
- 旧early_stoppingはepoch0でもrandpermを消費する。標準fixed/noneの初期学習0回を本specの範囲にし、その方式は未移植として残す。

## Design Decisions
- バッチ処理をruntimeに置く。前回値のみをprocessorが所有し、それ以外の一覧・帰属・統計・標本・採番・送信保留は既存ownerを借用する。
- 検出位置・切替位置と分岐を不変の結果で返す。適応記録や位置列への保存は後続のclientが担う（旧FedDriftはadaptation_eventsを記録しない）。
- 旧の部分更新を改善し、標本payload・owner型・全モデルとoptimizer対応・現在ID・寸法・設定を先に検査する。数値結果と統計・snapshotを準備した後、ownerを更新する。全失敗のtransaction rollbackやRNG保存復元は実装しない。
