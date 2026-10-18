# FedSDAの同期末メモリ回収

通常の`run_experiment_sweep.py`と、直接の`execute_fedsda_experiment_run`は、Linux/glibcで正常な同期が終わるたびに`malloc_trim(0)`を呼ぶ。既存の実験コマンドのままで有効になり、新しいオプション・環境変数の指定は不要。

モデル・標本は保持したまま、解放済み領域をOSへ返す。学習・評価・乱数消費・計算量の計数・保存形式を変更しない。独立したprocessで動く各workerにも適用され、追加threadは作らない。回収処理の所要時間は実際の実行時間に含まれる。

Windowsや非glibcのLinux、回収機能をロードできない環境では回収を省略して従来どおり動く。返却対象がない場合も正常である。FedDrift/FedAvgと低レベルstream-only実行には適用しない。

[uraniumの対照結果](../research/fedsda-live-reclamation-uranium-20261019.md)では、MNIST2・5000/client・seed0・間隔50・1workerの最大RSSが8.87→1.64 GiBとなり、所要時間は約1.8%増加した。他条件の削減率や多workerの速度を保証する数値ではない。従来のworker数指針をこの比率だけで引き上げない。

[試作ランナー](run-memory-reclamation-prototype.md)は、当時のsource hashを固定した過去の検証用であり、今回のsource更新後は開始前に停止する。通常実験では試作を起動せず、通常CLIを使う。既存の来歴・再開時照合も維持するため、新版を同じ数値動作とみなして旧版の不完全成果物を自動上書きする扱いは追加しない。

実装の範囲・検証・独立レビューの正本は[spec](../../.kiro/specs/fedsda-runtime-memory-reclamation/tasks.md)。
