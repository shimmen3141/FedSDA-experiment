# Research / Decisions

既存の本規模対照は最大RSS8.868→1.637 GiB、wall+1.79%、133配列bytes/最終params一致。追加の採用価値調査ではなく、通常経路への小さい接続を行う。

通常sweepの`execute_planned_experiment_run`は`execute_fedsda_experiment_run`を呼び、計測つき実行のserver wrapperを経由する。試作のpatch箇所と同じ正常同期末に接続できるため、共通loopへのcallback追加やsweepの新引数は不要と判断した。低レベルloopに環境依存を入れる案と、実験設定に回収方式を足す案を採らない。

参照skill：fable-method、kiro-spec-init/requirements/design/tasks。要件の境界と観測可能性、light discoveryの既存接続点確認、命名レビューを適用。stdlibのみで新依存なし。FedDrift/FedAvgまで今回の性能結果を一般化せず、FedSDAに限定する。
