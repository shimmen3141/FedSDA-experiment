# Design Document

## Overview / Boundary Commitments

回収機能をruntimeの独立moduleに閉じ、既存のFedSDA計測wrapperから1回呼ぶ。共通stream loop・sweep・モデル・設定には変更を入れない。既存の通常掃引は全体runからこのwrapperを必ず通るため、逐次とprocess pool両方へ自然に適用される。FedDrift/FedAvgは変更しない。

## Components and Interfaces

`runtime/process_memory_reclamation.py`が`reclaim_unused_process_memory() -> None`を公開する。環境判定とnative bindingは非公開factoryに集約し、process内cacheで再利用する。Linuxかつ`platform.libc_ver()[0] == 'glibc'`の場合だけ`ctypes.CDLL(None).malloc_trim`を取得し、`argtypes=[ctypes.c_size_t]`, `restype=ctypes.c_int`として引数0で呼ぶ。

非対応環境、symbol欠落、ロード時のOSError/AttributeErrorは回収なしとする。戻り値0/1は採否・失敗判定へ使わない。準備後の予期しない呼出し例外は握りつぶさない。ログ・標準成果物のfield・環境変数・GC・monitor threadを追加しない。

`fedsda_measured_run_execution._RoundMeasuringServerOperations.synchronize_models`で、元同期→計算量snapshot追加→回収の順に呼ぶ。同期が失敗したらsnapshotも回収も実行されない。既存のモデル計算meterは時間ではなく演算計数なので、native回収は保存値に影響しない。モデルに触れず、既存の例外変換は実行枠へ任せる。

## Numerical / State Invariants

モデル・optimizer・標本・RNGを操作しない。元serverの引数と呼出し回数を維持し、イベントは追加しない。processごとに同じbindingを再利用するが、runの可変設定や結果はcacheしない。低レベルstream-only実行に自動挿入せず、計測つきFedSDA全体runの組立てだけが所有する。

## Validation / Traceability

| 要求 | 検証 |
| --- | --- |
| 1 | 同期→snapshot→trim順、引数転送、複数同期、例外で未回収、RNG・既存状態保持 |
| 2 | Linux/glibcと非対応、symbol欠落/ロード失敗、戻り0/1、cache再利用をfake native bindingで検証 |
| 3 | 既存計測/全体run/boundary test、共用fresh smoke、全pytest、Ruff/Pyright、Linux通常CLIの実NN対照と2worker接続 |

Linux対照はgit管理外の一時検証scriptで回収呼出しだけをno-opへ差し替えたcontrolと、製品のままのreclaimを独立processで実行する。完全設定/指標/全133配列bytes/最終paramsを比較し、2workerでは異なるseedの2runを同条件で照合する。回収呼出しはそのprocessで観測し、親processのcacheやpatchに依存しない。元の試作runnerは固定source hashを持つため改造・再利用しない。

新しい普遍的な削減率や多worker性能率は受け入れ条件にしない。効果の本規模証拠は[uranium結果](../../../docs/research/fedsda-live-reclamation-uranium-20261019.md)で、今回の接続後は正しい経路と数値不変を検証する。
