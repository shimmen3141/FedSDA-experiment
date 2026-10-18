# Design Document

## Overview / Boundary Commitments

`tools/research_memory_reclamation/run.py`に試作・起動・照合を閉じ込める。既存の研究CLIや製品共通フローから呼ばれない。Linux/glibcだけを対象にする。Windowsは計画・専用unit checkだけで、非対応の本実行は出力作成前に拒否する。

本試作は効果検証のための独立ランナーであり、製品機能の既定化ではない。unsupported環境で黙ってno-opにせず、常に同じ対照を比較したと分かるようにする。GNU time・glibc malloc_trimがない場合は明示停止する。

## Architecture / Components

1. 親processはstdlibだけで引数、対象src hash、runner hash、出力の未存在、GNU time/platformを検証する。`--print-plan`では純読取で終了。既定の設定は固定し、`--sample-count`は正の50の倍数、`--purpose`は空でない1行、`--output-directory`を必須とする。両runのsourceは固定hash`3861f16d77fad2449baf39e37c433bb5343b56f4291efa315ce9c3aea8cf71e2`。
2. 独立したcontrol/reclaimの子processを`/usr/bin/time -v`配下で順に呼ぶ。thread環境は1、stdioはUTF-8、GNU time表示はC locale、allocator変数は保持して記録する。失敗時は次armを実行しない。既存出力は拒否し、新規root/arm/cliだけを使う。
3. 子process内で`_RoundMeasuringServerOperations.synchronize_models`を一時wrapし、元関数の正常復帰後にRSS/high-waterを`/proc/self/status`から読む。reclaimだけ`ctypes.CDLL(None).malloc_trim(0)`を呼び、controlも同じ順序の計測/JSON記録を行う。bindingは両armで準備し、argtypes=[c_size_t], restype=c_int。戻り0は失敗ではなく、その値をそのまま記録する。例外時には元例外を隠さず、patchを必ず解除する。元同期前の回収はしない。
4. 元CLIと同じsweepの`execute_fedsda_experiment_run`の参照を一時wrapし、run返却のhigh-waterを先に記録、その後に最終サーバモデルをhash化する。結果objectを変更・余分に保持せず元CLIへ返し、標準保存をそのまま行う。ハッシュ生成と保存を含むGNU timeピークとrun返却時ピークを分ける。乱数状態やモデルのforwardを呼ばない。
5. 元の`run_experiment_sweep.py`を`runpy`で実行し、引数は固定設定＋専用の新規出力＋purposeとする。標準CLIのJSON/NPZ/READMEに独自fieldを足さない。独自のevents.jsonl/parameters.json/environment.json/gnu-time.txt/logはarmの横に置く。
6. 両arm正常終了と`finished`記録を確認後、親で設定・指標・全133配列bytes・最終paramsを照合する。source/runner identityと必要な環境項目も前後/arm間で照合する。成功でも削減幅の採否は人が判断する。不一致はcomparison.jsonに理由を残しexit1。例外/中断時は不完全な記録を成功扱いしない。

## State / Numerical Invariants

- run中の学習標本・Tensor・モデルへの参照を捨てず、GCやcache clearを追加しない。回収は正常な同期の後で、全参加者が生存している間に実行する。
- 新規モデル作成・forward/backward・optimizer/RNG呼出しを挿入しない。記録値/順序は元のもの。
- 一致判定は誤差を緩めずbytesで行う。NaNと符号付き0も対象。環境間差をgolden更新の理由にしない。
- 同期末RSSの前後差を学習中の回収量として示す。終了後の回収量やサンプル和からピーク削減量を推定しない。

## Validation / Traceability

| 要求 | 証拠 |
| --- | --- |
| 1 | 計画read-only、引数/既存出力/環境拒否、逐次process実行、source identity checks |
| 2 | fake同期/trimで順序・例外/patch復元、control回収なし、データ参照保持、Linux実NNの全保存値一致 |
| 3 | 指標/配列値/shape/dtype/params/identity/未完了の不一致対照、時間/RSS集計、実NN対照 |

専用checksは`checks.py`として明示実行し、通常pytest/CI/fresh/Ruff設定へ登録しない。手元Linuxで50/clientの逐次smoke、Windowsで専用checks/Python3.10文法、isolated Ruffを行う。製品src未変更のため全pytestは今回は必須にせず、元src hashと固定旧/共通フロー差分なしを確認する。独立レビューを行い、サーバ本規模未実行は明示して配布する。
