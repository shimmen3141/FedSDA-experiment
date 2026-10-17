# Design

## Overview / Boundary Commitments
対象は新実装run_experiment_sweep.pyとその掃引実行。run_behavior_fix_comparison.pyの選択UI、旧CLI、学習アルゴリズム、既存artifact writer/loaderは変更しない。共通execute_experiment_sweepにも出力先排他とrun開始記録を接続する。

## Architecture
CLI → 過去掃引探索・選択 → 出力先排他 → 元のREADME・掃引実行・CSV/実行記録。
実行 → run開始補助記録 → 元のrun実行・保存。
新module sweep/experiment_sweep_resumption.pyは補助ファイル・排他・計画照合・候補情報を担当。既存run状態検査はexecutionが所有し、候補探索はCLIから渡した状態検査callbackを使い循環importを避ける。

## Identity / Discovery
run設定を既存convert_settings_to_plain_mapping→JSON roundtripで正規化し、構成名と全設定からなる順序非依存リストを比較。run識別子だけで一致としない。探索範囲はリポジトリresults配下のREADME.md/.resume/sweep.jsonを持つ掃引root（symlinkを辿らない）。新補助計画があればそれを優先し、不正時はREADMEへfallbackしない。
旧READMEは既存writerが作る「コマンドの引数」fence内だけをshlex.splitで解析し、CLI parserと同じ計画生成を使う。shellとして実行しない。READMEの動作基準版を確認する。旧オプション等の解析不能・未知基準版は候補外。sweep_recordはCtrl+C時にないため探索の必須条件にしない。
全runのfind_existing_run_stateでcompleted/conflicting/missing/baseline mismatchを数える。開始記録により復旧可能なNPZ単独を補足する。完全な候補選択はexit0で無変更。候補なしは通常の新規実行。--new-outputは候補を再開せず新規作成。非TTYで候補あり・出力先なしはexit2で明示指定を要求。EOF/Ctrl+C/0は無変更キャンセル。--output-directoryは従来どおり明示再開。新規時は既存の同秒パスと衝突しない連番を選ぶ。

## Auxiliary Files / Recovery
既存形式とその版は不変。追加のみ:
- .resume/sweep.json: version=1、behavior baseline、正規化全run集合。CLIが排他取得後・計算前に原子的作成。同一なら保存内容を維持。
- .resume/<configuration>/<run-id>.json: version=1、完全run設定、configuration/run-id、RunProvenanceの辞書、環境system/machine/python/numpy/torch。各runの計算前に原子的に保存する。同じrunの通常再実行では現在の記録を用意するが、NPZ単独の既存記録を先に書き換えない。
- .resume/recovered/<uuid>/<元npz名>: 復旧対象をrenameで退避。既存結果の消去はしない。
- .sweep.lock: OS advisory排他用の1byteファイル。Windowsはmsvcrt.locking、Linuxはfcntl.flockで非blocking取得。PIDの生死・lockファイルの存在で判定しない。ファイルを残してinode差替えによる二重取得を防ぐ。process終了でOSが解放する。ネットワークFSはOS排他を正しく提供する環境が前提。

JSONなしNPZありに限り、開始記録が現在の完全設定・基準版・cleanかつ既知の同commit・同環境と一致した場合に復旧可能。dirty/不明commit、旧NPZ単独、JSONのみ、読めない完了組、設定不一致は自動処置しない。開始記録は実行中の証拠でありNPZの内容を認証するものではない。利用者によるファイルの手動改変は保証外。
executionの全状態検査前に排他取得し、全worker終了まで保持する。CLIは同じ排他をREADME/実行/集計まで保持し、executionには既取得を示す内部引数を渡す。候補表示時は変更しないが、選択後に排他と全状態を再確認する。範囲外symlink経由の退避・補助記録は拒否する。

## Validation
中断を学習中（artifactなし）と保存途中（NPZ単独）に分けて試験する。clean同一証拠のみ復旧、退避保持、証拠なし/異なるcommit/環境/基準版/設定・dirtyでは無変更。二重process拒否と終了後再取得、worker数変更再開、旧README探索、入力順差、完了候補無変更、非TTY、print-plan無変更を検証。既存artifact schema testと全pytest/golden、Ruff/Pyright、共用fresh CPU smokeで確認。

再開時の説明メタデータ: 元READMEは不変。sweep_recordは今回のpurpose/study/引数を従来通り記録する（READMEは初回、recordは呼出しごとの記録）。再開時にこの区別を表示する。条件照合とは独立で、異なるpurposeでも元READMEの結論を変更しない。

実装レビューMajor1への対処: 開始証拠にexecution_source_sha256を追加する。gitに未追跡のものも含め、src配下とroot直下の全Pythonの相対path・LF内容をSHA256へ含める。追加・変更・削除があれば復旧しない。ソースのsymlinkは証拠生成時に拒否する。既存のrun_provenance/既存成果物形式は変更しない。
