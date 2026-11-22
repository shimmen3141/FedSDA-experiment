# Implementation Plan

依存: 1.1 → 1.2 → 1.3 → 1.4 → 2.1。どれも`test_single_run_dependency_boundaries.py`か`test_experiment_sweep.py`を編集するので、順に進める。

- [ ] 1. runの成果物のまとめ
- [ ] 1.1 まとめ・読み出し・戻しのmodule
  - 設計書のTesting Strategyの「module」のtest（研究ツールの`unpack`で戻ること、研究ツールの`pack`が飛ばすこと、研究ツールでまとめたzipを製品が理由を付けて読めないとすること、`lzma`が無いときの3つの関数の`ValueError`を含む）を先に書き、`sweep/packed_run_artifact_archive.py`を実装する。依存境界に新しいmoduleを登録する。
  - 完了条件: 新しいtest（研究ツールの`unpack`で戻ることを含む）と依存境界のtestが通り、`ruff check`が通る。
  - _Requirements: 1.2, 1.3, 1.4, 1.6, 3.1, 3.2, 4.1, 4.2, 5.2_
  - _Boundary: packed_run_artifact_archive_
- [ ] 1.2 zipからの状態の判定
  - 先に書くtest:
    - zipにだけrunがあるときの判定（済んだ、食い違う（要約のrun設定が違う、壊れたzip、`lzma`が無い状態で説明が分かれる）、基準版の違う（掃引を始めずに止まる）、無い）。
    - `--print-plan`の表示と、出力先に何も作らないこと。
    - 再開の候補として見つかること。壊れたzipの構成も候補に残り、そのrunが「食い違う」として数えられること。
  - zipはtestの中で1.1の関数で作る。その後、`find_existing_run_state`を変え、`experiment_sweep_execution.py`の許可リストを更新する。
  - 520 runのまとめた出力先で、`--print-plan`の所要時間を測り、Implementation Notesに記録する。手元の`results/`の結果は研究ツールでまとめた形なので、コピーを研究ツールの`unpack`で戻し、製品のまとめ（1.1の関数）でまとめ直したものを使う。2分を超えたら、設計書のとおり、構成ごとにzipを1回だけ開く形に直す。
  - 完了条件: 新しいtest・既存の掃引と再開のtest・依存境界のtestが通る。
  - _Requirements: 2.1, 2.3, 2.4, 5.2_
  - _Boundary: experiment_sweep_execution_
  - _Depends: 1.1_
- [ ] 1.3 計算の前の戻し
  - 先に書くtest:
    - `execute_experiment_sweep`を直接呼び、zipに無いrunを足したとき、戻してから計算すること。
    - `runs/`とzipが両方あるとき、計画の全runがzipにあっても戻してから続けること。
    - `runs/`が無く、全runがzipにあるときは戻さないこと。
    - 戻せないとき（壊れたzip、違う中身のファイル）、その構成の全runが、計算も`skipped_existing`もされずに失敗になること（例外が外へ出ない）。
  - その後、`execute_experiment_sweep`に戻しを入れる。
  - 完了条件: 新しいtest・既存の掃引のtestが通る。
  - _Requirements: 3.1, 3.2, 5.2_
  - _Boundary: experiment_sweep_execution_
  - _Depends: 1.2_
- [ ] 1.4 終わりのまとめの接続
  - 先に書くtest（掃引のコマンドと比較script）:
    - まとまること。
    - 流し直しで出力先が変わらないこと。
    - 掃引のコマンドで、まとめた出力先へ計画の違う掃引を流すと、今と同じく拒否されること。
    - 掃引のコマンドの経路で`lzma`が無いとき、例外が外へ出ないこと。
    - 失敗・読めない要約・例外（中断を注入）があるときはまとめず、`runs/`が残ること。
    - zipが残る構成では`run_summary.csv`を書き直さないこと。
    - 比較scriptで排他を取れないとき、まとめずに警告を出すこと。
    - 比較の集計scriptが、まとめた構成で`unpack`を示して止まること。
    - まとめの失敗で警告が出て、終了コードが変わらないこと。
    - 進み具合の報告に、まとめた構成と警告が出ること。
    - `--print-plan`がまとめず、`runs/`を消さないこと。
  - その後、両方の終わりから`pack_completed_sweep_run_artifacts`を呼び、表を書き直す条件を変え、比較の集計scriptに停止（`--chains`のときだけ）を入れ、許可リストを更新し、fresh smokeに確認を足す。
  - 掃引のコマンド・比較scriptの後に`runs/`を読む既存のtest（設計書の「既存のtestの書き換え」の10ファイル）を、戻してから読む形に直す。期待（ファイル名・数）は変えない。変えたtestと理由をImplementation Notesに書く。
  - 完了条件: 新しいtest・掃引と比較scriptのtest・依存境界のtest・fresh smokeが通る。
  - _Requirements: 1.1, 1.5, 1.7, 2.2, 2.5, 3.3, 5.2, 6.3_
  - _Boundary: experiment_sweep_command, run_behavior_fix_comparison, analyze_behavior_fix_comparison_
  - _Depends: 1.3_

- [ ] 2. 文書と確認
- [ ] 2.1 文書・全pytest・実装レビュー
  - `docs/experiments/experiment-sweep.md`と研究ツールの冒頭の説明に足す（設計書の「文書」）。
  - 全pytest（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、`tests/test_regression.py`・`tests/test_proposed_regression.py`を含む全部）、`ruff check src tests/refactoring`、pyrightを通す。
  - 独立CLIの実装レビューを1回受ける。
  - 完了条件: 文書があり、全pytestが通り（JUnitでgoldenのtestがskipされていない）、実装レビューがPASSである。
  - _Requirements: 5.1, 5.2, 6.1, 6.2_
  - _Depends: 1.4_

## Implementation Notes

### 仕様書群のレビュー（2026-11-23）
- 1回目: Claude Haiku 5.5（effort medium、読むだけ、session `14f54d85-2db5-4f63-a2bf-6d551584d4ea`、`modelUsage`はclaude-haiku-5-5）。判定NEEDS_FIXES（Blocker 0、Major 6、Minor 8、任意1）。状態の判定・排他・ファイル操作の順序が絡む設計なので、Haikuを選んだ。
- 採用（全部）: 途中で止まった後（`runs/`とzipが両方ある）も戻す。壊れたzipで一覧を読めないときも、その構成のrunを失敗にする（例外を外へ出さない）。zipが残る構成では`run_summary.csv`を書き直さない。流し直しのtestの対象を、zip・表・`.resume/`に限る（掃引の記録は新しく増える）。比較の集計script（`analyze_behavior_fix_comparison.py`）が`runs/`を読むことを一覧に足し、まとめた構成では止める。中断のtestを足す。まとめを掃引の記録の後へ移す。manifestのキーを研究ツールと同じ`byte_count`・`format_version`にする。戻り値で「まとめた」と「警告」を分ける。圧縮方式は`ZipInfo`で指定する。研究ツールの`pack`が飛ばすことのtestを足す。tasksの不足を足す。比較scriptで排他を取れないときは警告にする。読み出しの関数の名前を`read_packed_run_summary_and_record_table`にし、非公開の処理の重複を設計に書く。zipを何度も開く所要時間を実測する。
- 2回目: 同じモデル・effort（session `f4845f24-e94c-4b6a-803e-2569336546d0`）。判定NEEDS_FIXES（Major 2、Minor 6、任意1）。
- 採用（全部）: 1回目で足した「`runs/`とzipが両方あるとき戻す」と「全runがzipにあれば戻さない」の優先順位を明記した（`runs/`があれば戻す。戻さないのは`runs/`が無いときだけ。要求2.2もそう限る）。戻しに失敗した構成は状態のループから外し、全runを失敗にする（一部が戻ったrunを飛ばさず、zipに無いrunを計算しない）。戻しの`.partial`の後始末と`runs/`の作成。研究ツールの`unpack`は両方ある構成では止まるので、製品の流し直しで戻すと文書に書く。比較の集計scriptが止まる理由と対象を正確にした。所要時間の判断の基準（`--print-plan`が2分を超えたら、構成ごとにzipを1回だけ開く形に直す）。`lzma`が無いときの説明を壊れたzipと分ける。tasksの要求の対応（4.2と5.2）と、要求1.7・`--print-plan`のtestを足した。名前の違いを命名表に足した。
- 3回目: 同じモデル・effort（session `2e51e623-42ba-466f-a8d7-13a0fbf718f3`）。判定NEEDS_FIXES（Blocker 1、Major 3、Minor 8、任意2）。
- Blocker（採用、範囲を絞る）: 掃引のコマンドは、同じ出力先で計画の違う掃引を`.resume/sweep.json`の照合で止める（今の規則）ので、要求3（seedを足す）はコマンドでは成り立たなかった。レビュー担当は(a)計画の上位集合なら`sweep.json`を置き換える、(b)要求3を外す、をユーザーの判断として挙げた。主担当は、今の再開の安全規則を変えない(b)寄りの扱いを選んだ: コマンドでの計画の違う掃引は今と同じく拒否し（Out of scopeに明記、testで確かめる）、戻しは`execute_experiment_sweep`を直接呼ぶ経路（比較scriptなど）と、`runs/`とzipが両方ある途中の状態のために残す。理由: ユーザーの依頼はまとめることで、再開の規則の変更ではないため。ユーザーへ報告する。
- 採用（Major）: 掃引のコマンドは`ensure_lzma_compression_available`を通らないので、読み出し・一覧・戻しの3つ全てで`OSError`以外を`ValueError`へ変える（LZMAの説明を分ける）。研究ツールでまとめたzip（NPZを書き直している）は製品で読めないので、manifestの`arrays`で見分けて理由を示し、文書に「`unpack`してから流し直す」と書く。所要時間の測定は製品でまとめ直した出力で行う。掃引のコマンド・比較scriptの後に`runs/`を読む既存のtest 10ファイルを洗い出し、戻してから読む形に直すとtasksに書いた。
- 採用（Minor・任意）: `runs/`を読む一覧に`write_run_start_record`を足した。戻しの置き換えにやり直しを付け、中断で残った`.partial`を先に消し、リンクを確かめる。照合の基準の書き方。集計scriptは`--chains`のときだけ止める（要求6.3も直した）。壊れたzipの構成が再開の候補から落ちないこと。要求の番号の並び、要求1.7を「名前」に直した。文書に、排他を取らない読み取りでの一時的な見え方を足した。命名表に、完了の確認は呼出し側と書いた。空の`runs/`は何もしない。
