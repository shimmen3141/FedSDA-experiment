# Implementation Plan

依存: 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 2.1。どれも`test_single_run_dependency_boundaries.py`か`test_experiment_sweep.py`を編集するので、順に進める。

- [ ] 1. runの成果物のまとめ
- [ ] 1.1 まとめ・読み出し・戻しのmodule
  - 設計書のTesting Strategyの「module」のtest（研究ツールの`unpack`で戻ること、研究ツールの`pack`が飛ばすこと、研究ツールでまとめたzipを製品が理由を付けて読めないとすること、戻しの段1で止まると（中身の壊れたmemberを含む）`runs/`に何も書かないこと、段2の途中の失敗で戻したファイルが残り`.partial`が残らないこと、`lzma`が無いときの3つの関数の`ValueError`を含む）を先に書き、`sweep/packed_run_artifact_archive.py`を実装する。依存境界に新しいmoduleを登録する。
  - 完了条件: 新しいtest（研究ツールの`unpack`で戻ることを含む）と依存境界のtestが通り、`ruff check`が通る。
  - _Requirements: 1.2, 1.3, 1.4, 1.6, 3.1, 3.2, 4.1, 4.2, 5.2_
  - _Boundary: packed_run_artifact_archive_
- [ ] 1.2 zipからの状態の判定
  - 先に書くtest:
    - 同じrunを個別のファイルとzipの両方で判定し、状態と説明の種類が一致すること（済んだ、基準版の違う、run設定の違う、読めない要約）。
    - zipにだけrunがあるときの判定（済んだ、食い違う（要約のrun設定が違う、壊れたzip、`lzma`が無い状態で説明が分かれる）、基準版の違う（掃引を始めずに止まる）、無い）。
    - `--print-plan`の表示と、出力先に何も作らないこと。
    - 再開の候補として見つかること。壊れたzipの構成も候補に残り、そのrunが「食い違う」として数えられること。
  - zipはtestの中で1.1の関数で作る。その後、`find_existing_run_state`を変え、`experiment_sweep_execution.py`の許可リストを更新する。
  - 520 runのまとめた出力先で、`--print-plan`と全runが済んだ流し直しの所要時間を測り、Implementation Notesに記録する（流し直しは1.4の後に測る）。元のcheckoutの`results/`（worktreeの外、`../../results/`）の結果は研究ツールでまとめた形なので、コピーを研究ツールの`unpack`で戻し、製品のまとめ（1.1の関数）でまとめ直したものを使う。`../../results/`が無い環境では、小さいrunの成果物を識別子を変えて520件に複製し、製品のまとめでまとめた出力を作って測る（`--print-plan`は計画の識別子と合わないので、判定の関数を直接520回呼ぶ時間を測る）。どちらで測ったかを書く。2分を超えたら、設計書のとおり直す。
  - 完了条件: 新しいtest・既存の掃引と再開のtest・依存境界のtestが通る。
  - _Requirements: 2.1, 2.3, 2.4, 5.2_
  - _Boundary: experiment_sweep_execution_
  - _Depends: 1.1_
- [ ] 1.3 計算の前の戻し
  - 先に書くtest:
    - `execute_experiment_sweep`を直接呼び、zipに無いrunを足したとき、戻してから計算すること。
    - `runs/`とzipが両方あるとき、計画の全runがzipにあっても戻してから続けること。
    - zipのmanifestに片方だけのrunがあるとき、戻さずに、その構成のrunが失敗になること。
    - 基準版の違う要約がzipにあるとき、内側だけを直接呼んでも、戻さずに今と同じ例外になること。
    - `runs/`が無く、全runがzipにあるときは戻さないこと。
    - 戻せないとき（壊れたzip、違う中身のファイル、manifestに無いmember、zipを消せない）、その構成の全runが、計算も`skipped_existing`もされずに失敗になること（例外が外へ出ない）。
    - 直接呼びでzipに無いrunを足し、呼出し側の終わりのまとめ（`pack_completed_sweep_run_artifacts`。比較scriptと同じ呼び方）で、zipに元からあるrunと足したrunの両方が入り、全memberのバイト列が元と同じこと（要求3.3）。
    - 研究ツールでまとめたzipの構成で、戻しが書く前に止まり、`runs/`に何も作られず、その構成のrunが失敗になること。
  - その後、`execute_experiment_sweep`に戻しを入れる。
  - 完了条件: 新しいtest・既存の掃引のtestが通る。
  - _Requirements: 3.1, 3.2, 3.3, 5.2_
  - _Boundary: experiment_sweep_execution_
  - _Depends: 1.2_
- [ ] 1.4 掃引のコマンドの終わりのまとめ
  - 先に書くtest（掃引のコマンド）:
    - まとまること。進み具合の報告に、まとめた構成と警告が出ること。
    - 流し直しで、zip・`run_summary.csv`・`.resume/`が変わらないこと。
    - 計画の違う掃引を流すと、今と同じく拒否されること。
    - 失敗・読めない要約・例外（中断を注入）があるときはまとめず、`runs/`が残ること。
    - zipが残る構成では`run_summary.csv`を書き直さないこと。
    - まとめの失敗で警告が出て、終了コードが変わらないこと。壊れたzipでは終了コード1になること。
    - `lzma`が無いとき、例外が外へ出ないこと。
    - `--print-plan`がまとめず、`runs/`を消さないこと。
    - 元の削除が途中で止まった状態で、`--print-plan`が`completed`を表示すること。
  - その後、コマンドの終わりから`pack_completed_sweep_run_artifacts`を呼び、表を書き直す条件を変え、`experiment_sweep_command.py`の許可リストを更新する。
  - 完了条件: 新しいtest・既存の掃引のtest・依存境界のtestが通る。
  - _Requirements: 1.1, 1.5, 1.6, 1.7, 2.2, 2.4, 2.5_
  - _Boundary: experiment_sweep_command_
  - _Depends: 1.3_
- [ ] 1.5 比較scriptと比較の集計script
  - 先に書くtest:
    - 比較scriptで、まとまること。失敗したrun・読めない要約があるときはまとめないこと。排他を取れないとき、まとめずに警告を出すこと。
    - 比較の集計scriptが、`--chains`のとき、まとめた構成で`unpack`を示して終了コード2で止まること。`--chains`の無い集計は今と同じ出力になること。
  - その後、比較scriptの終わりにまとめを入れ、比較の集計scriptに停止を入れる。
  - 完了条件: 新しいtestと既存の比較scriptのtestが通る。
  - _Requirements: 1.1, 1.5, 1.7, 6.3_
  - _Boundary: run_behavior_fix_comparison, analyze_behavior_fix_comparison_
  - _Depends: 1.4_
- [ ] 1.6 既存のtestの読み方とfresh smoke
  - 掃引のコマンド・比較scriptの後に`runs/`を読む既存のtest（設計書の「既存のtestの書き換え」の10ファイル）を、戻してから読む形に直す。期待（ファイル名・数）は変えない。変えたtestと理由をImplementation Notesに書く。
  - 共用のfresh smokeに、1回目の後に`runs.zip`ができ、2回目が全部飛ばされることを足す。
  - 完了条件: 書き換えたtest・fresh smokeが通る。`-m "not slow"`の広い確認が通る。
  - _Requirements: 3.3, 5.2_
  - _Depends: 1.5_

- [ ] 2. 文書と確認
- [ ] 2.1 文書・全pytest・実装レビュー
  - `docs/experiments/experiment-sweep.md`と研究ツールの冒頭の説明に足す（設計書の「文書」）。
  - `BEHAVIOR_BASELINE_VERSION`が16のままであることを、全pytestのgoldenのtestで確かめる。
  - 全pytest（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、`tests/test_regression.py`・`tests/test_proposed_regression.py`を含む全部）、`ruff check src tests/refactoring`、pyrightを通す。
  - 独立CLIの実装レビューを1回受ける。
  - 完了条件: 文書があり、全pytestが通り（JUnitでgoldenのtestがskipされていない）、実装レビューがPASSである。
  - _Requirements: 5.1, 5.2, 6.1, 6.2_
  - _Depends: 1.6_

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
- 4回目: 同じモデル・effort（session `d5c16674-b313-4069-b92f-072cf4269788`）。判定NEEDS_FIXES（Major 2、Minor 8、任意3）。
- 採用（全部）: 比較scriptでも、失敗・読めない要約があればまとめないと明記した。元の削除が途中で止まり`runs/`に片方だけ残ったrunは、その片方がzipと同じバイト列ならzipから判定する（要求2.1も直した）。manifestに片方だけのrunがあるzipは戻さずに失敗にする。消す直前に元のsha256を読み直して照合する。戻しのzipの削除にやり直しを付けた。内側でも戻しの前に基準版を確かめる。task 1.4を、コマンド・比較scriptと集計script・既存testとfresh smokeの3つ（1.4〜1.6）に分け、要求IDの対応（1.6、2.4）を直した。1つの構成のまとめの結果を`ConfigurationRunArtifactPackingOutcome`にした。計測の場所を元のcheckoutの`results/`と明記した。要求3の見出しを「掃引を直接呼ぶ経路」にした。製品の外の研究ツールの扱いを今後の候補として書いた。基準版が16のままであることを全pytestで確かめる。
- 5回目: 同じモデル・effort（session `66959f58-1733-49f7-aa74-c1f858b00827`）。判定NEEDS_FIXES（Major 2、Minor 6、任意1）。
- 採用: 戻しでzipを消せなければ戻しの失敗にする（zipが残ったまま計算に進むと、表が書き直されず新しいrunの行が欠けるため）。読み出し・一覧・戻しの3つで、zipのmemberの集合がmanifestと一致することを確かめる。所要時間の基準を流し直しにも当てる。比較scriptの記録の名前と、読めない要約を集めて持つこと。要求5.2(x)に`--chains`の条件。`runs`の定数を新しいmoduleに持つ理由（循環を避ける）と、値が同じことのtest。`Outcome`を`Result`に揃えた。測定に使う結果が無い環境での代わりの方法。`.partial`の後始末をglobでなく文字列で比べる。
- 不採用（任意）: `pack_completed_sweep_run_artifacts`と`pack_completed_sweep_run_start_records`の名前の近さ。両者は同じ役割（済んだ掃引の物をまとめる）で対象だけが違い、命名表に違いを書いたので、名前が並ぶ方が分かりやすいと判断した。
- 6回目: 同じモデル・effort（session `82ecf958-cad7-4fe2-9ead-f2e54268744c`）。判定NEEDS_FIXES（Blocker 0、Major 2、Minor 4、判断事項1）。ユーザーの指示（2026-11-23「次も落ちたら止まってください」）で、ここで反映を止めた。未反映の指摘:
  - Major: 戻しが、研究ツールのzipを見分ける前に`runs/`へ書く（JSONは一致して置かれ、NPZで失敗して一部が残る）。全項目を先に検査し（パス、リンク、同名のファイル、`arrays`の有無）、1つも書かないうちに止める形にする。
  - Major: 要求3.3（直接呼びで足した後、まとめ直す）のtestがtasksに無い。1.3か1.5へ入れる。
  - Minor: 依存境界の許可を完全な名前（`zipfile.ZipFile`など）で設計に書く。`ConfigurationRunArtifactPackingResult.warning_description`を「まとめなかった理由」と「警告」に分けるか改名する。`lzma`が無い状態のtestの差し替え方（`zipfile.lzma`を`None`にするなど）を書く。
  - 判断事項: `runs/`をglobする研究ツールが、まとめた構成を黙って0件と読む点の扱い。掃引のコマンドで計画の違う掃引（seedの追加）を今と同じく拒否する扱いの、ユーザーの確認。
- 状態: 仕様書群のレビュー待ち（`specification_review.status=pending`）。実装は始めていない。次の一手は、上の未反映の指摘を直し、7回目の仕様レビューを受けること（ユーザーの判断を待つ）。

### ユーザーの決定（2026-11-23、6回目の後）
- 同じ出力先にseedを足して流すことはほとんど無い。掃引のコマンドで計画の違う掃引を今と同じく拒否する扱い（3回目のBlockerへの主担当の判断）を、そのまま採る。
- Codexが使えるようになったので、7回目からの仕様レビューはGPT-6 Luna（`gpt-6-luna`、effort medium）へ切り替える。実装はGPT-6.1-sol（`gpt-6.1-sol`）に任せる（Claudeは確認・全pytest・goldenを担う）。
- 6回目の指摘はまだ直さない（決定だけを反映する）。7回目のレビューは55分後に行い、結果が出たら止まる。7回目では、6回目の未反映の指摘を含め、各指摘がspecを変えうる実質的なものか、実装の段階で拾える細部かの判断も求める。
- 7回目: GPT-6 Luna（`codex exec`、ログのmodel行`gpt-6-luna`、`reasoning effort: medium`、読むだけ、session `01a20527-43d6-75a0-a34e-ca3edba80b7f`、基点`e22773ea`、2026-11-23 06:26開始）。判定NEEDS_FIXES（Major 2、Minor 2）。新しいMajorは無く、Majorの2件は6回目の未反映の指摘と同じ。
  - Major（specを変えうる、と判定）: 戻しの前にmanifestの全項目を検査する（パス、リンク、既存名との衝突、`arrays`の有無）。書き込みの前に止める条件と、書き込みの後のI/Oの失敗で一部が戻るのを許す条件を、設計で分ける。
  - Major（検証の計画を変えうる、と判定）: 直接呼びでzipに無いrunを足し、呼出し側の終わりのまとめで、既存のrunと足したrunの両方がzipに入るtestを、tasks 1.3か1.5に明記する（要求3.3）。
  - Minor（実装で拾える）: 要求4.1の「研究ツールの`unpack`で戻せる」を、製品が作った通常のzipに限り、`runs/`とzipが両方ある途中の状態は製品の流し直しで戻すと明記する。同じ入力を個別のファイルとzipで判定して状態が一致するtest。
  - 実装へ進んでよいかの意見: 「2件のMajorを設計・tasksへ反映してから進むのがよい」。
- 状態: 未反映のまま、ユーザーの指示で止めた（仕様レビュー待ち）。

### 7回目の指摘の反映（2026-11-23、ユーザーの指示「2件のMajorを設計・tasksへ反映してから進む」）
- Major 1: 戻しを2つの段に分けた。段1で全項目を検査し（リンク、memberの集合、研究ツールの形、パス、片方だけのrun、既にあるファイルの中身）、合わなければ`runs/`に何も書かずに止める。段2の書き込みの途中の失敗だけ、戻したファイルが残ることを許す（要求3.2）。testを1.1と1.3に足した。
- Major 2: 直接呼びでrunを足し、呼出し側の終わりのまとめで元のrunと足したrunの両方がzipに入るtestを、1.3に足し、要求3.3を1.3の対応に入れた。
- Minor（7回目・6回目の残り）: 要求4.1を製品が作った通常のzipに限った。個別のファイルとzipで判定が一致するtestを1.2に足した。依存境界の許可を完全な名前で書いた。まとめの結果を`not_packed_reason`と`warning_description`の2つの欄に分けた。`lzma`が無い状態のtestの差し替え方（`zipfile.lzma`を`None`）を書いた。
- 8回目（7回目の反映の確認）: GPT-6 Luna（`gpt-6-luna`、effort medium、読むだけ、session `01a207b3-5c99-7f73-b855-2d9043d8e1de`、基点`deab6e61`）。判定NEEDS_FIXES。Major 2（直接呼びの再まとめのtest）は解消。Major 1は一部だけ解消: 段1でmemberの中身を読まないので、圧縮データの壊れを段2まで見つけられず、一部が戻ったまま止まりうる。Minorの反映は整合、新しい矛盾なし。1回目の起動は、前面から裏へ回ったため標準入力を待って1時間止まった（主担当の起動の誤り。`< /dev/null`で起動し直した）。
- 反映: 段1で戻す項目の全memberを読み、sha256とバイト数をmanifestと照合する（中身は持たない）。段2の失敗は、読み書きの`OSError`と、段の間のzipの書き換えだけになる。testに中身の壊れたmemberの場合を足した。
