# Implementation Plan

依存: 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 2.1。どれも`test_single_run_dependency_boundaries.py`か`test_experiment_sweep.py`を編集するので、順に進める。

- [x] 1. runの成果物のまとめ
- [x] 1.1 まとめ・読み出し・戻しのmodule
  - 設計書のTesting Strategyの「module」のtest（研究ツールの`unpack`で戻ること、研究ツールの`pack`が飛ばすこと、研究ツールでまとめたzipを製品が理由を付けて読めないとすること、戻しの段1で止まると（中身の壊れたmemberを含む）`runs/`に何も書かないこと、段2の途中の失敗で戻したファイルが残り`.partial`が残らないこと、`lzma`が無いときの3つの関数の`ValueError`を含む）を先に書き、`sweep/packed_run_artifact_archive.py`を実装する。依存境界に新しいmoduleを登録する。
  - 完了条件: 新しいtest（研究ツールの`unpack`で戻ることを含む）と依存境界のtestが通り、`ruff check`が通る。
  - _Requirements: 1.2, 1.3, 1.4, 1.6, 3.1, 3.2, 4.1, 4.2, 5.2_
  - _Boundary: packed_run_artifact_archive_
- [x] 1.2 zipからの状態の判定
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
- [x] 1.3 計算の前の戻し
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
- [x] 1.4 掃引のコマンドの終わりのまとめ
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
- [x] 1.5 比較scriptと比較の集計script
  - 先に書くtest:
    - 比較scriptで、まとまること。失敗したrun・読めない要約があるときはまとめないこと。排他を取れないとき、まとめずに警告を出すこと。
    - 比較の集計scriptが、`--chains`のとき、まとめた構成で`unpack`を示して終了コード2で止まること。`--chains`の無い集計は今と同じ出力になること。
  - その後、比較scriptの終わりにまとめを入れ、比較の集計scriptに停止を入れる。
  - 完了条件: 新しいtestと既存の比較scriptのtestが通る。
  - _Requirements: 1.1, 1.5, 1.7, 6.3_
  - _Boundary: run_behavior_fix_comparison, analyze_behavior_fix_comparison_
  - _Depends: 1.4_
- [x] 1.6 既存のtestの読み方とfresh smoke
  - 掃引のコマンド・比較scriptの後に`runs/`を読む既存のtest（設計書の「既存のtestの書き換え」の10ファイル）を、戻してから読む形に直す。期待（ファイル名・数）は変えない。変えたtestと理由をImplementation Notesに書く。
  - 共用のfresh smokeに、1回目の後に`runs.zip`ができ、2回目が全部飛ばされることを足す。
  - 完了条件: 書き換えたtest・fresh smokeが通る。`-m "not slow"`の広い確認が通る。
  - _Requirements: 3.3, 5.2_
  - _Depends: 1.5_

- [x] 2. 文書と確認
- [x] 2.1 文書・全pytest・実装レビュー
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
- 9回目（8回目の反映の確認）: GPT-6 Luna（`gpt-6-luna`、effort medium、読むだけ、session `01a207b5-4888-72c0-a482-656b3c4a4e72`、基点`310c2420`）。判定PASS（指摘なし）。Lunaが確かめていないこと: requirements.mdの要求3.2との逐語の照合（確認の対象を設計とtasksに限ったため）。
- 仕様書群のレビューはPASSで済んだ。実装（GPT-6.1-solへの依頼）は、ユーザーの指示（2026-11-23「solが別の作業もやっているため、いったん待つ」）で待っている。

### 実装の前半（task 1.1〜1.3、2026-11-25）
- ユーザーの指示（2026-11-24「solは空いたので、実装を依頼してください」）で、GPT-6.1-sol（`codex exec -m gpt-6.1-sol --sandbox workspace-write`、ログのmodel行`gpt-6.1-sol`、`reasoning effort: medium`、session `01a20ec3-731c-73a0-b486-a4d15545b3cb`、基点`b197a4fc`）に task 1.1〜1.3 を依頼した。依頼文とログは`../../results/_sol_prompts/packed-runs-part1.txt`・`.log`（Git管理外）。続きを頼むときは `codex exec resume -m gpt-6.1-sol -c 'sandbox_mode="workspace-write"' 01a20ec3-731c-73a0-b486-a4d15545b3cb -`。
- 実装: `sweep/packed_run_artifact_archive.py`（まとめ・manifestの照合・読み出し・一覧・2段の戻し）、`experiment_sweep_execution.py`（zipからの状態の判定、基準版の確認の後の戻し、戻しの失敗でその構成の全runを失敗にする）、test（`test_packed_run_artifact_archive.py` 36件、`test_experiment_sweep.py` に状態の判定・再開の候補・追加と戻しとまとめ直し・失敗のtest）、依存境界の登録。Solの報告では、設計の変更・新しい公開の名前は無い。
- Solの検証: Windowsのvenvで新しいmoduleの36件と依存境界の静的検査3件が通った。sandboxからWSLを起動できず（`Wsl/Service/E_ACCESSDENIED`）、torchを読むtestと計測は流せなかった。Solが作った空の一時フォルダ`.tmp-packed-pytest`（削除がsandboxで拒否された）は主担当が消した。
- 主担当の検証（WSL Ubuntu、`/home/yshin/.venvs/fedsda-experiment`、Python 3.14.4、torch 2.12.1+cpu、逐次）: `test_packed_run_artifact_archive.py`・`test_experiment_sweep.py`・`test_experiment_sweep_resumption.py`・`test_single_run_dependency_boundaries.py` で 2,438 passed・1 failed（10分53秒）。失敗の1件は `test_temporary_model_id_allocation_rejects_every_import_except_annotations[from __future__ import *-False]` で、Python 3.14 が `from __future__ import *` を構文の誤りにするため（Windowsの Python 3.13.15 では同じ文を読める）。Solの差分にこのtestは無く、環境の違いによる。Windowsで`ruff check src tests/refactoring`と、変えた5ファイルの`ruff format --check`が通った。手元のWindowsは torch のDLLの読み込みがアプリケーションの制御ポリシーで拒否されるので、torchを読むtestはWindowsでは流していない。
- task 1.2 の所要時間（主担当、WSL）: 520 runの出力は研究ツールの変種（製品の計画で作り直せない）なので、製品の掃引の400 runの出力（`fedsda-synchronized-prediction-v16/20261124_165503_all-datasets-seeds0-9`。研究ツールでまとめた形）を WSL の /tmp へコピーし、研究ツールの`unpack`で戻して、同じ引数＋`--output-directory`で`--print-plan`を測った。`runs/` のまま 21.0秒、製品の`pack_configuration_run_artifacts`でまとめた後 22.3秒（どちらも400 runすべて`completed`、`--print-plan`の後もzipと`run_summary.csv`だけのまま）。520 runに換算して約29秒で、2分を下回るので直さない。全runが済んだ流し直しの時間は、1.4の後に測る。
- 独立の実装レビューは、全taskの後に1回受ける（未実施）。全pytestも全taskの後に流す（未実施）。

### 実装の後半（task 1.4〜1.6 と、task 2.1 の文書、2026-11-25）
- 同じSolのsession（`01a20ec3-731c-73a0-b486-a4d15545b3cb`）を `codex exec resume` で続けて依頼した（依頼文とログは`../../results/_sol_prompts/packed-runs-part2.txt`・`.log`）。基点`949e93d1`。
- 実装: 掃引のコマンドの終わりのまとめ（表と掃引の記録の後。失敗・読めない要約があればまとめない。zipが残る構成では`run_summary.csv`を書き直さない）、比較scriptの終わりのまとめ（排他を取ってから。排他・まとめの失敗は警告）、比較の集計scriptの`--chains`での停止（まとめた構成があれば`unpack`を示して終了コード2）、依存境界の許可、fresh smoke（1回目の後に`runs.zip`、2回目が全部飛ばされ、zipが変わらないこと）、文書（`docs/experiments/experiment-sweep.md`、研究ツール`tools/research_results_archive`の冒頭）。
- 既存のtestの書き換え（戻してから読む形。期待は変えない）: `test_experiment_sweep.py`・`test_experiment_sweep_resumption.py`・`test_feddrift_experiment_sweep.py`・`test_fedavg_local_updates_per_aggregation.py`・`test_fedsda_fixed_share_time_scale_axis.py`・`test_fedsda_pending_assignment_fifo_capacity_axis.py`。設計書の10ファイルのうち`test_fedavg_experiment_sweep.py`と`test_dynamic_weighted_majority_experiment_sweep.py`は、通常の実行が掃引の関数の直接の呼出しで、コマンドは計画の表示だけなので、まとめた後に`runs/`を読まず、変えなかった（Solの判断。主担当も妥当と判断。2ファイルとも下のWSLの実行で通った）。
- 整形: ルートの`run_behavior_fix_comparison.py`・`analyze_behavior_fix_comparison.py`と研究ツールは、変更の前から`ruff format --check`の対象外の状態で、足した部分だけを整形した（既存の部分の大量の整形を混ぜない）。`ruff check src tests/refactoring`は通った。
- 主担当の検証（WSL、逐次）: 変えた・関係する10ファイル（`test_experiment_sweep.py`、`test_experiment_sweep_resumption.py`、`test_behavior_fix_comparison_scripts.py`、`test_fedavg_local_updates_per_aggregation.py`、`test_feddrift_experiment_sweep.py`、`test_fedsda_fixed_share_time_scale_axis.py`、`test_fedsda_pending_assignment_fifo_capacity_axis.py`、`test_packed_run_artifact_archive.py`、`test_fedavg_experiment_sweep.py`、`test_dynamic_weighted_majority_experiment_sweep.py`）で 292 passed（14分18秒）。`tests/refactoring/fresh_process_smoke.py` は PASS（全部の流れ）。
- 全runが済んだ流し直しの所要時間（task 1.2 の残り、主担当、WSL）: 上の400 runの出力を製品の関数でまとめ、同じ引数＋`--output-directory`、8 workersで掃引のコマンドを流し直した。74.7秒で、400 runすべて`skipped_existing`、終了コード0、`runs.zip`と`run_summary.csv`のsha256は流し直しの前後で同じ。520 runに換算して約97秒で、2分を下回る。

### 実装レビューと全pytest（2026-11-25）
- 独立の実装レビュー: GPT-6 Luna（`codex exec -m gpt-6-luna -c model_reasoning_effort="medium" --sandbox workspace-write`、ログのmodel行`gpt-6-luna`、`reasoning effort: medium`、session `01a20ef6-738b-7512-a90f-a8e91a0a555c`、対象`b197a4fc..39727c97`）。判定 PASS（指摘なし）。戻しが書く前に全memberを検査すること、まとめがzip全体の照合と元のファイルの再照合の後に置いてから消すこと、zipが残る構成で`run_summary.csv`を書き直さないこと、公開の名前、既存testの期待、変えなかった2ファイルの判断、文書を確認した。独立に実行できたのは`ruff check src tests/refactoring`（成功）だけで、`test_packed_run_artifact_archive.py`は一時領域`pytest-of-yshin`の権限でfixtureの準備に失敗し、torchを読むtestはWindowsで動かないので、主担当のWSLの実測と照合した。
- 全pytestの環境: このPCのWindowsのvenvは、torch のDLLの読み込みがアプリケーションの制御ポリシーで拒否され、全pytestを流せない。WSL Ubuntu の `/home/yshin/.venvs/fedsda-experiment`（Python 3.14.4、torch 2.12.1+cpu、numpy 2.4.6）で流した。ユーザーの指示（2026-11-25）で、このvenvに `pytest-xdist==3.8.0`・`scikit-learn==1.9.1`（`requirements-dev.txt`の固定の版）と依存（execnet 2.1.2、scipy 1.18.1、joblib 1.6.0、threadpoolctl 3.7.0、cloudpickle 3.1.2、narwhals 2.28.0。Windowsのvenvと同じ版）を、OSVの既知の脆弱性・PyPIの取り下げ・最近のサプライチェーンの報告を確かめた上で、`--require-hashes --only-binary=:all: --no-deps`で入れた（`pip check`も通った）。
- 全pytest（対象 commit `39727c97`、WSL、`-n 10 --dist loadfile`、`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、除外なし）: 14,199 passed、4 failed、11 skipped（23分30秒）。JUnitは`../../venv/refactoring-tests/packed-completed-sweep-run-artifacts-full-wsl.xml`（Git管理外、このPCだけ）。同じ件数を、その前の `-n 12` の実行（23分13秒）でも得た。
  - 失敗の4件はどれも環境の違いで、今回の変更と無関係: `test_single_run_dependency_boundaries.py`の`from __future__ import *`の1件（Python 3.14 が構文の誤りにする。Windowsの3.13では読める）、`test_fedsda_minimum_description_length_alarm_interval_assignment.py::test_current_model_selection_without_baseline_adopts_a_new_model`、`tests/test_regression.py`・`tests/test_proposed_regression.py`（旧実装のgoldenはWindows・Python 3.13で作ったもので、実行の環境が違うという警告が出る）。後の3件は、変更の前の commit `b197a4fc` を一時のworktreeに取り出し、同じWSLの環境で流しても同じく失敗した（3 failed）。
  - skipの11件: riverがこのvenvに無い9件、`test_important_behavior_regression_goldens.py`の設計どおりのskip 1件（この環境では件数0のクロス評価のcaseが無い）、Windows用のgoldenとの照合をWindowsでだけ行う1件。新実装のgoldenのtest（`test_important_behavior_regression_goldens.py`）は 398 passed・1 skipped（上の1件）で、goldenの照合はskipされていない。
  - `BEHAVIOR_BASELINE_VERSION`は16のまま（変更なし。goldenのtestが通った）。
  - Windowsでの全pytest（旧の数値の回帰を含む正規の環境）は、torch のDLLの拒否が解けるまで流せない（未実施）。pyright も未実施（Windowsでは torch の解決が要る）。
- 状態: 全taskを完了し、実装レビューPASS。spec を完了にした（phase `completed`）。

