# 独立レビューと検証記録

主担当はClaude Code。レビュー担当の選択は共通引継ぎ手順に従う。

## 要求r1・設計r2・命名r1・tasks r1 — APPROVED

- 選択: 文書と命名の通常のレビューなのでGPT-6 Luna。`codex exec -m gpt-6-luna -c 'model_reasoning_effort="medium"' --sandbox read-only --ephemeral`で起動し、ログのmodel行`gpt-6-luna`と`reasoning effort: medium`を確認した。代替は行っていない。依頼文には設計レビューの確認項目（検査がすべて最初の状態更新より前にあるか、上流が検査しないfieldへの手組みの値、要求との1文ずつの照合）を入れた。
- 1回目（session `01a11af3-b45a-7993-af2b-5d8881716780`、4文書のr1）: 要求APPROVED、設計REJECTED、命名APPROVED、tasks APPROVED。旧の2メソッドの事実（actionの対応、切替位置の条件、終端位置、再利用件数を加算しないこと）、既存の確定の4結果で保留標本の帰属先が確定後の学習帰属IDと一致すること、field改名の影響範囲は問題なしと報告。
  - 設計Major: 棄却（または維持）と、前後のIDが同じ変更記録を組み合わせ、完了情報の変更前IDと帰属先IDも同じにすると、IDが異なるかの検査を通過して記録が追加される。→ 事実と確認して採用。検査「変更記録の変更後IDが変更前IDと異なる」を追加し、結果種別ごとに受理する組合せの表を設計へ加えた（設計r2）。下書きのsourceと拒否条件のtestへ反映した。
  - 設計・任意: 結果種別ごとの整合規則を表で示す。→ 採用（上の表）。
- 2回目（別session `01a11af5-95d3-7ad2-8893-7a4adcdc3d6c`、設計r2）: APPROVED、指摘なし。受理表以外の組合せが更新前に拒否されること、追加した検査が既存の確定の正常な4結果を拒否しないこと、要求r1・tasks r1との矛盾がないことを確認したと報告。
- 手順上の事実: 命名の事前登録のため、sourceとtestをリポジトリ外で下書きし、作業ツリーの一時複製へ適用して実行した（worktreeは変更していない。設計r2の反映後、対象103件が成功）。`spec_checks.py names`は、下書きを適用した複製で未登録・役割の再利用とも報告なし。承認の時点でworktreeに新src・新testはない。
- 外部証拠: 元checkoutの`venv/refactoring-tests/candidate-validation-adaptation-recording-spec-review-r1.md`/`.log`、`-r2.md`/`.log`。

## Task 1 — 実施記録と中断地点（2026-10-08）

### commit `3a007bf`までの実測

- 実装前RED: 新testはmoduleなしで収集失敗、field名を置換した既存testは40 failed、注入契約testはguardなしで22 failed/48 passed（`venv/refactoring-tests/candidate-validation-adaptation-recording-task1-red.log`）。
- GREEN: 対象103＋依存境界2681＝2784 passed、Ruff・Pyright成功、`spec_checks.py names`報告なし。実source変異42/42検出（1回目は未知の結果種別を棄却として記録する変異が未検出で、拒否条件の元を棄却へ変えて検出）。fresh新CPU 2/4class×5シナリオ成功。
- 全pytest（`3a007bf`）: 9496 passed / 3 skipped / 2 warnings、187.83s、exit 0。これは下のレビュー反映より前のcommitの値で、最終の証拠には使えない。

### Haiku 1回目（session `62792e3a-2fe5-446d-a624-812d375155df`、HEAD `3a007bf`）— CHANGES_REQUESTED

選択: 上流の未検査の型を扱う検査順序のレビューなのでClaude Haiku 5.5（`--effort high`指定、JSONの`modelUsage`で実モデルを確認）。Blocker・Majorなし。検査がすべて最初の更新より前にあること、受理する組合せ、旧との一致、既存5結果が変わっていないこと、依存9 symbol、範囲外の実装がないことを確認したと報告された。指摘と採否:

1. Minor: 変更記録がない維持・棄却で、保留標本の帰属先だけが違う入力をtestも変異も覆っていない。→ 採用。拒否条件2件と変異1種を追加。
2. Minor: 記録の位置が提案位置より前でも受理される。要求2.1の「位置が不正」に含めるか、含めないなら設計に明記すること。→ いったん「含めない」と設計r3へ明記したが、Luna（session `01a11b14-5e8c-7af0-8cff-40b498fb736e`、medium）が「要求2.1の文言を設計だけで狭めることはできない。順序検査を入れるか要求を改訂すること」とREJECTした。要求は変えず、順序検査を実装する方へ改めた（下の未コミット差分）。
3. Minor: 証拠文書の「入れていない変異」の理由が誤り（変更前IDの型検査を消すと例外の型が変わり検出される）。→ 採用。変異を追加して検出を確認し、文書を訂正。
4. 任意: 終端の記録ownerの型検査の変異が表にない。→ 採用。
5. 任意: sourceのコメント（確定結果は結果種別を検査する）。→ 採用。

指摘1・3・4・5の反映後（順序検査を入れる前）の実測: 対象105＋依存境界2681＝2786 passed、変異45/45検出、fresh新CPU成功、Ruff・Pyright成功。

### 中断地点: 実行環境の障害

順序検査の反映の途中で、固定venvのtorchのimportが`ImportError: DLL load failed while importing _C`（Windowsのアプリケーション制御ポリシーによるブロック）で失敗するようになった。numpyはimportできる。45秒待って再試行しても同じだった。回避は試みていない（システムの保護として扱う）。このためpytest・変異・fresh CPUを実行できない。

worktreeの未コミット差分（いずれも未検証・未承認）:

- `design.md`: revision3。4節へ「確定位置・終端位置が提案位置以上であること」の検査（型はTypeError、順序はValueError、更新前）を追加。8節の拒否条件数を到達時24・終端10へ。**独立レビュー未依頼**（最初のr3案＝検査しない、はREJECTED）。
- `naming.md`: revision2。module内のhelper `_validate_sample_index_not_before_proposal`を追加。**独立レビュー未依頼**。
- `src/.../runtime/candidate_validation_adaptation_recording.py`: 上のhelperと2箇所の呼出し、コメント1行。**順序検査を入れた後は一度も実行していない**。
- `tests/refactoring/test_candidate_validation_adaptation_recording.py`: 拒否条件を追加（維持・棄却で帰属先だけが違う2件は実行済み・成功。位置の順序と提案位置の型の4件は**未実行**）。
- `mutation-and-cpu-evidence.md`: 順序検査を入れる前の再生成（変異45種・対象105件）。順序検査の後に作り直す必要がある。
- Git管理外: 変異script（`venv/refactoring-tests/candidate-validation-adaptation-recording-mutations.py`）へ順序検査の変異5種を追加済み（未実行）。順序検査を入れる前のsourceの控えは`venv/refactoring-tests/candidate-validation-adaptation-recording-runtime-before-order-check.py`（REDの記録に使える）。

再開時の次の一手: (1)torchがimportできることを確かめる。(2)順序検査を入れる前のsource（`git stash`は使わず、helperと2箇所の呼出しを一時的に外す）で新しい拒否条件4件がREDになることを記録し、現在のsourceでGREENを確かめる。(3)変異scriptとfresh CPUを実行し、証拠文書を作り直す。(4)`spec_checks.py names`。(5)設計r3・命名r2をLunaへ出す（差分と、要求2.1との対応を入口にする）。(6)commitして全pytestを取り直し、Haikuへ再レビュー（1回目の指摘の解消と順序検査）。その後Task 2・3と最終GO。

### WSLでの再開（2026-10-08、ユーザー経由のCodexの提案による）

障害の原因: Windowsのスマートアプリコントロール（有効）が、署名のない`venv/Lib/site-packages/torch/_C.cp313-win_amd64.pyd`の読込みを拒否している（コード整合性ログのevent 3033/3077）。ファイルの更新日時は10月1日のままで、同じブロックが10月3日・5日にも記録されている。保護設定・venv・goldenは変更していない。

以後の実行はWSL 2 Ubuntu（Python 3.14.4、torch 2.12.1+cpu、NumPy 2.4.6、pytest 9.1.1、リポジトリ直下の`.venv`）で行った。worktreeは同じファイル（`/mnt/c/...`）。環境変数はOMP/MKL各1、`PYTHONDONTWRITEBYTECODE=1`、`MPLCONFIGDIR=/tmp/fedsda-mpl`、`FDE_MNIST_DATA_DIR`はデータの`/mnt/c`側のパス。一時ディレクトリはLinuxの既定。**WSLの結果はWindows基準での検証ではない。** Ruff・Pyright・`spec_checks.py`はtorchを読み込まないので、Windows側のPythonで実行した。

- 位置の順序検査: 検査なしのsourceで拒否条件4件が失敗（RED）、検査ありで成功。
- 設計r3・命名r2のレビュー（Luna、medium）: 1回目（session `01a11b5f-b03b-7dc3-a507-2cdb820f742d`）は命名APPROVED、設計REJECTED（提案位置-1・記録位置0が通る）。→ 採用し、提案位置の非負検査を追加。非負検査なしのsourceで拒否条件2件が失敗（RED）、検査ありで成功。2回目（別session `01a11b62-7321-7552-bc9d-959cfb93084c`）は設計r3・命名r2ともAPPROVED、指摘なし。正常な上流の出力（終端位置が提案位置と等しい場合を含む）を拒否しないことを上流の実コードで確認したと報告された。
- WSLでの実測（反映後）: 対象111（新71＋既存40）と依存境界2681のうち2791 passed、1 failed。失敗は本specと無関係の既存test 1件で、Python 3.14の構文解析の違いによる（証拠文書の「依存のexact一致」に記載）。変異51/51検出、復元後111 passed。fresh新CPU 10条件成功。
- Windows側のPythonでの静的検査: Ruff check/format成功（172 files）、Pyright 0 errors、`spec_checks.py names`報告なし。

Windows基準で残る検証: 対象test・依存境界test・変異・fresh CPU・全pytest（旧11・最終3golden）。torchが読み込めるようになってから実施する。それまで完了ゲート（Task 3と最終GO）は通過扱いにしない。

### Haiku 2回目（session `4fda1d4d-3253-4cba-8458-049150f9ce4b`、HEAD `af3ffa7`）— CHANGES_REQUESTED

Blocker・Majorなし。1回目の指摘1〜3と任意2件の解消、追加した拒否条件8件がそれぞれ意図した検査で拒否されること、追加した検査がstoreへの追加より前にあり正常な上流の出力を拒否しないことを確認したと報告された。指摘と採否:

1. Minor: 推定変化点が提案位置より後でも受理される（上流のsession開始は`0 <= 推定変化点 <= 提案位置`を検査している）。上限検査を足すか、上流の契約として扱うなら設計に明記すること。→ 検査は足さない。同じ`AdaptationRecord`の推定変化点を、警報応答の記録（承認済みの設計で順序を検査しない）と候補検証の記録とで別の規則にしないため。以前の設計レビューが「要求の文言を設計だけで狭められない」と指摘しているので、要求2.3を追加して「不正」の範囲を定義した（要求r2）。設計r4へ検査しないものと理由を明記した。
2. 任意: 判定記録の採否評価と確定結果の対応を検査しないことを設計に明記する。→ 採用（要求2.3と設計r4）。
3. 任意: testの切替位置の`in`判定が、直後の完全一致の検査と重複している。→ 不採用（挙動の検証には影響しない。testを変えると全回帰の取り直しになる）。
4. 任意: helper名が検査内容（提案位置の型・非負も見る）より狭い。→ 不採用（命名r2で役割を説明済み。Lunaが承認している）。

要求r2・設計r4・tasks r2のレビュー（Luna、medium、session `01a11b73-99c6-7b42-9bb1-1880dfb2d568`）: 3段階ともAPPROVED、指摘なし。検査しない2点を上流の契約とする根拠（session開始の不変条件、警報応答の記録の扱い）が実コードと一致すること、現在の実装が要求2.3の(a)(b)(c)を満たし、除外した2点以外に未検査の「不正」がないことを確認したと報告された。source・testは変更していない。

### WSLでの全pytest（commit `af3ffa7`、tmux内）

9504 passed / 3 failed / 3 warnings、273.63s。JUnitとlogは元checkoutの`venv/refactoring-tests/candidate-validation-adaptation-recording-full-wsl.xml`/`.log`。Windowsではskipになる3件（POSIX bashが要るtest）はWSLでは実行されて成功した。

失敗3件:

- `tests/refactoring/test_single_run_dependency_boundaries.py::test_temporary_model_id_allocation_rejects_every_import_except_annotations[from __future__ import *-False]`: Python 3.14の`ast.parse`が構文解析の時点でSyntaxErrorにする（基準のPython 3.13では成功）。本specと無関係の既存test。
- `tests/test_regression.py::test_regression`（旧11ケース）: 11ケース中の複数（FedSDAのADWIN/ClassADWIN/ESR/ClassESR系とObliviousのblobs）で、精度が小数第3位前後で異なり、一部は検出遅延・通信量・最終モデル数も異なる。
- `tests/test_proposed_regression.py::test_proposed_regression`（最終3ケース）: sine2の精度の時系列のhashが異なる（精度0.9204対golden 0.9191）。イベント件数（coverage）は一致。

golden回帰の不一致について確かめたこと: この2つのtestが実行するのは旧実装`federated_drift_experiment`だけで、旧実装・2つのgolden・2つの回帰testは固定旧`748c3aa`から差分が空（`git diff 748c3aa HEAD`で確認）。したがって本specの変更はこの結果に影響しない。Windowsの基準環境では、障害の前に同じ2testが成功している（commit `3a007bf`の全pytest 9496 passed。旧実装は同一）。torchの版はどちらも2.12.1で、違いはOS（Linux/Windows）、Python（3.14.4/3.13）、数値ライブラリの実装。小さな浮動小数の差が学習と検出の分岐を変えたと考えるのが自然だが、どの演算で最初に差が出るかは調べていない（未確認）。goldenの更新、許容誤差の変更、testのskipは行っていない。docs/experiments/refactoring-baseline.mdのとおり、別OSでの不一致を理由にgoldenを変えない。

結論: WSLでは新実装のtest（tests/refactoring）は既知の1件を除いて成功するが、golden回帰はWSLでは判定できない。旧11・最終3goldenの確認は、Windowsの基準環境でtorchが読み込めるようになってから行う。（追記: 同じ日にWindowsの基準環境で実施し、旧11・最終3goldenを含む全pytestが成功した。下の「Windows基準環境での再実行」。）

### Haiku 3回目（session `026f935d-98c2-4a21-a55e-89b95458d815`、HEAD `2a6aff1`）— Task 1 APPROVED

指摘なし（Blocker・Major・Minorなし）。要求r2の2.1・2.3の(a)(b)(c)がすべて更新前に検査されていること、検査しない2点を実際に読んでいないこと、未検査の「不正」がないこと、受理する組合せが設計4節の表と一致し上流の正常な4結果を拒否しないこと、2回目の指摘の採否の記録が実装・要求・設計と一致することを確認したと報告された。読取り専用のためtestは実行していない。Task 1を完了とした。source・testのcommitは`af3ffa7`。

## Windows基準環境での再実行（2026-10-08、commit `733994b`）

Task 1の承認後、Windowsの基準環境でtorchが読み込めるようになった。次のspec（held-candidate-validation-progress）のsource・testを含むcommit `733994b`で、対象test・依存境界・変異・fresh CPU・全pytestを実行した。本specのsource・test（`af3ffa7`）は`733994b`まで変更していない。結果は[mutation-and-cpu-evidence.md](mutation-and-cpu-evidence.md)の末尾と[integration-validation.md](integration-validation.md)。WSLの結果との違いは、既知の3件（Python 3.14の構文解析の違い1件、golden回帰2件）がWindows基準では成功することだけだった。

### Task 2・3（Luna、session `01a11bc6-d0eb-73d0-82b4-e422dc44727d`、HEAD `b99ce32`）— TASK 2: APPROVED / TASK 3: APPROVED

選択: 証拠の照合と再実行が中心の、分量の多いレビューなのでGPT-6 Luna（`codex exec -m gpt-6-luna`、effort `medium`を明示、実行ログのmodel行とreasoning effort行で確認）。sandboxはworkspace-write。指摘なし。レビュー担当が独立に実行したもの: 対象test＋依存境界（2910 passed）、fresh CPU（10条件成功）、Ruff check/format、`spec_checks.py identity`（承認hash、固定旧差分、source hash、JUnit 9666 testcase、golden回帰2件の成功）。変異scriptは読んで照合（実行していない）。全pytest・Pyright・pip checkは独立実行していない。

レビュー担当は「`git diff --stat 733994b b99ce32 -- src tests`に4ファイルの差分があった」と報告したが、主担当が同じコマンドを実行した結果は空で、同時に実行した別specのレビュー担当（session `01a11bc6-d0c9-7c63-81fe-30c5e7197075`）も空と報告した。報告された4ファイルは`git diff --stat af3ffa7 733994b -- src tests`の結果と同じで、取り違えと判断した。判定には影響しない。

### feature最終レビュー 1回目（Luna、別session `01a11bd0-b136-7d01-a3d4-171fd8d00d1b`、HEAD `c76f6fb`）— NO-GO

選択: 文書・承認・証拠の照合が中心の最終レビューなのでGPT-6 Luna（effort `medium`を明示、実行ログで確認）。これまでのレビューとは別のsession、読取り専用。要求9項目はすべて「満たす」と判定され、仕様・実装の機能的な不一致はないと報告された。`spec_checks.py identity`を独立に実行して成功。pytest・fresh CPU・変異・Ruff・Pyright・pip checkは独立実行していない。

1. Minor: spec.jsonの`implementation_progress.completed`が1、`phase`が`implementation-in-progress`のままで、全3task完了のtasks.md・task_reviewsと矛盾する。→ 事実で、採用。Task 2・3の承認を記録したときの更新漏れ。`completed`を3、`phase`を`feature-final-review-pending`へ改めた。source・test・承認済み文書は変更していない。

### feature最終レビュー 2回目（Luna、別session `01a11bd3-0503-7fc1-9fe6-4764c79e15fb`、HEAD `0bfddc0`）— NO-GO

1回目とは別のsession、読取り専用。1回目の指摘の解消（spec.jsonの進捗）、要求9項目（すべて「満たす」）、設計と実装の一致、承認の記録を確認したと報告された。`spec_checks.py identity`を独立に実行して成功。pytest・fresh CPU・変異・Ruff・Pyright・pip checkは独立実行していない。指摘はどちらも、解消済みの待ち状態が記録に残っていること:

1. Minor: 再開案内（.kiro/steering/resume.md）が本specを「Windows基準の検証待ち」のままにしている。→ 採用。再開案内を現在の状態へ更新した。
2. Minor: このreview.mdの「WSLでの全pytest」の結論が「Windowsで読み込めるようになってから行う」のままで、後段の実施記録と対応していない。→ 採用。結論へ実施済みの追記を入れた（元の文は当時の記録として残した）。次のspecのreview.md冒頭の実行環境の記述にも同じ追記を入れた。
