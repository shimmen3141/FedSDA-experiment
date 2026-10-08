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
