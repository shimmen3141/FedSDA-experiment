# 独立レビューと検証記録

主担当はClaude Code。レビュー担当の選択は共通引継ぎ手順に従う。実行環境はWSL 2 Ubuntu（Python 3.14.4、torch 2.12.1+cpu）。Windowsの基準環境はtorchの読込みがスマートアプリコントロールでブロックされており、WSLの結果はWindows基準の検証ではない。

## 要求r1・設計r1・命名r1・tasks r1 — APPROVED

- 選択: 文書と命名の通常のレビューなのでGPT-6 Luna（`model_reasoning_effort="medium"`、read-only、ephemeral。ログのmodel行と`reasoning effort: medium`を確認）。代替は行っていない。依頼文には設計レビューの確認項目（検査がすべて最初の状態更新より前にあるか、手で組み立てた入力で誤った保持や記録が残らないか、要求との1文ずつの照合）を入れた。
- 結果（session `01a11b86-d96d-76e0-9379-79558a7186fa`）: 4段階ともAPPROVED、指摘なし。検査の順序、要求との対応、旧実装との対応、tasksの検証ゲートを確認し、下書きに対して、異常入力で誤った保持や記録が残る具体例は見つからなかったと報告された。
- 手順上の事実: 命名の事前登録のため、sourceとtestをリポジトリ外で下書きし、作業ツリーの一時複製へ置いてWSLで実行した（worktreeは変更していない。対象38件が成功）。`spec_checks.py names`は、下書きを置いた複製で報告なし。承認の時点でworktreeに新src・新testはない。
- 外部証拠: 元checkoutの`venv/refactoring-tests/held-candidate-validation-progress-spec-review.md`/`.log`。

## Task 1 — 実施記録（WSL）

- 実装前RED: 新testはmoduleなしで収集失敗、注入契約testはguardなしで45 failed/73 passed（`venv/refactoring-tests/held-candidate-validation-progress-task1-red-wsl.log`）。
- GREEN: 対象38＋依存境界2799のうち2836 passed、1 failed（本specと無関係の既存test。Python 3.14の構文解析の違い）。Windows側のPythonでRuff check/format・Pyright・`spec_checks.py names`が成功。
- worktreeのsourceとtestは、仕様の承認時にLunaが読んだ下書きから、import順（Ruff）だけが変わっている。
- 変異28/28検出、fresh新CPU 10条件成功（[証拠](mutation-and-cpu-evidence.md)）。
- WSLの全pytest（commit `27bd187`、tmux内）: 9660 passed / 3 failed（前specの9504＋対象38＋注入契約118）。失敗3件は前specと同じ（Python 3.14の構文解析の違いによる既存test 1件と、golden回帰2件の環境差による不一致。旧実装は固定旧から無変更）。下のレビュー反映より前のcommitの値。

### Haiku 1回目（session `8357ceb2-138a-4dfc-9226-ec2a3c92c021`、HEAD `27bd187`）— CHANGES_REQUESTED

選択: 保持の更新順序と、上流の未検査の値を扱う実装のレビューなのでClaude Haiku 5.5（`--effort high`指定、JSONの`modelUsage`で実モデルを確認）。検査の位置、旧との一致、部分更新の主張、testのoracle、依存1＋20 symbol、範囲外の実装がないことは問題なしと報告された。指摘と採否:

1. Major: 応答の再実行は、区間解決を持つ応答のsessionを「区間解決が開始したsessionと同一」としか見ない。応答と区間解決の両方を差し替えると、再利用の応答へsessionを足した入力が通って開始していないsessionを保持し、開始の応答からsessionを外した入力が通って何も保持しない。→ 事実と確認して採用。「sessionを持つのは候補検証中と開始の応答だけ」の検査を追加した（設計r2）。拒否条件2件を先に追加し、検査なしのsourceで失敗（RED、WSL）することを確認してから実装した。
2. Minor（任意で扱ってよい）: 上流の進行は標本位置が提案位置より後であることを検査しないので、提案位置より前の位置で確定すると、上流の更新の後で記録が拒否される。→ 本specでは検査を足さない（標本位置は上流へ渡すだけの引数で、要求3.3の範囲外）。設計4節へ既知の限界として明記した。上流の進行に位置の検査を足すかどうかは、標本ごとのclient進行のspecで判断する。
3. 任意: 設計の「応答の`__post_init__`がexact session型を保証済み」は開始の応答では成り立たない。→ 採用（設計r2で訂正）。
4. 任意: sourceのコメントが過大。→ 採用。

設計r2・命名r2・tasks r2のレビュー（Luna、medium、session `01a11b99-d68d-74d2-a822-3275599958b5`）: 3段階ともAPPROVED、指摘なし。その後、変異検査で「応答の再実行の削除」が未検出になったため、再実行だけが拒否する入力（正式な5値でない結果種別）の拒否条件を1件追加し、設計8節の拒否条件数を12へ改めた（設計r3）。設計r3はLuna（別session `01a11ba3-5cb8-7f72-b34b-58a37c7d4b5c`）がAPPROVED。Pyrightが、保持の条件を結果種別で書いた版でOptionalの受渡しを指摘したので、「sessionがNoneでない」の条件へ戻した（対応の検査により、ここへ来るのは開始の応答だけ）。

反映後の実測（WSL）: 対象41＋依存境界2799のうち2839 passed、1 failed（既知の既存test）。変異29/29検出、復元後41 passed。fresh新CPU 10条件成功。Windows側のPythonでRuff check/format・Pyright・`spec_checks.py names`が成功。
