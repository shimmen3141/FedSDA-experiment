# 独立レビュー記録

## 要求 revision1 — REJECTED

- 担当: GPT-6 Luna。新規native subagent `/root/luna_alarm_preparation_requirements`、モデルを明示し、会話履歴なしで起動した。起動指定を担当モデルの根拠とする。
- 対象: ローカル外部下書き `../../venv/refactoring-tests/alarm-preparation-requirements-draft.md` と `brief.md`。requirements.mdはゲート通過前には生成しない。
- 境界: 分割と前区間保存/吸収のみ。active session、最小件数判定、区間解決、FIFO消費、検出器reset、記録/通知は後続。
- 照合対象: 固定旧`_resolve_drift`、公開FIFO分割、評価標本保存、帰属確定標本吸収。
- 結果: REJECTED。要求3.1は位置を持たない標本列で順序不一致を識別できるという、検証不能な拒否契約になっていた。
- 採用: 位置・観測標本・概念IDを結び付けた入力とし、明示位置列が保留位置列と一致しないとき拒否する。payloadの意味上正しい位置対応は供給側の保証と明記してrevision2にした。
- 任意提案の不採用: CPU float32などの適用範囲を設計だけへ移す提案。既存隣接specと同じ数値対照の対象を要求段階から限定するため維持した。
- 追加確認: 評価保存より前に全前区間を公開契約に基づいて検査する方法を設計で具体化すれば拒否時不変は実現可能。非公開呼出しや全体rollbackを要求してはいない。IMPROVE-004の事実/仮説の区別は妥当と確認された。

## 要求 revision2 — APPROVED

同じ独立担当へ修正案を再提示してAPPROVED、追加必須指摘なし。要求の数値ID/EARS、境界、拒否条件、公開部品での実現可能性、旧正常挙動/Python乱数対照を確認した。下書き実ファイルhashと提示hashも一致確認。

- 要求r2のLF SHA-256: `43c6198376ae7f2d219c0ec5590d9eb6f3f757c5a4948b1a70a74b730381d5eb`。
- 同じ内容をrequirements.mdへ生成し、spec.jsonのrequirementsだけを承認した。src/testは未変更、テストは未実施。

## 設計 revision1・初期命名 revision1 — APPROVED

別の履歴なしnative GPT-6 Luna `/root/luna_alarm_preparation_design`へ、外部設計draftと命名一覧、要求、公開sourceを提示。設計APPROVED、初期命名APPROVED、必須指摘なし。要求12項目のtraceability、具体的な境界とファイル配置、既存API、位置対応の限界、事前評価の追加forwardと拒否時不変、Python乱数照合を確認した。ゲート通過後に同じ設計をdesign.mdへ生成した。

- 設計r1 LF SHA-256: `82f81dad77eaedcc43d3a5ea76b301014bc4c55af83d3aca4e2551df9766d3f0`。
- 命名r1 LF SHA-256: `57eb69a4618ef0f1739a82df3a6e4e716d892a99ce2f4fbcaaf820cb1bdc6087`。
- 同担当が最終照合で上記2hashを独立再計算し、ObservedTrainingSampleを含む最新許可依存、命名の進捗参照文言、spec.jsonのrevision/hash一致、実装ゲートfalseを確認した。双方の承認は最新版にも有効。
- 任意提案を採用: 事前評価に使った分類器と吸収が取得する分類器が同じ登録状態であることを実装task/testで確認する。並行更新は保証外という既存境界を維持する。設計本文の変更は不要。
- test/helper/局所名は外部下書きAST確認と追加命名レビューを経るため、初期命名承認だけで実装を開始しない。tasks未作成、ready_for_implementation=false。

## 再開と未実施

## 2026-10-08の再開・task/追加命名承認

ユーザーが「task作成前に停止する必要はなくspec終了まで進める」と訂正したため、停止の範囲をspec完了へ更新した。共通引継ぎ手順も同じ指示へ修正した。

- task案r1: CLI GPT-6 LunaがNEEDS_FIXES/REJECTED。recordとguardを統合taskと明示、NN接続と変異検出を分割する指摘を採用した。
- r2の最初の呼出しは、PowerShell→Pythonの日本語置換が失敗してr1が残っていたため、旧対象のREJECTEDであり修正版への承認として使わない。apply_patchで修正後のr2再判定もNEEDS_FIXES（粒度/責務の分割）。
- r3: exact依存とfresh CPUを分けて7taskとし、runtime約160行・既存oracle再利用と前specの全suite332秒を根拠に粒度を明記。規模による懸念は解消されたが、record2module/両resolverの範囲と結果の受渡しに明確化要求。
- r4: recordの定義2moduleだけをTask1、runtime登録をTask2へ明記。結果からtraining/concept tupleを既存区間解決へ渡す接続を明記。fresh CLI GPT-6 Luna、session `01a11845-db16-74f3-9704-2390b3157893`がTASK_GRAPH PASS、TASKS APPROVED。要求12項目のcoverage/逐次依存/完了条件/境界を確認。task正本LF hash: `cfba185185392aa1312502992b1fe33fd7ddc75a3570ba058ef39d5641718187`。
- task計画のゲートでは、初回を含む2回の判定で終了するskillの既定上限を超えて修正した。今回のユーザーのspec終了まで進める指示を優先し、未承認で実装へ進む代わりに独立承認まで文書の明確化を継続した。今後は最初から具体的なscope/handoffと規模の根拠を提示する。
- 命名r2: 外部runtime/testのASTで関数/class/引数/代入/内包/import別名を抽出。Lunaが既存importの明示inventory不足を指摘しREJECTED。既存型・oracle/helper・標準libraryの定義元と同じ責務をr3へ追記した。
- 命名r3: fresh CLI GPT-6 Luna、session `01a11849-4b9a-7d62-9e89-03a29b9c50e4`がAPPROVED、必須指摘なし。LF hash: `0e23365ca203c4b11ca80abb73d0a35db74d333405a479b96bbc8bfc6be36820`。主担当もAST識別子の欠落なしを確認。主担当のimport集計は45unique binding（複数箇所で再利用）、レビュー文の112は集計方法を確認していないので独立した件数証拠として用いない。
- native thread一覧を確認したが新規レビュー起動がthread上限で失敗。close APIが提供されていないため、cleanup済みとは表現せずfresh CLI Lunaをworkspace-write/ephemeralで起動した。モデルの根拠は実起動指定と各ログのmodel行。

承認後にTask1のテストを追加しREDを実行した。後続の実装と実測は以下へ記録する。上の「未実施」は前回の停止時点の記録であり、今回の完了状態はspec.json/tasks/integration-validationを参照する。

## Task1 — APPROVED

- 主担当がrecord testを先に追加し、未実装moduleのcollection error（exit2、2.54秒）を確認。guard注入も先行して22 failed/28 passed、exit1。その後にrecord2moduleとexact guard/両resolverを実装した。
- GREEN: 対象1＋AST2066 = 2067 passed、4.05秒、exit0。Ruffの対象lint/formatも成功。frozen/keyword専用/全field必須、tuple/Tensor借用、不正入力検査をrecordで行わないことを確認。
- 独立fresh CLI GPT-6 LunaがAPPROVED、指摘なし。focused対象＋注入51 passedとRuffを独立実行し成功。主担当の2067はログを照合し、全2067の再実行とは表現していない。runtimeは未実装でレビュー範囲外。
- 証拠: `venv/refactoring-tests/alarm-preparation-task1-*-red.log`、`alarm-preparation-task1-green.log`、`alarm-preparation-task1-review.md/log`（元checkout基準のローカルパス）。

次は外部test下書き→AST識別子の命名表照合→命名r2独立レビュー→task graph/各taskの独立レビュー→REDから実装。実装単位では事前評価の追加forward、後半不正時に評価store/Randomも不変、吸収後統計での区間評価、同一分類器参照を確認する。

src/test/旧実装/goldenは変更していない。今回は文書のみ。要求IDと設計coverage、JSON、LF hash、git diff --check、固定旧source/golden差分が空であることを主担当が確認した。pytest・Ruff・Pyrightや数値再実測は未実施。直前specの8455 passedを今回の検証結果とは扱わない。
