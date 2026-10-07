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

次は外部test下書き→AST識別子の命名表照合→命名r2独立レビュー→task graph/各taskの独立レビュー→REDから実装。実装単位では事前評価の追加forward、後半不正時に評価store/Randomも不変、吸収後統計での区間評価、同一分類器参照を確認する。

src/test/旧実装/goldenは変更していない。今回は文書のみ。要求IDと設計coverage、JSON、LF hash、git diff --check、固定旧source/golden差分が空であることを主担当が確認した。pytest・Ruff・Pyrightや数値再実測は未実施。直前specの8455 passedを今回の検証結果とは扱わない。
