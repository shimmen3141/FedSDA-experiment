# Implementation Plan

- [x] 1. runごとの全参加者を事前学習から準備する
  - 実旧の初期準備が実行できることを先に確認し、対照testをstubでREDにする。
  - 検査を生成前に揃え、全client/serverの初期状態・optimizer・3乱数が実旧に一致する集合を返す。
  - 独立owner/共有Random、再準備と失敗時の直前集合保持を確認できれば完了。
  - _Boundary: runtimeのFedDrift参加者初期準備_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2_
- [x] 2. 準備した集合から観測・同期と独立起動を接続する
  - 実旧の初期準備後の観測・同期へ対照をつなぎ、全状態/通信/RNGが一致することを確認する。
  - exact依存と禁止依存例を登録し、共用freshでfactoryから既存client/serverへ進める。
  - 対象/依存境界/freshが成功すれば実装taskを完了する。
  - _Boundary: 初期準備から観測・同期への接続検証_
  - _Depends: 1_
  - _Requirements: 3.1, 3.2_

## Implementation Notes

- task graph gateはinline PASS。前提部品は移植・照合済み。要求7項目を2責務へ対応付けた。仕様書群の独立レビューPASS後に1taskずつmanual --review offで実装する。
- 全task後の独立実装レビュー、指摘反映、全pytest/JUnitと品質検査、source identityは別のspec完了ゲート。Minor以下は修正・採否記録のみ、Major/Blockerは修正箇所だけ再確認する。

- 仕様書群レビュー（GPT-6 Luna medium、`/root/feddrift_factory_spec_review`）PASS、指摘なし。旧/新構築契約とtask graphを読合せ。実旧oracleの起動も主担当が確認しtask 1へ進む。

- task 1: oracle設定名と必須のモデル設定fieldを修正後、4件がstubのNotImplementedErrorでRED。実装後43 passed（5.13秒、接続4件を含む）。実旧setupの全model/optimizer/統計/owner/server/3RNG、共有Randomと状態分離、拒否時の乱数/直前集合保持と再準備を確認。独立モデルでは未使用の要求rankを1に固定する。

- task 2: 対象43件＋依存境界3082件で3125 passed（13.26秒）。実旧setup出力から3時刻の観測/R回同期を行い全状態/通信/3RNG一致。exact依存/禁止3例、新構成だけの共用fresh 1 passed（89.34秒）を確認。Ruff check/format（316 files）、names、progress、pip checkとPyright 0 errors/0 warnings成功（Protocolの具体属性はgetattrで検査）。実装レビューとその後の全pytestは未実施。

- 実装レビュー（GPT-6 Luna medium、`/root/feddrift_factory_implementation_review`）PASS、指摘なし。独立対象/依存3125 passed（11.15秒）、Ruff check/format、names、progress、diff --check成功。fresh/全pytestは独立再実行なし。命名の実態も承認。主担当がこの後の全pytestで最終判定する。

- 独立実装レビュー後の最終検証（2026-10-12）: 全pytestは13214 passed / 4 skipped / 9 warnings、exit 0（673.80秒、-n 8 --dist loadfile）。前specの13168件へ対象43件と禁止依存3件を加えた数と一致。JUnitは13218件、failure/error 0、対象43件・既存11ケース/提案3ケースgolden回帰・fresh smoke成功。skipはPOSIX bashなし3件と既存回帰へ委ねる環境golden1件。最終Ruff check/format（316 files）、Pyright 0 errors/0 warnings、pip check、names、progress、diff --check成功。固定旧748c3aa・golden・tools差分なし。全7要求の初期準備/拒否/再準備/観測・同期接続を主担当が照合。source/testはこの検証後に変更しない。全pytest/freshは独立再現なし、WSL/変異は未実施。終端/精度評価/全体run指標/成果物/CLI/golden移植は後続。ログ/JUnitは`../../venv/refactoring-tests/feddrift-run-participant-factory-full.log`と同名`.xml`。

- 完了確認: 検証済みsource commit `2eaeb18ef43acf146f777c68de4d677c788a29e7`、source hash LF `3ddfe117d6c0bc76cd8ba51347b6331707cb1ff34cb652c52ac263fd94ac62d7`（420パス）。identityで固定旧差分/clean tree/JUnit/2回帰/件数を確認してPASS。初期準備から観測・同期までの境界・7要求・独立承認・全体検証が揃いspec GO、completedへ更新。以後は完了文書のみ変更。
