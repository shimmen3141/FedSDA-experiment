# Implementation Plan

- [x] 1. 終端のbuffer確定から条件付き最終配布まで接続する
  - 実旧flush/run_round oracleを起動確認して、入口stubでREDにする。
  - 全client preflightと順序を保持し、終端の回収・集約・配布と待機pending昇格を実装する。
  - 通常同期の集約を共通helperへ整理し、既存対照と終端の全状態/通信/3乱数対照、拒否/再呼出しが成功すれば完了。
  - _Boundary: runtimeのFedDrift終端進行_
  - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 3.1_
- [x] 2. factoryからの連続実行と独立起動を接続する
  - 実旧初期準備から観測/通常同期/終端へ対照をつなぐ。
  - 共用freshへ終端経路を接続し、exact依存・対象/既存同期/依存境界/freshを確かめる。
  - _Boundary: 準備済み参加者から終端への接続検証_
  - _Depends: 1_
  - _Requirements: 3.2_

## Implementation Notes

- task graph gate inline PASS。既存部品と参加者factoryは完了済み。仕様書群の独立レビュー後にmanual --review offで順に実装する。実装レビューは全task後1回、Minor以下は修正/採否のみ、Major/Blockerは修正箇所だけ再確認する。最終全pytest/source identityはその後の完了ゲート。

- 仕様書群レビュー: Haiku利用不能のためGPT-6 Luna medium（`/root/feddrift_finalization_spec_review`）。基点HEAD `2e2affa`の未コミット仕様書群と実旧/新runtime読合せでPASS、指摘なし。主担当受入。実装/テストの独立実行は未実施。

- task 1: 実旧oracle起動確認後、入口stubのNotImplementedErrorで2件RED。終端69件と既存同期149件で218 passed（15.84秒）。NaNを含む拒否前後の比較器だけ1件失敗し、debuggerによる読取り確認（`/root/finalization_test_diagnosis`）後float NaN同士を一致扱いに修正した。sourceの拒否は初回から成功。通常/終端の全状態・通信・3RNG、grad無効・待機pending・再呼出し、全flush前拒否を確認。

- task 2: factoryの実旧setup→2時刻の観測/通常R同期→部分batch→終端がsine2/mnist2×SGD/Adamで一致。対照fixtureの旧関数引数順とverboseを既存helperの呼出しへ揃えた。終端89件＋既存同期149件＋依存境界3082件で3320 passed（22.26秒）。ready snapshot後の統計更新とcached正重みも追加対照済み。新module/import追加なしのため既存exact依存登録をそのまま検証。共用freshはexit 0、FedDriftの準備/観測/同期/終端PASS、旧/test importなし。Ruff check/format（317 files）、Pyright 0 errors/0 warnings、pip check、names、progress、固定旧差分なし。ready payloadを全回収通信前に集め、Optionalの型境界も明示した。独立実装レビューとその後の全pytestは未実施。

- 実装レビュー: GPT-6 Luna medium（`/root/feddrift_finalization_implementation_review`）PASS、重大度のある指摘なし。基点HEAD `2e2affa`の未コミット差分。独立対象/同期/依存3320 passed（27.29秒）、対象Ruff/diff --check成功。freshのFedDrift終端等はPASSしたが末尾の既存artifact保存の一時NPZ作成がsandbox PermissionErrorで失敗し、TMP/TEMP変更でも再現。fresh全体は独立再現なし。全pytest/Pyright/pip/names/progressも独立実行なし。主担当の成功証拠と実旧行照合を受入れ、最終全pytestへ進む。

- 独立実装レビュー後の最終検証（2026-10-12）: 全pytest 13303 passed / 4 skipped / 8 warnings、exit 0（670.85秒、-n 8 --dist loadfile）。前spec 13214件へ対象89件を加えた件数と一致。JUnit 13307件、failure/error 0、対象89件、既存11ケース/提案3ケースgolden回帰、共用fresh成功。skipはPOSIX bashなし3件と既存回帰へ委ねる環境golden1件。Ruff check/format（317 files）、Pyright 0 errors/0 warnings、pip check、names、progress、diff --check成功。固定旧748c3aa/golden/tools差分なし。全8要求のbuffer確定/条件付き回収/重み保持/待機昇格/無変化拒否/factory接続を主担当が実装と対照へ照合。source/testはこの検証後に変更しない。独立全pytest再現なし、独立freshはsandboxの既存成果物保存で未完了（主担当単独freshと全pytest内freshは成功）。WSL/変異は未実施。全体runの指標/成果物/CLI/golden移植は後続。ログ/JUnitは`../../venv/refactoring-tests/feddrift-stream-finalization-full.log`と同名`.xml`。
