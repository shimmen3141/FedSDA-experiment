# Implementation Plan

- [x] 1. 環境依存の回収を独立した実行資源操作として実装する
  - テストを先に作り、未対応環境・binding失敗・戻り0/1・process内再利用を確認する。
  - native呼出しが対応環境だけで発生し、他では追加設定なしで継続することを完了条件とする。
  - _Requirements: 2_
- [x] 2. FedSDAの正常同期末へ接続する
  - 同期成否・計算量snapshotとの順序をテストで先に固定する。
  - 正常同期ごとに1回、例外同期では0回であり、引数・状態・RNG・既存記録を維持することを確認する。
  - _Requirements: 1, 3_
- [x] 3. 通常のLinux掃引と回帰を検証して完了する
  - 逐次・2workerの実NNで回収有無の全保存値と最終paramsを比較する。
  - 全task実装後の独立CLIレビューを受け、指摘を反映して全pytest・fresh・品質検査を実測する。
  - 適用範囲と未対応環境を文書化し、通常コマンドと保存形式の維持を確認する。
  - _Requirements: 1, 2, 3_

## Implementation Notes

仕様書群の独立CLI Luna highはPASS、Blocker/Major/Minor0。sessionとログはspec.json。PASS後にtestを先行作成し未実装import失敗を確認、実装した。新しい依存を境界testへ最小登録後、対象/既存計測/依存境界3140件成功。型検査は固定venvの絶対pathを明示した許可済み実行で0 errors。全体品質Ruff/format成功。

独立実装CLI Luna highはソースレビューPASS、Blocker/Major/Minor0。対象3140件と対象Ruff/formatを独立実行し成功。freshはTemp権限で停止したため、再試行せず主担当の16フロー成功ログを照合した。主担当Pyrightログも照合。レビュー時点の未完了だったLinux対照・全pytestを、下記のとおり主担当が実測した。

- Linux Ubuntuを明示し、Python3.14.4/NumPy2.4.6/PyTorch2.12.1+cpu/glibc2.43で通常CLIを実行。逐次1000/client・seed0と、2workerの100/client・seed0/1で回収有無を比較した計6runはexit0。全3対の完全設定・全指標・各133配列dtype/shape/bytes・最終paramsが一致。逐次は各20同期、2workerは各worker2同期、両processでnative回収対応と呼出しを確認。接続確認であり、多worker性能測定とは扱わない。
- 証拠と一時scriptは`results/fedsda-memory-diagnosis/runtime-integration-validation/`。`output-v2/validation.json`と`completion-audit.json`を元データで照合。検証対象src bytes hash（相対path+bytes）は`6cdbbbf7139a06435dd72607815186cde5236fd6f63a56e674259c66ddeef931`で全6runと現作業ツリーが一致。旧試作runner・固定hashは変更なし。
- 主担当の固定Windows全pytestは14412 passed/8 skipped、exit0、779.64秒。直前の全pytest14401 passed/8 skippedに新規11件を足した数と一致。旧11/最終3goldenを各1testcase内で確認。JUnit `../../venv/refactoring-tests/runtime-memory-full.xml`、ログ同`runtime-memory-full.log`。
- 主担当freshは未変更の共用scriptで16フロー成功、旧/test module importなし（同領域`runtime-memory-fresh.log`）。全体Ruff/format・Pyright・pip check成功。仕様命名の登録検査成功。
- 汎用`spec_checks identity`は、過去に追加済みの研究toolsも固定旧の比較対象へ含めるためNG（加えてcommit前のdirty状態を検出）。これを成功とは記録しない。旧本体・2golden・2回帰testの748c3aaからのcommit済み/作業ツリー差分は別途空を確認。当specでtools、旧実装、golden、共通loop/sweep、成果物schemaは変更していない。

完了証拠の独立CLI Luna highは、元3対の全保存値/params・source hash・JUnit・fresh成功マーカー・旧差分を照合。判定はFAIL、Blocker0/Major0/Minor1で、PID別回収数のassert不足を指摘した。各PIDの回収点が逐次20回・spawn各2回で、その全回収点の後にrun返却が1回あることをassertへ追加し、主担当の再照合は成功した。個々のround番号はtraceにないため、正常同期との位置関係はレビュー済みsourceと同期順序の単位testを合わせた証拠とし、個別roundの実測traceとは表現しない。Minorのみの修正なのでユーザー指定に従い再レビューなし。製品source/testは実装レビュー・全pytest後に変更していない。

仕様・実装・完了証拠のレビューsession/log/元判定と修正後の扱いはspec.json。対象sourceの検証commitはcommit後に追記する。全task完了、mainへマージしない。
