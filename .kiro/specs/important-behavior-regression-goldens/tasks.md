# Implementation Plan
- [x] 1. 新実装だけから基準を取得・比較する検証を用意する
  - ケース定義・snapshot・環境選択・版/破損拒否・項目比較をtest先行で接続する。
  - metrics/記録列/parameter/settings変更が項目を示して失敗し、未知環境でも構造検査を実行できる。
  - _Boundary: golden test infrastructure_
  - _Requirements: 1.1, 1.4, 2.1, 2.4, 3.2, 3.4_
- [x] 2. 重要経路の基準を2環境で固定する
  - 独立process2回の一致・経路確認と新規出力の拒否を検査して基準を生成する。
  - Windows/WSLの代表17ケースが各環境goldenと一致し、候補採否/再利用/統合とFedDrift/FedAvgの境界処理を確認できる。
  - _Boundary: golden generation and numerical regression_
  - _Depends: 1_
  - _Requirements: 1.2, 1.3, 2.2, 2.3, 3.1_
- [x] 3. CIと既存回帰の検証入口へ接続する
  - 数値markerを登録し、構造検査を必須、ホスト数値照合を診断へ接続する。
  - 版の更新/環境別追加の運用を基準手順へ接続し、全pytest・品質・fresh・固定旧不変を確認する。
  - _Boundary: validation integration_
  - _Depends: 2_
  - _Requirements: 3.3, 3.4_

## Implementation Notes
仕様作成時点: 要求gate・設計gate・task graph inline sanityを確認した。独立CLI仕様レビューは待機中、テスト/実装前。

独立仕様レビュー初回FAIL: Major2を採用し、差分表示は項目名/field pathまで、FedSDA追加2ケースそれぞれの必須経路を要求/設計に明記。Minorの環境別命名規則を明記、13構成はFedSDA11＋FedDrift/FedAvgと明記（指摘の13すべてFedSDAという解釈は採らない）。重大箇所の再確認待ち。

仕様修正箇所のLuna medium独立再確認PASS。実装準備可。下位taskは主担当が順次実装しtask別レビューoff、全task後に独立実装レビューを行う。

Task1: 項目比較のno-opが相違4種類を見逃すREDを確認後、比較/shape/dtype/digest/13構成登録/旧依存なしの11testが成功。小条件のbatchはcandidate batchと同値の契約があるため一緒に4へ設定。FedDriftのgenerated tailは観測されないことをsourceで確認し、要求1.3の終端表現を修正、影響task2の前に再確認する。

Task1完了: 23passed。構造・未知版・破損・重複環境・NaN拒否・Unicode dtypeを含む。既定tempのACLエラーは検証用tempと通常の権限承認で解消。指標NaN拒否/Unicode dtype受理のRED後にhelperを修正。FedDrift終端表現の独立確認PASS、Task2へ進む。

Task2生成: Windows/WSLとも17ケースを各2つの新processで実行し全snapshot一致、既存先への生成は拒否する。基準sourceは7bb91d8312ddd7ba8c6d01d7275e6c2f4222a010。最初のWindows呼出しは主担当が誤ったcommit文字列を渡しHEAD不一致で出力前に拒否、実HEADから取り直した。WSLはWindows形式worktree参照をnative Gitが読めず、同じ対象をgit.exe -Cで読取る対応を確認した（Git設定変更なし）。生成ログはvenv/refactoring-tests/behavior-{windows,linux}-generation.log、指定出力は同directoryのbehavior-v1-{windows,linux}.json。

Task2完了: 両環境で48passed（数値17含む）。その後、未完了候補を棄却に計数しないtestを先に追加しREDを確認、coverageを修正（重要2ケースの基準に未完了行はなく既存基準値は維持）。全49件の最終確認は独立レビューと全pytestで行う。

Task3実装完了: markerで構造32件/数値17件のcollection分離を確認、構造32passed。CI必須stepは旧2数値回帰を従来通り除外し新markerだけ除外、診断stepは旧2回帰＋新数値17ケースを指定。ホストCI自身は未実行。Windows生成2回一致とWSL生成2回一致を各ログで確認。仕様のWindows機種名は実環境のAMD64に合わせたファイル名規則へ確定した。独立実装レビューと全pytestは未実施、spec完了ではない。

レビュー前機械検査: namesは新source公開名なし（test2fileは対象外）、progress3/3、共用fresh直接exit0、Pyright0、pip check成功、Ruff format348files。未完了候補testのimport整形1件をRuffで修正後lint成功。固定旧748c3aaに対する旧実装/回帰test/golden/toolsの差分は空。

独立実装レビュー: Haiku 5.5 medium指定、session 40c8a477-a1d1-4b32-b6cf-4a0647bec720、is_error=false/実モデルclaude-haiku-5-5、PASS。Blocker/Majorなし、Minor4/任意1。3140passed（対象49＋依存3091、数値17/skip0）、fresh直接exit0、Ruff/format348、progress3/3を独立実行。全pytest/WSL/生成CLI/ホストCIは独立再現していない。主担当WSL修正後49passed/41.90秒。

Minor1は指標の非finiteをnullへ統一し記録列は3種類を区別する既存の実装/文書へ設計文面を合わせた。Minor2は13小runが構成固有イベントを網羅しない限界と既存対照/部品test併用を基準手順へ追記。Minor3は実Windows FedAvg goldenの設定・指標・列・parameterの各コピーを変えた検出力test4件を追加。Minor4のlist要素位置の表示は要件の項目/field pathまでを満たすため追加しない。任意のsubprocess check=True実挙動は標準ライブラリの契約を重ねて検査せず、実2process生成成功・失敗注入での無出力を証拠とする。Minorだけであり再レビューは不要。

レビュー反映後: Windows構造36passed/数値17deselected、Ruff lint/format348成功。主担当WSLでは修正後49passed、追加の実goldenコピー検出4passed（合わせて追加53件）。CI collectionは構造/非数値36と数値17。全pytestは規定の8worker/loadfileで進行中、期待成功件数は前spec13724＋53=13777。

最終検証: 固定Windows venvの全pytestは13777passed/8skipped/8warnings、906.68秒、exit0。前spec13724＋追加53と一致。JUnit13785件、failure/errorとも0、旧回帰2fileと追加53件すべて成功。JUnitはvenv/refactoring-tests/important-behavior-regression-goldens-full.xml、logは同directoryのbehavior-golden-full.log（Git管理外）。

検証source/test/新golden/CI/基準手順commitは593aed77340b6cc9937047699838db564c21b338。tracked Pythonと旧2goldenの452pathをLF正規化したSHA256は43d6587d776d1e71e3a3af085d982dae4329b3d868c87861b953f81f4a5fa1c3。新2goldenはこのhashの範囲外なので、個別LF SHA256をspec.jsonのbehavior_golden_sha256_lfへ記録した。完了文書commit後にsource identity・JUnit・新2golden hash・clean treeを確認する。変異testとホストCIは未実行。固定旧748c3aaから旧実装/旧回帰test/golden/toolsの差分は空。
