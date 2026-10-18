# 実装タスク

- [x] 1. 保存結果の比較可能性を保証する
- [x] 1.1 読取りと完全設定・来歴・環境・期待格子の照合を、合成成果物によるテストから実装する
  - _Requirements: 1.1–1.4_
  - _Boundary: 新CLI、対象テスト、Ruff登録_
  - _Done: 誤設定・重複・欠損が比較へ混入しないことを対象pytestで確認_
- [x] 2. 対比較を再現可能に報告する
- [x] 2.1 指標表、paired差の統計、Markdown/CSV、CLI exitとstdoutの契約をテストして実装する
  - _Depends: 1.1_
  - _Requirements: 2.1–2.4, 3.1–3.2_
  - _Boundary: 新CLI、対象テスト_
  - _Done: 手計算fixtureとsubprocess CLIが成功し、未定義値・不足を明示_
- [x] 3. 研究上の判断と引継ぎを接続する
- [x] 3.1 判断基準・主張と証拠・最終比較の未使用seed方針・実際の適用手順を文書化する
  - _Depends: 2.1_
  - _Requirements: 3.3–3.4_
  - _Boundary: docs、resume、本spec_
  - _Done: 採否は未決のまま、実在未確認の成果物パスを記録しない_
- [x] 4. 独立レビューと回帰検証を行う
- [x] 4.1 Luna highの実装レビュー、必要な修正、対象/全pytest、fresh、品質、spec検査を完了する
  - _Depends: 3.1_
  - _Requirements: 1–3_
  - _Boundary: 本specと必要な修正のみ_
  - _Done: 証拠と制約を記録し、明示ファイルのみcommit/push。mainへmergeしない_

## 実行方針
主担当が下位taskごとに --review off 相当で実装し、全体の独立CLIレビューを最後に行う。ネイティブ子エージェントは起動しない。taskグラフは1.1→2.1→3.1→4.1で循環・並行編集なし。

## 検証の記録
- task1.1/2.1: 対象pytest20件成功（JUnit: ../../venv/refactoring-tests/component-ablation-target.xml）。最初は新CLI未作成でREDを確認。通常sandboxのpytest一時領域走査がPermissionErrorになり、主担当は同じコマンドを通常権限で実行した。
- 実在する基準JSON240件の読取り: 設定一致T120120件、別設定T240120件除外。NPZの参照・存在確認成功。サーバ側の変更runは未取得。
- task3.1: 新手順、主張/証拠表、一次資料による追加比較の候補選別を文書へ接続。採否は行っていない。
- task4.1: 全pytest14396 passed /8 skipped /9 warnings（504.78s）。新しい26テストは全体内で成功。元のfresh16フロー、Ruff lint/format（371ファイル）、Pyright 0 errors、pip check成功。spec naming/progress成功。旧参照/旧golden/srcの数値実装に差分なし。
- 最終証拠レビュー: Luna highでsource hash（482パス、590f78616bfe27d85277709c9cf0b69eb53d5a930b7b8829533b0058a0464118）、JUnit、freshログ、基準120件と全17指標を照合しPASS。全pytest/freshの独立再実行はしていない。ユーザーの権限例外を適用。
- 要件対応: 1は設定・重複・環境/来歴・NPZ・不正JSONの契約テスト、2は手計算統計/欠測/対数/同値性、3はstdout・新規出力・fresh CLIと文書を照合。未解決の指摘なし。サーバ結果の実分析と要素の採否は本specの実装完了と区別する。

## 研究用解析の分離（2026-10-18追加）
- [x] 4.2 解析CLI・専用検証を共通検証から分離する
  - 移設後の専用26件・CLI接続を確認し、共通Ruff登録を撤去する。
  - 共通数値/CI/freshと過去の結果を維持し、全pytest/golden・freshを実測する。
  - 実行手順と引継ぎを更新し、変更仕様と実装の独立CLIレビューを受ける。
  - _Requirements: 4.1, 4.2_
- 分離仕様レビュー初回: Major（checksのparents[2]がルートを指さない）は事実誤認として不採用。旧/新は同じ深さであることをPurePath計算で照合し同sessionへ再判定依頼。Minor2（要求番号の明示、現在手順の更新範囲）は採用して明記。
- 移設後の専用26件成功。通常JUnitは14401 passed/8 skipped/0 failure/errorで専用26件を含まず、旧/提案golden成功。共用fresh16flowsと共有Ruff372files/明示研究lint成功。CLI/testは指定パス置換だけのAST一致。Luna CLIは利用上限で判定なし、代替Haiku5.5 mediumも429週次上限で起動不能。独立Luna highの最終判定待ち。
- 最終独立レビューLuna high（/root/component_analysis_isolation_review）PASS・指摘0。CLI root2行とtest参照2か所以外にAST差がないことを独立確認。help/Ruff独立成功、pytestは一時領域権限で停止。主担当26件/全JUnit14401+8skip/fresh16と通常収集除外を独立照合。追加task完了、学習/保存形式・過去結果・72runのPコードは変更なし。
