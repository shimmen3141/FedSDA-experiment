# 実測: 採用候補の初期ローカル登録

2026-10-07、共有../../venvのWindows CPU基準環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。詳細はdocs/experiments/refactoring-baseline.md。主担当はClaude Code、独立レビューはcodex exec経由のGPT-6 Luna。

## Task1

- RED: 未実装moduleによるModuleNotFoundError、1 collection error/3.06秒、exit1。
- GREEN初回107passed/1failed/4.52秒。失敗は候補共有部へNaNを注入した拒否testで、test側の不変比較がtorch.equal(NaN,NaN)=Falseになったもの。productionは期待どおり拒否し状態も不変だった。注入値を-infへ変更。
- 最終: 対象108passed/3.96秒、exit0。Ruff check成功/format 129files整形済み、Pyright基準venv明示0 errors/0 warnings。
- 実旧対照48条件（class2/4×保有一覧(4,)/(4,9)/(9,-3,4)×現在ID保有/非保有×保有モデルの共有部が同一/別々×別ID既存保留なし/あり）、統計標本8条件（単一標本/singletonクラス/欠落クラス）、optimizer3種、反映先へ接続済み候補1。
- 拒否: 一時ID9、使用済みID3箇所、待機ラウンド数7、owner4×3型、保有0件1、候補/管理器/特徴/ラベル/非有限parameter15。各拒否で一覧record identity・共有部接続先・全parameter値/grad・全optimizer本体/state・統計・送信保留・現在ID・torch/random/NumPy乱数の不変を確認。呼出順1。
- 保有モデルが別々の共有部を持つ条件で、反映先が現在IDのモデル（非保有時は一覧先頭）の共有部だけになることを実旧と値・参照構造で照合。単一標本では実旧torch.varが自由度0のUserWarningを出すため、当該testだけfilterした（旧はNaN分散を0.1へ置換、新は同値を直接生成）。
- 依存境界の全走査testは新module用exact guard未追加で1件失敗（Task2のAST RED）。
- 実Luna Task1 APPROVED、指摘なし。Lunaのpytest独立実行はsandboxの一時ディレクトリ制約で未実施。

## Task2

- AST RED: 注入契約test追加直後20 failed/788 passed/0.90秒、exit1。generic runtime規則が必要symbolを拒否し、全走査testも新moduleで失敗。新module専用の13symbol guard/束縛symbol解決/module import拒否を追加後808 passed/0.64秒。
- 12条件（class2/4×Adam標準/AMSGrad/SGD×共有部更新有無）の実NN接続は初回実行でGREEN。保有2モデルの共同更新→独立した共有部を持つ候補だけの学習（実旧ResidualAdapterMLP.updateと新の単一batch共同更新）→旧登録＋待機設定/新登録→3モデルの共同更新2回→現在ID切替え（後続specの責務のためtest-only接続）→実旧confirm/既存新confirm→(4,9,12)での共同更新。各段階でloss、全parameter値/grad、個別・共有optimizer state、共有参照構造、出力、統計、送信保留値、torch/random/NumPy乱数を照合。登録でresetされた候補の個別optimizerが後続学習でstateを蓄積し、正式ID確認後も同じobjectで継続することを確認。
- 最終: 対象＋AST 928 passed/5.11秒、exit0。Ruff check成功/format 129files整形済み、Pyright基準venv明示0 errors/0 warnings。
- fresh `../../venv/refactoring-tests/adopted_candidate_initial_local_registration_cpu_smoke.py`成功、exit0。class2/4で保有モデル学習→候補学習→一時ID登録（共有値反映/出力不変/個別reset/統計/保留）→後続学習→待機満了→正式ID確認→学習。旧importなし。
- 実Luna Task2 APPROVED、独立928 passed/smoke PASS/Ruff成功、指摘なし。

## Task3

- 検証対象実装commit: dd9f57d（productionはTask1の4ed1156から変更なし）。要求revision2・設計revision2・命名revision3のLF hashは承認値と一致。tasksは承認時hash（revision3）を維持し、check後hashを別fieldへ記録。
- tracked Python＋2goldenの229パスをパス順、パスUTF8＋NUL＋内容CRLF→LF＋NULで連結したSHA256: `463e40ad2e6d917bd20727104799aba585ff7b47a30b7cfe8f9209f068dcc6cb`。同じ手順で前specの検証commit 03f24e6を計算すると227パス/記録値e87d426e…と一致し、手順の同一性を確認した。
- 全適用対象Ruff成功/format129files、Pyright基準venv明示0 errors/0 warnings、pip check成功（No broken requirements）、git diff --check成功。
- 固定旧基準748c3aaからHEADへの差分は、federated_drift_experiment/・2golden・旧回帰test2本・旧実行script・tools/のいずれも空。
- 全pytest: 5678passed/3skipped/1既存warning、123.72秒、exit0。前spec完了時5518に今回の160（対象120＋AST契約40）を加えた件数と一致。旧11条件と最終3条件の固定goldenを含む。既存Windows wrapper3skip、qint8 fixtureのTypedStorage warning。
- golden回帰2本（tests/test_regression.py、tests/test_proposed_regression.py）の単独再実行も2passed/98.75秒、exit0。goldenは更新していない。
- JUnit: `../../venv/refactoring-tests/adopted-candidate-initial-registration-full.xml`、5681tests/0failures/0errors/3skipped。新全体runを実行したとは扱わない。
- 上の全pytestは主担当のClaude Codeセッション（共有venv配下の一時ディレクトリへ書込み可能）での実測。独立レビュー担当Lunaの初回再実行はcodexのworkspace-write sandboxがworktree外の`../../venv/refactoring-tests`への書込みを拒否し、1 failed/5649 passed/3 skipped/28 errors（PermissionError起因）となり一致を確認できなかった。同じWindows PC・同じvenvでの、実行主体のsandbox権限の差である。対象＋AST928/smoke/Ruff/JUnit集計/旧差分空はLunaが独立に一致を確認した。
- Luna再実行（2回目）: worktree内のgit管理外の一時ディレクトリを指定しても、codex sandboxの書込み制限で同じ1 failed/5649 passed/3 skipped/1 warning/28 errors（134.86秒）。失敗はtests/test_proposed_regression.pyがgolden比較用NPZを一時保存する際のPermissionError、errorsはpytest一時ディレクトリ/test生成物への書込み。Lunaの判断でも全て環境起因で、コード起因の失敗は確認されていない。失敗・error以外の5649件は主担当実測と同じく成功。全pytestの成功（特に最終3golden）をLunaが独立に再現できていないことは未解消の制約として残す。前specまでのLuna Task3確認も、対象testの独立実行と全pytestのJUnit照合であった。Lunaが作成した一時ディレクトリは主担当が削除した。

## 要件trace

|要求|証拠|
|---|---|
|1.1|実旧対照48条件。保有モデルが別々の共有部を持つ条件で現在ID優先/一覧先頭の反映先選択を値と参照構造で照合、候補接続先identity、個別optimizer reset|
|1.2|統計標本8条件＋全対照条件で統計全fieldを実旧と一致、呼出順testで損失評価一回|
|1.3|一覧末尾への登録、同ID統計、snapshot値/key順/storage独立、待機と満了境界、別ID既存保留の置換|
|1.4|現在ID・既存record identity・既存個別parameter/grad/optimizer・共有optimizer state・既存統計の不変、torch/random/NumPy乱数不変、12条件の継続学習|
|2.1|一時ID拒否9、全状態不変|
|2.2|学習状態一覧/統計store/送信保留の使用済みID各1、全状態不変|
|2.3|待機ラウンド数拒否7、全状態不変|
|2.4|4owner×None/object/派生型、全状態不変|
|2.5|保有0件LookupError、全状態不変|
|2.6|候補/管理器/共有部構造/特徴/ラベル/非有限parameterの拒否15、共有部値・接続先・optimizerを含む全状態不変|
|2.7|呼出順test（損失評価→統計→snapshot→共有反映→一覧→統計→送信保留）|
|3.1|状態なしruntime関数、exact13symbol AST guardと注入契約40、標本store/計数store/登録確認module/現在ID変更recordを拒否|
|3.2|実旧登録＋待機設定との全値一致、12条件で登録前学習→登録→学習→正式ID確認→学習の全数値/optimizer state一致、fresh新CPU smoke|

実Luna Task3は3回目でAPPROVED（経緯と残る制約はreview.md）。

候補session（生成・初期学習・採否後の計数/標本追加/現在ID切替えと通知）・計算量診断・欠落model復元・通信/new client/runは後続。新たな旧正常経路の不具合は観測していない。旧golden成功は新全体runの検証と区別する。
