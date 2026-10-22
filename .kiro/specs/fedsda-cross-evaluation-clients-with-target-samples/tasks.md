# Implementation Plan

- [x] 1. 対象の標本を返せるclientだけを評価させる扱いを選べるようにする（既定は、まだ変えない）
  - 部品のtest（新しい扱いで評価させるclientと0件の評価の無さ、今の扱いで0件の評価が起きること、判定と評価の標本の選び方の一致）と、設定の検査のtestを先に書き、REDを確かめる。
  - 値の名前と検査、参加者の設定の新しい項目と受け渡し（サーバ・同期・クロス評価）、標本の選び方の切り出しと判定、clientのメソッド、候補の作り方の分岐。全部の参加者の設定の組立てで今の扱いを明示し、旧と比べる入口と、基準版1〜11の「後に足された項目」の表へ足す。
  - 研究ツールとの照合（基準版11の既定の上、4データ）を行い、結果を記録する。
  - 完了条件: 新しいtestが通り、既存のtest（実旧との照合、基準版1〜11）が値を変えずに通り、研究ツールとの照合が一致する。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.2, 5.1_

- [x] 2. 最終構成の既定を替え、研究ツールの基点を固定する
  - 既定と全FedSDA構成の値（候補検証ありの構成は今の扱い）のtestを先に書き、REDを確かめてから、既定を替える。既定に依るtest（runの識別子ほか）を、失敗したものから合わせる。研究ツールの基点の固定に新しい項目を足し、`default`と`default-sample-holding-cross-evaluation`の小さい条件の結果が替える前と同じことを確かめる。
  - 完了条件: 構成のtestが通り、基準版1〜11と実旧との照合が値を変えずに通り、研究ツールの既存の変種の値が変わらない。
  - _Requirements: 2.1, 5.2_

- [ ] 3. 基準版12を足す
  - [x] 今回の依頼範囲のコード部分: 版12の定義・経路の記録と必須の経路・goldenを使わない版11→12の差の説明のtest（golden生成・設置・定数の切替は主担当が行うため、この親taskは未完了として残す）。
  - 経路の記録の新しい項目（版12から）、版12の定義と必須の経路、版11→12の差の説明のtestを足し、commitしてから、Windowsの基準環境とWSLでgoldenを生成して足し、現在の版を12へ上げる。
  - 完了条件: 版1〜12の照合と、版11→12の差の説明のtestが、WindowsとWSLで通る。
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 4. 文書を合わせる
  - 手法の説明、基準（版12）、実験の入口、LEGACY-022・IMPROVE-028、再開案内。
  - 完了条件: 文書が新しい扱いを説明する。
  - _Requirements: 6.1_

- [ ] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL。
  - 完了条件: 実装レビューの判定と指摘の採否、全pytest（件数・exit・JUnit）、Ruff・Pyright、WSLの照合の結果を「Implementation Notes」とspec.jsonへ記録し、レビューがPASSで、全pytestが失敗0。
  - _Requirements: 全部_

## Implementation Notes
- 最終の対象検証（未commitの作業ツリー）: `../../venv/Scripts/python.exe`、`PYTHONPATH=src;.`、OMP/MKL各1、MNISTは指定のデータディレクトリ。新spec、`test_model_cross_evaluation.py`、`test_model_clustering_and_consolidation.py`（同期を含む）、`test_fedsda_run_server.py`、`test_fedsda_final_configuration_run_settings.py`、`test_single_run_dependency_boundaries.py`、`test_run_settings_validation.py`、`test_experiment_sweep.py::test_run_identifiers_are_readable_stable_and_unique`をまとめて実行し4534 passed（104.49秒）。その後、標本判定がtorchの乱数も消費しない検査を足し、新specを再実行して20 passed（9.09秒）。既定切替後の版1〜11の代表の小さい`fedsda`の数値照合11件・Windows/WSLの全ケースの設定と構造の照合11件・ケース登録1件は、計23 passed（10.10秒）。新しい版の差のtestは前述の24 passed。構成の実行可能部分は前述の63 passed / 10 skipped（各実行の件数は重複を含むため合算しない）。ログは`results/_sol_tmp/{final_targeted_tests,new_spec_final,previous_baseline_final,task2_available_tests,task3_tests}.log`。
- 品質検査: 指定どおり`../../venv/Scripts/python.exe -m ruff check src tests/refactoring tools`成功、同じ範囲の`ruff format --check`成功（390 files already formatted）、`git diff --check`成功、spec.jsonはJSONとして読込み成功。仕様書群3ファイル・旧実装・旧の回帰test・既存golden・AGENTS.md・CLAUDE.md・agent-handoff.mdに差分なし。版の定数2つは11、v12ディレクトリは存在しない。commit/push/stashは行っていない。
- 未検証と引継ぎ: 全pytest、goldenの生成と全ケースの数値照合、旧の回帰2件、WSL、実装レビュー（task 5）は今回の指示に従い未実施。fresh process smokeも未実施（変更した組立てでは旧の選び方を明示してある）。構成の実旧全体run照合70件は前述のpytest一時ディレクトリの権限エラーで実行できていない。追加で試したPyright（相対・絶対の`--pythonpath`）は、venvのNumPy・Torch等を解決できず198 errorsで失敗したため、型検査成功とは扱わない。これらと上限5未満の扱いを主担当へ引き継ぐ。
- Task 3（コード部分）: testを先に追加し、版12が未登録であることによるRED 1 failedを確認した。`SUPPORTED_BASELINE_VERSIONS`へ12、版11と同じ22ケース、版12からのFedSDAの経路の記録2項目と必須の経路、版11の0件評価を実行の記録から数える補助を追加した。版11→12の設定の差・両版をその場で実行した結果・0件評価の対照のtestは24 passed / 452 deselected（skipなし、200.54秒）。`BASELINE_VERSION`・`BEHAVIOR_BASELINE_VERSION`は11のまま。goldenの生成・v12ディレクトリへの配置は行っていない。未配置で失敗する既存testを明示実行して確認した: `test_committed_goldens_cover_both_reference_environments[12]`（1 failed、Windows/WSLのJSONが2つ必要だが0）。版12の数値goldenの照合そのものは未実施。
- Task 2: 構成ごとの選び方のtestを先に足し、RED 14 failed / 1 passedを確認した。その後、最終構成を新しい選び方に替え、候補検証ありの構成だけ旧の選び方を明示した。既定に依る既存testの7件の失敗（既定値6件、候補検証ありの構成の差1件）を確認して期待値を合わせた。新spec・既定・候補検証あり・構成の実行できるtestは63 passed / 10 skipped / 70 deselected（10 skippedは重い照合の結果を読む網羅のtest）。実旧との全体run照合70件は標準のpytest一時ディレクトリへのアクセス拒否でsetup error。workspace内の専用`--basetemp`でもアクセス拒否となり、終了時のcleanupでもPermissionErrorになったため、この70件は未検証として引き継ぐ。設定の構成と、tmp_pathを要さない構成の経路は成功している。
- Task 2の研究ツールの固定の確認: 一時script `results/_sol_tmp/compare_remaining_research.py before`で既定切替前を保存し、切替後に`check_research_base.py`で比較した。SINE2・seed 0・10 client・500標本・間隔50、`default`と`default-sample-holding-cross-evaluation`の両方で、実行秒数`seconds`を除く全結果項目が一致した（全指標・通信・計算回数・予測/パラメータdigest・診断）。最初の比較は可変の実行秒数も含めたため失敗し、それを比較対象から外して確認した。切替前後のJSONは`results/_sol_tmp/research_{before,after}_default_change.json`。
- Task 4: File Structure Planの6文書（手法・基準・実験入口・LEGACY-022・IMPROVE-028・resume）を更新した。版12はコード準備済み・golden未配置・定数は11と明記し、生成元commitや最終照合の件数・結果は「主担当が追記」とした。文書だけの変更なので新しい動作testによるREDは対象外。
- Task 1の研究照合（既定の切替前、基準版11の既定）: 一時script `results/_sol_tmp/compare_research.py`・`compare_remaining_research.py`。全条件10 client、OMP/MKL各1 thread。SINE2（1500標本・seed 0・間隔50）は呼出し9回・候補列26件・選択列76件・最大候補10・統合18件、CIRCLE2（1500・0・50）は5回・16件・58件・9・5件、SEA4（5000・1・50）は6回・15件・39件・10・7件、MNIST2（2500・0・100）は2回・5件・13件・8・2件。候補列は上限適用前の各対象、選択列は候補モデル×対象ごとに記録のclient列を直接比較した件数。4データとも、全部の候補列・選択列、精度、通信量の全項目（上下のbyteとメッセージ数を含む）、予測記録digest・最終パラメータdigest、実行・準備の計算回数が完全一致した。標本条件を満たすが対象を保有しない（client、対象）の組は全呼出しで0件。各条件に候補数3超と統合がある。SEA4の1500/2500標本・seed 0・間隔50は、両経路の値は一致したがクロス評価・統合が0回だったため採用せず、既存の研究結果からイベントのある条件を選び直した。結果は同ディレクトリの`research_comparison*.json`（Git管理外）。
- 実装担当の範囲（2026-10-23、基点`92fd126e`）: 最新のユーザー指示により、task 1〜4を直接実装する。再委譲・実装レビュー・commit/push/stash・golden生成は行わない。task 3はコードだけで、現在の基準版の定数2つは11のまま。仕様書群（requirements.md・design.md・naming.md）は変更しない。
- Task 1のRED: 新しい部品testを先に追加して実行し、`target_model_cross_evaluation_samples_are_available`の未定義による収集失敗（1 error）を確認した。標本の選択を共有し、必須の参加者設定からサーバ・同期・クロス評価まで渡し、旧と比べる入口と版1〜11の補完表で旧の扱いを明示した。新規testは5 passed（候補順・上限によるsample・乱数状態・候補なし・保有モデルなし・0件の評価の対照・標本選択の境界・不正設定の先行拒否）。対象moduleと依存境界3168 passed、client/全体run/設定検査1401 passed。版1〜11の代表の小さい`fedsda`ケースは各版の既存goldenに一致（11 passed）。版ごとの全ケースのgoldenの実行は、長い検証を実施しないという最新の指示により行わない。
- 設計の前提に関する引継ぎ事項: 現在の`FedsdaRunClientScalarSettings.maximum_cross_evaluation_sample_count`の検査は正数だけを求め、1〜4も受理する。これを指定すると、標本の条件を満たすclientでも、上限で抜き出した後に既存の評価下限5件を満たさず0件を返す。最終構成の既定の上限では起きない。本specは評価そのものと件数の上限を対象外としているため、この実装では追加の組合せ制約を入れず、承認済み設計の標本選択・候補条件を維持する。上限5未満の拒否を足すかは主担当が判断する（ユーザーへ確認事項を提示済み。承認を得たとは扱わない）。
- 進め方（2026-10-23のユーザー指示）: トークンの節約のため、実装（task 1〜4のtest・source・goldenの生成・文書）はGPT-6.1-sol（独立CLI、`codex exec`）が行い、主担当（Claude）は、仕様書群の作成と、最後に意図どおりかの確認（差分と仕様書群の照合、対象testの実行）を行う。共通の規約（agent-handoff.md）の「実装は主担当が行う」とは違う、このspecの扱い。実装レビュー（task 5）は、実装した担当とも主担当とも別の独立CLIで行う。
- 仕様レビュー1回目（GPT-6 Luna、effort medium——ログで確認——、session `01a165a9-99c0-7011-9c98-da488f58bfdc`、対象は未commitの仕様書群、基点`ca2c1210`）: CHANGES REQUIRED。Major 3件・Minor 3件・任意1件を全部反映した。
  - Major 1（要求1.5が、対象のモデルを保有しないclientの0件の記録と矛盾）: 要求1.5を「評価させたclientが対象のモデルを保有している場合」に限り、その場合を除く理由と、goldenの必須の経路で確かめることを書いた。
  - Major 2（候補の順・上限を超えたときだけの無作為の選択・候補が無い対象の検証が無い）: 部品のtestに(i)〜(iv)を足した（Validation Design）。
  - Major 3（必須の経路の範囲が曖昧）: 除くケースを4つの名前で列挙し、記録の行の数の項目を足して、「空でない」ことの確かめ方を定めた。
  - Minor 4（照合の要約だけでは候補の選び方の違いを捉えられない）: 照合にメッセージの数と、予測の記録・最終のパラメータの要約を含め、候補の選び方が違えば一致しない理由を書いた。条件を、候補が上限を超える同期を含むものにした。
  - Minor 5（版1〜11の照合の範囲）: 既存の版ごとの組立てと照合のtestを全部の版で通す、と書いた。
  - Minor 6（「包みは働かない」の書き方）: 製品が同じ候補の集合を作り、包みの対象をクロス評価で呼ばない、と書き直した。
  - 任意 7（設定の値と呼び方の対応）: 要求の冒頭に表を足した。
- 仕様レビュー2回目（GPT-6 Luna、effort medium——ログで確認——、session `01a165ac-1f4f-74f3-bc8d-19cf16ad3d1e`）: CHANGES REQUIRED。Major 3件・Minor 2件を全部反映した。
  - Major 1（標本はあるが対象のモデルを保有しないclientが、新しい扱いでも候補に入る）: 製品の候補の条件に、対象のモデルを保有すること（保有モデルの登録簿）を足し、要求1.5を無条件にした。研究ツールは標本の条件だけを使うが、最終構成では配布で全clientが全グローバルモデルを保有するので同じ候補を作る見込みで、照合（要求5.1）で確かめる。部品のtestに(v)を足した。
  - Major 2（必須の経路が、記録が0行のケースでも通る）: 統合が起きる2ケースで、記録の行が1以上であることも必須の経路にした。
  - Major 3（候補の列の順の正本）: `run_clients`の順と明記し、上限以下は全部を順に、上限を超えたときだけその列から`sample`する、と要求・設計で揃えた。
  - Minor 4（照合するメッセージの数の値）: `communication_volume`の下り・上りのバイト数とメッセージの数と明記した。
  - Minor 5（旧goldenの設定の組立ての経路）: `build_case_settings`が「後に足された項目」の表で置き換える経路を明記した。
- 仕様レビュー3回目（GPT-6 Luna、effort medium——ログで確認——、session `01a165ad-c480-7ec2-85a5-36f96ac71874`）: CHANGES REQUIRED。2回目のMajor 3件・Minor 2件の解消は確認された。Major 1件（製品の候補の条件——保有モデルであること——と研究ツールの条件の等価性を、指標の一致だけでは確かめられない。上限以下では候補の列の違いが指標に出ない）: 照合に、呼出しごと・対象ごとの、上限を当てる前の候補の列と評価させたclientの列の直接の比較と、標本の条件を満たすが対象を保有しない組の数が0であることの確認を足した（要求5.1、設計の研究ツールの節）。
