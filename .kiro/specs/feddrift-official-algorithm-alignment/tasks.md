# Implementation Plan

依存: task 1 → task 2 → task 3 → task 4 → task 5 → task 6。実装はGPT-6.1-sol（task 1〜3、task 5の文書の下書き）。task 4のgoldenと、task 6の確認は主担当。

- [x] 1. 設定の項目を足し、版1〜13の今の値を明示する（既定は、まだ変えない）
  - 設定の検査のtest（不正な値・型、Validation Designの(vi)）を先に書き、REDを確かめる。
  - naming.mdの項目と値、検査（拒否する組合せ）。全部の`FeddriftRunClientSettings`・`FeddriftServerSettings`の組立て（既定、構成、旧と比べる入口、golden case、test・fresh script）で今の値を明示する。
  - δのデータ別の既定の対応表を置く（まだ既定にはしない）。
  - 完了条件: 新しいtestが通り、既存のtest（旧との照合、版1〜13のgolden）が値を変えずに通る。
  - _Requirements: 1.4, 2.3, 4.2, 5.1, 7.3_

- [x] 2. clientの検出の塊の判定（D01・D02・D03）
  - Validation Designの(i)(ii)のtestを先に書き、REDを確かめてから実装する。
  - 誤り率の評価、最小のモデルの選択と検知、複写元、基準値。誤り率の予測の規則は今の予測の関数に合わせ、根拠を記録する。
  - 完了条件: (i)(ii)が通り、今の値の設定の既存のtestが値を変えずに通る。
  - _Requirements: 1.1, 1.2, 2.1, 2.2_

- [x] 3. サーバ側（D01の距離・D04・D08・D07の上限）
  - Validation Designの(iii)(iv)(v)(vii)(x)のtestを先に書き、REDを確かめてから実装する。
  - クロス評価の誤りの数の集計と距離（0での下限）、今の時刻に割り当てたモデルの集合と学習・集約の対象、統合の重み。集合の伝達と通信の計数の判断を記録する。
  - 完了条件: (iii)(iv)(v)(vii)(x)が通り、今の値の設定の既存のtestが値を変えずに通る。
  - _Requirements: 1.3, 3.1, 3.2, 3.3, 3.5, 4.1, 5.2, 6.1, 6.2_

- [ ] 4. 既定を替え、基準版14を足す
  - 既定のFedDriftの設定（新しい値の全部、上限＝clientの数、δのデータ別の既定）と、掃引で閾値を指定しないときの既定のtest（(ix)）を先に書き、REDを確かめてから既定を替える。
  - `BEHAVIOR_BASELINE_VERSION = 14`、runの設定の記録の新しい項目。既定に依るtest（runの識別子ほか）を直す。
  - goldenの版14（Windows・Linux）は主担当が作り、版14のFedSDA・FedAvgのcaseが版13と同じ値であることを確かめる（(viii)）。
  - 完了条件: 全部の対象のtestと版1〜14のgoldenが通る。
  - _Requirements: 1.5, 1.6, 6.1, 7.1, 7.2, 7.4_

- [x] 5. 文書
  - `docs/experiments/differences-from-feddrift.md`を書き直す（そろえた差・開示する差・根拠・監査と論文の参照）。`docs/experiments/refactoring-baseline.md`の版の表に版14。
    `docs/reference/options.md`は直接編集しない（必要なら生成の道具で再生成する）。
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 6. 確認（主担当）
  - 全pytest、WSLのgolden、pyright、ruff。独立の実装レビュー1回（文書だけの修正は再確認を要しない）。

## Implementation Notes

- 実装担当の範囲（2026-10-27、基点`720482d9`、未コミット）: ユーザーの今回の指示でtask 1〜5のコード・test・文書と版の定数の切替を担当する。版14のgoldenの生成・配置、全pytest、WSL、独立実装レビュー、commitは主担当が行う。下位taskごとのレビューは規約の`--review off`。task 4のコードは実装済みだが、golden未配置のため完了条件を満たすまではcheckboxを付けない。
- TDD: task 1は設定のtestを先に書き、正常な新設定を組み立てる段階の未対応引数による1 failed / 4 passedを確認。実装後、既存の参加者組立testと合わせ48 passed。task 2は確信度・同値の順序・複写元と基準値のtestを先に書き、未対応引数で3 failed / 5 passedを確認。実装後、既存の検出・処理・clientの実旧対照と合わせ200 passed。task 3は距離の下限、未使用モデルの除外、現在の割当を含む重み、実バッチ後のクロス評価を先に書き4 failed / 8 passedを確認。実装後、既存の同期とround学習を合わせ224 passed。上流の対照helperが入れる標本と計数は、新testの準備で明示的に空にした。W=1の既存の隔離の期限は本番の作成・登録・配布・次時刻のクロス評価で確認した。task 4は6データの既定と明示引数・掃引のtestを先に書き7 failed / 14 passedを確認してから切り替えた。task 5は文書のみ。
- 判断（命名）: naming.mdの設定項目・値は変更していない。新しい公開関数・moduleは作らず既存の責務へ接続した。既存の結果のfield名の`bounded_loss`は保持し、比較の基準を選ぶ経路では選んだ基準の値が入ることをdocstringへ明記した。内部の損失統計は引き続き有界損失で、検知・基準値・距離だけが誤り率になる。
- 判断（混在）: 完全なrun設定では旧5項目の組、または新5項目の組だけを受け付ける。単独切替えは受け付けず、比較の基準・基準値・重みの解釈を混ぜた未検証の実験条件を増やさない。参加者設定で混在とclient/serverの基準不一致を拒否し、サーバの直接組立でも基準不一致を拒否する。新しいserverの項目とW≠1も設定で拒否する。
- 判断（D04）: `prepare_timestep`の統合・登録によるIDの付替え後の現在の割当IDの和集合を、各roundの全clientへ引数で渡す。旧設定では絞らない。割当の報告は既存のprepareの上り通知へ含め、集合の配布は時刻ごとに各clientへ下りの軽量通知1件として数える（`CommunicationVolumeRecordStore`は小さい制御情報をmessage件数へ数え、parameterの値数・byteへは入れない）。対象外のモデルはroundの学習・集約・上下parameter転送から外し、保有する分類器とoptimizerをそのまま残す。別clientだけに現在割当があるモデルも、標本数の条件を満たす全clientで学習する。
- 判断（2値の誤り）: 既存の`predict_class_labels_from_prediction_scores`を使う。1出力は確率`> 0.5`だけを1（0.5は0）、多クラスは最小番号のargmax。同じforwardの出力から求め、クロス評価で余分なforwardを増やさない。D08は統合前の全clientの標本数を数え、0件なら保存済み重みを渡す。統合後の集約は連結後の件数で重みを置き換え、二重加算しない。
- 版1〜13の固定は版13の前例と同様に、helperの後から足された設定の表で旧値を指定し、保存済み設定との照合時はその旧値であることを検査してから新項目だけを除く。閾値は通常0.1、既存の経路確認case `feddrift-drift-tail`の明示値0.03も維持する。版14の既定の閾値はデータ別で、掃引の未指定はbuilderの値を保持する。FedSDA・FedAvgの版13/14の完全な設定が等しいこともtestで確認した。
- 仕様との食い違い（主担当の判断待ち）: 要求1.6の「版1〜13の設定は0.1」と、既存の`feddrift-drift-tail`の明示値0.03は一致しない。このcaseの閾値変更は止め、今回の「版1〜13のgoldenの値を変えない」というユーザー指示に従い0.03を保持した。requirements.md・design.md・naming.mdは変更していない。通常caseの0.1固定とほかの実装・検証は進めた。
- 途中の検証: FedDrift全対象・設定・掃引・依存境界は5464 passed / 113 skipped（51.05秒）。skipはアクセス拒否になるtmp_path系を明示的に除いたものと、既存の環境専用・先行testに依存する確認。後のサーバ直接組立の検査と計数assertの追加後、FedDrift新test・同期・共用クロス評価・通信重複除去・来歴は233 passed / 1 skipped（58.73秒）。新specのtestはその後の歴史設定の確認を含め29 passed（3.04秒）。版14のFedDriftの2caseも、goldenを作らずメモリ上で実行し必須経路を通った。
- 環境制約: 通常のpytestは一時ディレクトリの`PermissionError`で16 setup errorsになった。workspace内の新しいbasetempでも同じアクセス拒否を短く確認し、再試行せずtmp_path系をskipする実行用plugin（`results/_feddrift_available_checks.py`、製品・testは未変更）で実行した。Pyrightの通常の`--pythonpath`ではvenvのnumpy/torchを解決できず198 errors。変更したsourceだけを対象に、同じ固定venvの`Lib/site-packages`をextraPathsへ明示した一時設定では0 errors / 0 warnings（`results/_feddrift_pyright.json`）。元のプロジェクト設定・権限は変更していない。
- 最終の対象検証（固定venv、`PYTHONPATH=src;.;tests/refactoring`、OMP/MKL各1）: `test_feddrift_*.py`（新specの29件を含む）・`test_run_settings_validation.py`・`test_experiment_sweep.py`・`test_fedavg_experiment_sweep.py`・`test_single_run_dependency_boundaries.py`は5472 passed / 113 skipped（51.57秒、`results/_feddrift_target_final.log`）。共有部の追加の検証は上記233 passed / 1 skipped（`test_model_cross_evaluation.py`・`test_cross_evaluation_download_deduplication.py`・`test_experiment_run_code_provenance.py`を含み、対象検証との重複あり）。
- 最終のgolden検証: `test_important_behavior_regression_goldens.py`は499 passed / 41 skipped / 5 failed（769.77秒、`results/_feddrift_golden_checks.log`）。版1〜13の全caseの数値照合が成功。失敗はすべて版14のファイル未配置によるもの（両環境の配置の確認1件と、現在版の実goldenへの変更の検出4件）。skipは一時ファイルを要する16件、版14の数値照合22件、既存の条件付き3件。Windows上での実行で、Linuxは保存済みのgoldenの構造・設定の照合のみ。WSLでの数値実行は行っていない。
- 版1〜13の数値照合は計260case。最後のサーバ組立の基準不一致の検査追加後、FedDrift・FedAvgの数値照合だけを取り直して52 passed / 4 skipped / 489 deselected（17.33秒）。skipは版14のファイル未配置の4case。数値処理の追加変更はない。
- 品質検査: Ruff check（`src tests/refactoring`）、変更したPythonのRuff format --check、変更sourceのPyright（元のpyprojectのstrict対象も維持し、固定venvの依存先を明示）、`spec_checks.py names ... --base HEAD`、`git diff --check`が成功。新しい公開名の未登録はなし。requirements.md・design.md・naming.md、版1〜13のgolden、保留中の3資料、options.mdは変更していない。全pytest、版14のgolden、WSLでの数値実行、独立実装レビュー、commitは未実施で、主担当へ引き継ぐ。
