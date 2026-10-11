# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. 計画
- [x] 1.1 構成の名前と、掃引の計画、識別子
  - _Boundary: experiment_configurations、experiment_sweep_plan_
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. 実行
- [x] 2.1 逐次・並列の実行、再開、失敗の扱い
  - _Depends: 1.1_
  - _Boundary: experiment_sweep_execution_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3. 記録
- [x] 3.1 一覧の表と、実行の記録
  - _Depends: 2.1_
  - _Boundary: experiment_sweep_records_
  - _Requirements: 3.1, 3.2_

- [x] 4. コマンド
- [x] 4.1 コマンドと、直下のscript、使い方の文書
  - _Depends: 3.1_
  - _Boundary: experiment_sweep_command、run_experiment_sweep.py_
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5. 検証
- [x] 5.1 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果（並列の実行を含む）を、区別して記録してある。
  - _Depends: 4.1_

## Implementation Notes

### 実装（task 1.1〜4.1）

- 手順: testのファイルを先に書き、sourceを書いてから、まとめて実行した（sourceがない状態での実行は、していない）。最初の実行で、1件が失敗した（エラーの出力先を、moduleを読み込んだときの`sys.stderr`に束縛していた）。`ArgumentParser.error`で出す形へ直した。
- 新しいpackage `sweep/`を、依存境界のtestへ、moduleごとの許可集合と、packageの境界として登録した。直下のscriptを、Ruffの対象へ足した。
- 実際のコマンドでの確認（testにはしていない）: client 10・標本500・sine2・seed 0・集約間隔50を実行し、旧の`run_pareto_sweep.py`（提案構成の引数）の同じ条件の結果と、精度（0.9658）・定常精度・通信量（上り101、下り118）・モデル数（2）・検出数（2）が一致した。
- **速度（このspecの対象の外。次のspecで扱う）:** 上の条件で、新実装は約60秒、旧実装は約17秒（旧は、図の作成を含む。ほかの処理が動いている中での計測）。プロファイル（client 10・標本300）では、実行時間の約45%が、学習のたびに、保持している学習標本の全部を検査し直す処理（`learning/training/held_model_training_batch_sampling.py`の`_validate_sampling_request`）だった。保持する標本の数に比例するので、標本数が増えると、全体は、標本数の2乗に近づく。本番の規模（標本5000）の1 runは、手元で、開始から40分以上たっても終わらなかった。この検査は、完了済みのspecが、意図して決めた挙動（抽出に入らない標本も、事前に検査する）。

### 独立レビューと検証

- 独立レビュー（1回目、2026-10-11）: Claude Haiku 5.5（独立CLI、effort `medium`を指定）。GPT-6 Lunaは利用上限で使えない期間。session `07f94291-245b-457b-acf0-2436cd3c3baa`、対象`ee15630..9c50e92`。判定は`IMPLEMENTATION: CHANGES_REQUESTED`（Major 3件、Minor 5件、任意 1件）。レビュー担当は、このspecのtestと、依存境界、Ruffを独立に実行した。全pytest、Pyright、実際のコマンドの実行は、していない。レビュー担当の逸脱（本人の申告）: 読取りに、許可していないshellのコマンド（`cat`・`sed`・`grep`）を、数回使った（ファイルは変えていない）。
- 指摘の採否（1回目）: (1)Major: 並列のworkerで、OpenMP・MKLのthreadを、1に保っていない→採用（直下のscriptが、NumPyとtorchを読み込む前に、3つの環境変数を、未設定なら1にする。実行の記録の環境に、値を残す）。(2)Major: 途中の例外やCtrl-Cで、待ちのrunが止まらない→採用（例外のとき、待ちのrunを取り消してから、伝える）。(3)Major: 壊れた要約が1つあると、全runの後に、表と記録が書かれない→採用（読めない要約は、表から外して報告し、記録を書き、終了コード1）。(4)Minor: 要求2.2の文言→要求を直した。(5)Minor: 記録の名前の、同じ秒の衝突→番号をつける。(6)Minor: 保存の途中で止まると、`.npz`だけが残り、次の実行で失敗になる→挙動は変えず、設計と使い方へ書いた（自動で消して実行し直す形は、保存の仕様の変更になるので、していない。必要なら、ユーザーの判断で足す）。(7)Minor: 並列の、実際のコマンドを通すtestがない→足した。(8)構成の名前の案→設計の表を直した。(9)任意: 記録の環境→(1)で対応。testを先に直し、失敗（読込みのエラー）を確かめてから、sourceを直した。
- 修正の確認（別session、2026-10-11）: Claude Haiku 5.5。session `bac5df71-0a87-4f9f-89c0-dcd397360ccc`、対象`9c50e92..307ed51`。判定は`FIX CONFIRMATION: APPROVED`（Blocker・Majorなし。Minor 5件、任意 2件）。このspecのtest（51 passed）とRuffを独立に実行した。
- 確認の指摘の採否: (a)直下のscriptを通さずに、関数を呼んだ場合、workerのBLASのthreadが決まらない→採用（並列の実行は、workerのprocessを作る前に、環境変数を、未設定なら1にする）。(b)「まだ始まっていないrunを取り消す」は、厳密でない（workerへ渡したrunは、取り消せない）。Windowsでは、Ctrl-Cが、workerにも届いて、実行中のrunも止まりうる→コメント・要求・設計・使い方の文を、実態に合わせた（Ctrl-Cの挙動は、実測していない）。(c)実行条件の値の型が違う要約が、表の並べ替えを壊す→採用（型を確かめて、表から外す）。(d)記録の`torch_thread_count`は、コマンドのprocessの値→コメントを直した。(e)testの`__import__("os")`→直した。testを先に直し、2件の失敗を確かめてから、sourceを直した。
- WSLで成功（`307ed51`、WSL Ubuntu）: このspecのtestが、51 passed・skipなし（`spawn`のprocessでの並列と、直下のscriptでの並列を含む）。1回目（`9c50e92`）は、2件が失敗した——WSLのgitが、Windowsのworktreeを読めず、コードの版が取れない。testが、取れない場合を許していなかった——。testを直した（Windowsの基準環境では、取れることを求める）。確認の指摘の反映の後は、WSLでは実行していない。
- 検証（Windows基準環境）: `9c50e92`で、全pytest（並列、`-n 8 --dist loadfile`）12448 passed / 4 skipped / 3 warnings。`307ed51`で、12451 passed / 4 skipped / 4 warnings。確認の指摘の反映の後の`06dc386`で、12451 passed / 4 skipped / 3 warnings、exit 0。warningは、既知のtorchの警告（実行によって、3件か4件）。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。skipの4件は、POSIX bashがない3件と、旧実装の照合を既存の回帰testに任せる1件。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、新規・変更のsourceで「未登録: なし」。この後は、specの文書とsteeringだけを変えた。
- 変異テストは実行していない（既定）。

### 未検証・残る制約

- 構成は、提案構成（`fedsda`）だけ。構成の名前の案（設計の表）は、ユーザーの確認を、まだ受けていない。
- 本番の規模（client 10・標本5000）での実行は、完了を確かめていない（速度の問題。上の「速度」）。提案構成を、ablationの規模で、新旧で比べる確認は、速度のspecの後。
- Ctrl-Cでの中断の挙動（Windows・Linux）は、実測していない（待ちのrunの取り消しは、testで確かめた）。
- 研究室サーバ（Linux、Python 3.10）での実行は、していない。
- 別の出力先との、runの重複は、調べない。図は、作らない。
