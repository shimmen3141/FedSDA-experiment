# Implementation Plan

- [x] 1. 設定
  - `src/single_stream_hard_assignment_soft_prediction/`（`__init__.py`）と`requirements-single-stream.txt`を作り、設定の型と既定値の組立てを実装する（design.mdの設定の一覧）。riverとtorchをimportしない。
  - 完了条件: 設定のtest（「製品と同じ」の行の値が製品の既定と同じ、範囲外の拒否）がWindowsの固定venvで通る。
  - _Boundary: single_stream_hard_assignment_soft_prediction（設定）_
  - _Requirements: 1.5, 7.3_

- [x] 2. 概念モデル
  - `HoeffdingTreeConceptModel`（平滑化した確率、監視用・MDL用の損失、1件の学習、写し、空の新しいモデル、節点の数）と、特徴の変換を実装する。
  - 完了条件: 概念モデルのtest（design.mdの「概念モデル」の項目）がWSLのRiver環境で通る。
  - _Boundary: single_stream_hard_assignment_soft_prediction（概念モデル）_
  - _Depends: 1_
  - _Requirements: 1.1, 1.6, 2.1, 2.2, 2.3_

- [x] 3. 警報の時点のMDLによる選択
  - `select_alarm_interval_assignment`を実装する（保有モデルのMDL用の損失、resetとkeepの学習者の逆順の前置きの和、製品の比較の関数）と、選ばれた学習者を作り直す関数。
  - 完了条件: design.mdの「MDL」の(a)〜(f)のtestが通る（WSL）。
  - _Boundary: single_stream_hard_assignment_soft_prediction（MDL）_
  - _Depends: 2_
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 4. 1標本の処理（警報の無い経路）
  - `SingleStreamHardAssignmentSoftPredictionProcessor.process_sample`の、予測、重みの更新、ウォームアップと監視、保留、警報でないときの割り当てと学習、記録を実装する。
  - 完了条件: ウォームアップ中は監視も警報も無い、容量を超えた最古の標本だけが学習される、各標本が1回だけ学習される、真のコンセプトのIDを替えても予測・学習が変わらない、予測の混合と重みの更新が製品の部品の呼び方どおり、のtestが通る（WSL）。
  - _Boundary: single_stream_hard_assignment_soft_prediction（1標本の処理）_
  - _Depends: 2_
  - _Requirements: 1.2, 1.3, 1.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4_

- [x] 5. 1標本の処理（警報の経路）
  - 警報の判定（min(n,k)<5で待つ）、要求5.5の(a)〜(d)、監視の始め直し、警報の記録を実装する。
  - 完了条件: 手で作ったA→B→Aの系列で、警報の後に保留が空になり監視が始め直される、再来で保有モデルが再利用される、新しいモデルが測った学習者と同じ作り方で作られる、n<5で警報とせず次の標本で合図が続けば選択をやり直す、Fixed-Shareの重みが再利用の警報では保たれ新しいモデルの警報では一様に戻る、各標本が1回だけ学習される、のtestが通る（WSL）。
  - _Boundary: single_stream_hard_assignment_soft_prediction（1標本の処理）_
  - _Depends: 3, 4_
  - _Requirements: 5.4, 5.5, 5.6_

- [ ] 6. runと指標
  - `run_single_stream`（系列の検査、全標本の処理、経過時間）と`compute_prequential_metrics`（精度、Kappa系、変化点の後の精度、概念の同定、資源）を実装する。
  - 完了条件: 指標が手で計算した値と一致（端の場合を含む）、概念の同定の分類が期待どおり、不正な系列・hashの不一致・K未満でないラベルの拒否のtestが通る（WSL）。
  - _Boundary: single_stream_hard_assignment_soft_prediction（run・指標）_
  - _Depends: 5_
  - _Requirements: 1.7, 6.1, 6.2, 6.3_

- [ ] 7. 成果物と実行コマンド
  - JSON・NPZの保存と読込みと、系列のファイルを並列に処理して成果物と一覧のCSVを書く実行コマンドを実装する。
  - 完了条件: 保存→読込みの往復、上書きの拒否のtestが通る（WSL）。書き出した系列（SINE2・SEA4・MNIST4、5,000標本、seed 0〜2）で実行コマンドが動き、成果物と一覧のCSVができる（WSL）。
  - _Boundary: single_stream_hard_assignment_soft_prediction（成果物・コマンド）_
  - _Depends: 6_
  - _Requirements: 6.4, 6.5_

- [ ] 8. 製品との境界と全体の検証
  - 境界のtest（静的なimportの検査、製品が新しいパッケージをimportしない、WSLでtorchが読み込まれない）を足す。naming.mdを実装に合わせる。独立CLIの実装レビューの指摘を反映した後、Windowsの全pytestとWSLの新しいtestの全件、Ruffを通す。
  - 完了条件: Windowsの全pytestが通り、件数が「前の件数＋新しいtestのうちWindowsで動く数」と、skipの数が「前のskip＋riverを使う新しいtestの数」と一致する。goldenと既存の識別子が変わらない。WSLで新しいtestが全件通る（設定の一致のtestはtorchが無いので飛ぶ）。件数とcommitをImplementation Notesへ記録する。
  - _Depends: 7_
  - _Requirements: 7.1, 7.2_

## Implementation Notes

### 仕様レビュー
- GPT-6 Lunaは使用量の上限で使えず、Claude Haiku 5.5（effort medium、読むだけ、独立CLI）へ代えた。
- 1回目（Haiku 5.5、session `b74feb35-4a62-4442-bb9a-ad91d4d07eb8`、JSONの`is_error=false`・`modelUsage`は`claude-haiku-5-5`、対象 `6d0ce8d1`）: FAIL（Major 5、Minorとした規約の指摘1、Minor 9、任意 1）。
  - Major 1（既存の比較手法`river_baselines.py`は事前学習の標本を学習してから評価していて、条件が違う）: 採用。比較の基準を「事前学習なし」とし、比較手法の事前学習なしの版と再実行をこのspecの範囲外として要求のBoundary Contextに明記した。
  - Major 2（新しいモデルの写しの時点と学習の順が、MDLで測った学習者と一致しない）: 採用。新しいモデルを、警報の時点の空の木か現在のモデルの写しに、後のk件を新しいものから順に学習させて作る（測った学習者のk件目の状態を再現する）ことにし、前のn−k件の割り当てより先に作る順にした（要求5.5、design「製品と違う点3」）。
  - Major 3（移動窓の精度、概念の同定、資源の指標、Kappa-Mの端の場合が無い）: 採用。変化点の後の30・100・500件の精度、概念の同定の分類、資源（モデル数、節点の数、経過時間）を要求6.2・6.3とdesignの指標に足し、Kappa-Mの端の場合を定義した。移動窓の精度は保存した標本ごとの正誤から集計の側で求める（要求6.5）。
  - Major 4（クラス数Kの決め方が無い）: 採用。系列のJSONの`class_count`か、製品のデータセットの定義から取り、ラベルから推定しない（要求1.7）。
  - Major 5（製品と違う点の網羅と、設定の一致のtestの対象）: 採用。予測を局所の版だけにすることを「製品と違う点1」に足し、根拠の記録の場所を書いた。設定の全fieldの一覧を作り、「製品と同じ」の行だけを要求7.3のtestの対象にした。監視の設定（基準を下げる方針、賭けの割合、候補の上限）を一覧に入れ、基準の引き下げは監視の部品の既定の挙動に従うと書いた。
  - 規約の指摘 6（下位taskに別の境界が混ざる、大きい）: 採用。task 1を設定と概念モデルに、task 3を警報の無い経路と警報の経路に、task 4をrun・指標と成果物・コマンドに分けた（8 task）。
  - Minor 7〜11（Fixed-Shareの更新に予測に使った重みを渡す、監視に毎回基準値を渡す、警報の後の基準値は`select_loss_monitoring_baseline_mean_loss`、そのままのときk=n、警報でFixed-Shareの重みを保つ）: 採用。designと要求3・4・5に書いた。
  - Minor 12（naming.mdに`process_sample`が無い、製品と同名の関数の違い）: 採用。
  - Minor 13（ウォームアップ中の対数e値、「最後の時点の」、適応の結果の意味）: 採用。NaN、最後の時点、`stay`・`reuse`・`new`・`too_short`と定義した。
  - Minor 14（Windowsでriverを使うtestが飛ぶ数え方）: 採用。task 8の完了条件に、件数とskipの数の照合を書いた。
  - Minor 15（特徴の変換）: 採用。`river_baselines.py`と同じ変換（要求1.6）にし、testで一致を確かめる。
  - 任意 16（200＝grace periodの確認、L_swのMの定義）: 採用。WSLのriver 0.27.0で既定のgrace periodが200であることを確かめ、L_swのMが製品と同じ定義で論文の|M|と違うことをdesignに書いた。
- 2回目（同じHaiku 5.5、session `49f54e75-4951-4ef1-afb8-b7d3f09d45a2`、対象 `6d0ce8d1..e53ae3b9`）: FAIL（Major 2、Minor 8）。1回目のMajor 1〜5・規約6・Minor 7〜10・12〜16は解消と確認された。
  - Major 1（Fixed-Shareの重みは、保有モデルの集合が変わると部品の中で一様に戻るので、「警報の前後で保つ」は新しいモデルの警報で成り立たない）: 採用。要求3.3・designの対応表・testを、再利用の警報では保ち、新しいモデルの警報では一様に戻る、に直した。
  - Major 2（n<5で警報としなかった後の扱いが未定義）: 採用。検出器を始め直さず、次の標本で合図が続けばその標本を最後に含むWで選択をやり直す（製品の待つ方針と同じ）と、要求5.4とdesignに書いた。
  - Minor 3（min(n,k)<5は実質n<5）、4（測った学習者のk件目の状態の定義）、5（概念の同定の「k件」・最後の時点の概念・概念なしの扱い）、6（変化点の後の精度のまとめ方）、10（基準値に渡す統計は`overall_loss_moments`）: 採用。designに書いた。
  - Minor 7（class_countは系列のJSONに無い）: 採用。製品のデータセットの定義から取る一本にした（要求1.7）。
  - Minor 8（要求1の見出し）: 採用。「既存手法に合わせる条件」にした（比較手法の再実行は範囲外のまま）。
  - Minor 9（論文のL_swのMの不一致）: 採用。designに書き、論文側の修正を再開案内の次の候補に残す。
- 3回目（同じHaiku 5.5、session `4339927f-e5e0-4537-8938-254c12fdaa61`、対象 `e53ae3b9..e6330b97`）: FAIL（Major 1、Minor 3）。2回目のMajor 1・2とMinor 3〜8・10は解消と確認された。レビュー担当は、残るMajorを文書だけの修正で足り再レビューは不要と判定した。
  - Major 1（L_swのMについて「製品と論文で定義が違う」は誤り。製品の「帰属先の候補の数＋1」は、保有モデルがすべて統計を持つので保有モデルの数と等しく、論文の式M=|M|と同じ。ずれているのは論文4章の実装注記の「+1」の方）: 採用。designの対応表と設定の一覧を直した。論文の注記の修正は再開案内の次の候補へ残す。文書だけの修正なので、規約（agent-handoffの「承認と独立レビュー」6）に従い再確認を求めない。
  - Minor 2（ウォームアップ中の監視と対数e値）、3（概念の同定の`new`の対象）、4（「写し」の語）: 採用。

