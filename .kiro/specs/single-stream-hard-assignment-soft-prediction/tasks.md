# Implementation Plan

- [ ] 1. パッケージの土台、設定、概念モデル
  - `src/single_stream_hard_assignment_soft_prediction/`と`requirements-single-stream.txt`を作る。設定の型と既定値の組立て（監視・Fixed-Shareは製品の設定の型を持つ）と、Hoeffding treeの概念モデル（平滑化した確率、監視用・MDL用の損失、1件の学習、写し、空の新しいモデル、節点の数）を実装する。
  - 完了条件: 設定のtest（既定の値が製品のFedHASPの既定と同じ、範囲外の拒否）がWindowsの固定venvで通る。概念モデルのtest（空の木は一様、平滑化で最大のクラスが変わらない、和が1、損失の式、写しが独立、新しいモデルが空）がWSLのRiver環境で通る。
  - _Boundary: single_stream_hard_assignment_soft_prediction（設定、概念モデル）_
  - _Requirements: 1.1, 1.5, 2.1, 2.2, 2.3, 7.3_

- [ ] 2. 警報の時点のMDLによる選択
  - `select_alarm_interval_assignment`を実装する（保有モデルのMDL用の損失、resetとkeepの学習者の逆順の前置きの和、製品の比較の関数）。
  - 完了条件: 手で作った列で、そのまま／保有モデルへ／新しいモデルへの3つの選択、保有モデルの状態が変わらないこと、前置きの和が逆順の1件ずつの予測→学習で作られることのtestが通る（WSL）。
  - _Boundary: single_stream_hard_assignment_soft_prediction（MDL）_
  - _Depends: 1_
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 3. 1標本の処理
  - `SingleStreamHardAssignmentSoftPredictionProcessor`を、design.mdの「1標本の処理の順」で実装する（予測、重みの更新、ウォームアップと監視、保留、警報の判定、割り当てと学習、監視の始め直し、記録）。
  - 完了条件: 手で作ったA→B→Aの系列で、ウォームアップ中は警報が出ない、容量を超えた最古の標本だけが学習される、各標本が1回だけ学習される、警報の後に保留が空になり監視が始め直される、再来で保有モデルが再利用される、新しいモデルが選ばれた学習者の種類で作られる、min(n,k)<5で警報としない、真のコンセプトのIDを替えても予測・学習が変わらない、のtestが通る（WSL）。
  - _Boundary: single_stream_hard_assignment_soft_prediction（1標本の処理）_
  - _Depends: 2_
  - _Requirements: 1.2, 1.3, 1.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 5.4, 5.5, 5.6, 5.7_

- [ ] 4. run、成果物、実行コマンド
  - `run_single_stream`・指標・経過時間、JSON・NPZの保存と読込み、系列のファイルを並列に処理する実行コマンドを実装する。
  - 完了条件: 指標が手で計算した値と一致、保存→読込みの往復、上書きの拒否、不正な系列の拒否のtestが通る（WSL）。書き出した系列（SINE2・SEA4・MNIST4、5,000標本、seed 0〜2）で実行コマンドが動き、成果物と一覧のCSVができる。
  - _Boundary: single_stream_hard_assignment_soft_prediction（run、成果物、コマンド）_
  - _Depends: 3_
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 5. 製品との境界と全体の検証
  - 境界のtest（新しいパッケージは許可した製品のmoduleとtorch以外の製品のmoduleをimportしない、製品は新しいパッケージをimportしない、WSLでimportしてもtorchが読み込まれない）を足す。naming.mdを実装に合わせる。独立CLIの実装レビューの指摘を反映した後、Windowsの全pytest（riverを使うtestは飛ぶことを確かめる）と、WSLの新しいtestの全件、Ruffを通す。
  - 完了条件: 全pytest（Windows）が通り、既存の件数＋新しいtestの数と一致し、goldenと既存の識別子が変わらない。WSLで新しいtestが全件通る。件数とcommitをImplementation Notesへ記録する。
  - _Depends: 4_
  - _Requirements: 7.1, 7.2_

## Implementation Notes
