# Implementation Plan

- [x] 1. 概念列
  - `random_concept_schedule.generate_random_concept_schedule`を実装し、境界のtestのriverを使わないmoduleの一覧へ足す。
  - 完了条件: 概念列のtest（製品の`generate_random_client_concept_traces`との一致と乱数の状態の一致、既定値が製品の`fedsda`の既定と同じ、引数の拒否、渡した乱数以外を変えない）と境界のtest（`RIVER_FREE_PACKAGE_MODULES`に概念列のmoduleを足し、別processの検査にriverの不在を加えたもの）が、Windowsの固定venvで通る。
  - _Boundary: random_concept_schedule_
  - _Requirements: 1.1, 1.2, 1.3, 6.2, 6.3_

- [x] 2. 読込みのクラス数
  - `load_single_stream_input`がJSONの`class_count`を使い、製品の定義との不一致・不正な値を拒否するようにする。
  - 完了条件: 読込みのtest（採用、不一致・1・bool・文字列の拒否、`class_count`の無い既存の形式は従来どおり）がWindowsで通り、既存の読込みのtestも通る。
  - _Boundary: single_stream_input_
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 3. データの定義と系列の生成
  - `river_synthetic_recurring_streams`（定義と`generate_river_synthetic_stream`）を実装する。
  - 完了条件: design.mdの「系列」のtestがWSLの`~/river-env`で通る。
  - _Boundary: river_synthetic_recurring_streams_
  - _Depends: 1_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 4. 書き出しのコマンド
  - `river_synthetic_stream_command.main`を実装する。
  - 完了条件: design.mdの「書き出し」のtest（`load_single_stream_input`での読込み、飛ばす・止める、知らないデータの拒否、木の版の`run_single_stream`で1本を処理）がWSLで通る。
  - _Boundary: river_synthetic_stream_command_
  - _Depends: 2, 3_
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 5. 比較手法の研究ツールの検査
  - `river_baselines.py`に、事前学習の標本の無い系列を事前学習ありで渡したときに処理の前に止まる検査を足す。
  - 完了条件: 研究ツールのtest（`test_river_synthetic_stream_command.py`。止まり、出力先を作らない。`--without-pretraining`では処理できる）がWSLで通る。
  - _Boundary: river_baselines（研究ツール）_
  - _Depends: 4（testの系列を書き出しのコマンドで作る）_
  - _Requirements: 5.1, 5.2_

- [x] 6. 全体の検証
  - 製品のファイルに差分が無いことを確かめ、Ruff（check・format）、Windowsの全pytest、WSLの単一ストリームのtestを通す。
  - 完了条件: 全pytestの失敗0、件数が前のspecの件数＋今回のtest数と一致、WSLの単一ストリームのtestが通る、`git diff`に`src/federated_learning_experiments/`が無い。
  - _Boundary: 全体_
  - _Depends: 1, 2, 3, 4, 5_
  - _Requirements: 6.1_

## Implementation Notes

### 仕様レビュー
- 1回目（Haiku 5.5、effort medium、session `e23d1a72-76ab-4839-9765-d02e03a6e838`、対象`48b1d8a3`）: FAIL（Major 1、Minor 8）。全部採用した。Major: 要求6.3（概念列のmoduleがriverを読まない）を確かめるtestが無い→境界のtestに概念列のmoduleのriver・torchの検査を足す設計にした。Minor: Random RBFの特徴の鍵が整数（`str()`して比べる）、比較手法の研究ツールの検査を出力先の作成より前に置く、書き出しの一時ファイルの名前と片方だけの系列の一括検査、直前の変化の初期値0、既定値は公開の組立ての関数から読む、要求1.3・2.3・3.1のtest、task 5の依存の理由とtestのファイル、moduleの名前を`random_concept_schedule`へ（関数の名前と揃える）。
- 2回目（同、session `fb2a287c-0b86-4ab7-a9f7-edea80787325`、対象`4e0125a0`）: FAIL（Major 1、Minor 2、任意2）。Major: riverの検査の書き方が二通りでtestが特定されていない→既存の別processの検査を広げる形（`RIVER_FREE_PACKAGE_MODULES`へ足し、riverの不在を加える）に決めて、design.md・tasks.mdを揃えた。Minor: 製品の`random_concept_schedule_generation`との違いを命名表へ書いた。`--seeds`の書式を木の版のコマンドと同じ整数の列に決めた。任意: Agrawalの摂動の中身は主担当がWSLのRiver 0.27.0のsourceで確かめた（research.md）。
- 3回目（同、session `880c934b-d022-4e02-b6d7-f061b8f667d1`、対象`dd909d3c`）: PASS（Minor 4）。全部採用し、文書だけを直した（規約により再確認不要）: 概念列のmoduleのASTのimportの検査、`--datasets`・`--seeds`は1つ以上、研究ツールのtestで出力先が作られないこと、spec.jsonへのレビューの記録。

### 実装（主担当）
- task 1〜5を`5104b098`・`ffe83a4e`で実装した。概念列のtestは製品のrun設定（torch）を読むので、torchの無いWSLではskipにした（Windowsで実行する）。

### 実装レビュー
- 1回目（Haiku 5.5、effort medium、session `f0102cb7-607b-42f4-b9f4-5d1d08429148`、対象`4a210b24..ffe83a4e`、Windowsとwslのtestを独立に実行）: PASS（Minor 5、任意1）。採用（`5641c2d7`）: testの名前を確かめている内容に合わせた、Agrawalの摂動のtestで数値の特徴の全体を比べる、JSONの概念列の設定を関数の既定から読む。不採用: import時のスレッド数の環境変数の設定（既存の`single_stream_command`と同じ書き方で、実行入口のmoduleなので揃える）、`random.Random`の型の厳しい検査（要求1.3で、渡すのは`random.Random`と決めた。製品は型を検査しないが、写しは入口なので検査する）、型注釈（任意。非公開のhelper）。修正はMinorだけなので再確認はしていない。

### 全体の検証（主担当）
- Windowsの全pytest（固定venv、OMP/MKL 1 thread、12 worker、`FDE_FASHION_MNIST_DATA_DIR`を設定）: 13,967 passed、16 skipped、失敗0。対象commit `5641c2d7`。JUnit: `venv/refactoring-tests/single-stream-recurring-synthetic-streams-full.xml`。
- 件数の照合（前のspec `single-stream-hard-assignment-soft-prediction`のJUnitと比べた）: 13,983−13,947＝36件を足した（Windowsで動く34件と、riverが要る2ファイルのskip 2件）。skipは14→16。
- WSLの`~/river-env`（Python 3.12、river 0.27.0、torchなし）: `tests/single_stream_hard_assignment_soft_prediction`は99 passed、3 skipped（torchが要るもの）。
- Ruff（check・format）: 0件。`spec_checks.py names`: 未登録なし。製品`src/federated_learning_experiments/`の差分なし。
- 未実施: pyright（riverの型をWindowsの環境で読めない）。

