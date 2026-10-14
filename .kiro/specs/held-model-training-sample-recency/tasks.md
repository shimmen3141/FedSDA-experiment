# Implementation Plan

- [ ] 1. 設定と抜き出し
  - [ ] 1.1 共有の検査（`validate_settings_field_values`）の`int | None`・`float | None`の扱い、`LocalTrainingSettings`の3つの項目と検査、`FedsdaRunClientSettings`のKとbatchの件数の検査。testを先に書く（既存の`LocalTrainingSettings`の生成——候補の学習・事前学習・test——が通ることを含む）。goldenのhelperの「後に足された項目」へ、版1〜5の既定の値を足す（設定の保存の形が変わるので、同じcommitで）。FedSDAのrunの識別子を期待するtestを合わせる。
    - _Requirements: 1.1, 1.4, 3.1_
  - [ ] 1.2 抜き出しの関数の選び方の引数（直近K件・指数減衰）と、共同更新の繰り返しからの受渡し。testを先に書く（既定は今と同じ、直近K件の母集団、指数減衰の重複なし・件数・偏り・条件付き確率・再現・切り替え、Kが長さ以上のとき。基準は設計のValidation Design）。依存境界: このmoduleの`import math`の禁止の例を外し（別の禁止の依存に置き換える）、許可の一覧へ足す。
    - _Requirements: 1.2, 1.3, 1.5, 3.2_

- [ ] 2. 掃引の軸
  - [ ] 2.1 計画の軸と文字列の変換、FedSDAの構成を含まない計画の拒否（run設定の型で）、変種の作り方。testを先に書く。掃引の記録の計画のキーと形式の版（4）の期待を合わせる。
    - _Requirements: 2.1, 2.3_
  - [ ] 2.2 コマンドの引数（不正な値で終了コード2）と、一覧の表の列（前の要約の行は空の欄で、表から外さない）。testを先に書く。
    - _Requirements: 2.1, 2.2_

- [ ] 3. 文書
  - 実験の入口の文書（軸、抜き出し方の意味、Kとαの目安）、研究バックログの7、再開案内。
  - _Requirements: 4.1_

- [ ] 4. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest（goldenの版1〜5を含む）・Ruff・Pyright・WSLの対象test。手元の当たり付けは、このspecの完了の後に行う（完了の条件に含めない）。
  - _Requirements: 全部_

## 仕様レビュー4回目（PASS）のMinorの扱い（実装で行う）

- 依存境界: 抜き出しのmoduleの許可へ`math.floor`・`math.log`・`bisect.insort`・`LocalTrainingSettings`（module名と名前）を足す。共有の禁止の例の一覧を分け、抜き出しのmoduleの分だけ`from math import ...`を許可する（`import math`のmodule形式は禁止のまま）。掃引の計画のmoduleは、`re`を使わず、文字列の操作とbuiltinの`int`・`float`で解析する。
- metadata: Kは下限1・包含、αは下限0・排他、上限1・排他（`minimum_value_is_inclusive`ほかのキーを全部書く）。
- 共同更新の入口（`joint_model_parameter_update.py`）の組み直しは、新しい3項目も渡して再検査する。抜き出しは、選び方を指定されたとき、Kとαの型と範囲も確かめる（乱数を進める前に、対象のモデルの有無に依らず）。
- 表の列の表示: `recent:<K>`、`decay:<repr(α)>`（入力の`0.9970`は`0.997`と表示）。文書に書式の例と拒否の形を書く。
- 「列の長さに依らない」は、抜き出しの関数の手数（呼出し元の標本の列のsnapshotは、今どおり列の長さに比例する）。
- task 1.2は、指数減衰を単独の下位task（1.3）として先に書いて確かめる。

