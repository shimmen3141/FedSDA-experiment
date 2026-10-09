# 実装タスク revision4

逐次実行。レビューは、全taskの実装と全回帰の後に1回で受ける（共通引継ぎ手順）。

- [ ] 1. 標本1件の処理を、実旧対照つきで実装する
  - 1.1 保留標本のownerと警報の記録のownerを実装する
    - 2つのownerの単独のtestを先に追加し、sourceを実装して、依存の許可集合を登録する。
    - 完了: 2つのownerのtestと依存境界suiteが成功する。変異toolで、ownerのメソッドの検査と更新が検出される（未検出は、testで補うか、等価と判断した理由を証拠文書へ書く）。
    - _Boundary: PendingSampleObservationStore、LossChangeAlarmRecordStore_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3_
  - 1.2 標本1件の処理の、入力の検査と、警報のない標本の経路を実装する
    - 対照testの土台（最初の警報の後から、実旧の標本処理と標本ごとに照合する）と、拒否のtest、警報のない標本の順序のtestを先に追加し、入力の検査→候補検証の進行→損失の監視→保留への追加→帰属の確定→保留標本を合わせる→学習要求、を実装する。
    - 完了: 拒否のtest（候補検証の保持の有無の両方）、警報のない標本の順序のtest、途中の段の失敗で後の段へ進まないことのtestが成功する（実旧との対照testは、警報のある条件を含むので、1.3の完了で確かめる）。依存の許可集合を登録し、依存境界suiteが成功する。
    - _Depends: 1.1_
    - _Boundary: process_observed_sample（検査と、警報のない経路）、ObservedSampleProcessing_
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 4.1, 4.2, 4.3, 4.4_
  - 1.3 警報のある標本の経路と、候補検証へ渡した標本の概念IDの保持を実装する
    - 警報のある標本の順序のtestと、概念IDの対応づけの単独のtest（保留の古い側に同じ標本のオブジェクトがある場合、末尾と同一でない場合の拒否）を先に追加し、保留中の学習要求の学習→警報の位置の記録→警報1回ぶんの処理→概念IDの保持→保留標本を合わせる、を実装する。
    - 完了: 対照testが、全条件で、各標本の後の全状態の一致を示し、全結果を通ったことを確かめるtestがskipされずに成功する。警報のある標本の順序のtestと、概念IDの対応づけのtestが成功する。
    - _Depends: 1.2_
    - _Boundary: process_observed_sample（警報の経路）、概念IDの対応づけ_
    - _Requirements: 1.1, 1.4, 5.1, 5.2_
  - 1.4 標本1件の処理の検出力を確かめる
    - 関数がそろった時点で、変異toolを関数全体（検査の関数、概念IDの対応づけの関数を含む）へ実行する。
    - 完了: 未検出の変異がない、または、testで補うか等価と判断した理由を証拠文書へ書いてある。
    - _Depends: 1.3_
    - _Boundary: 検出力の証拠_
    - _Requirements: 1.4, 1.5, 4.1_
  - 1.5 共用のfresh process scriptへ、標本1件の処理を続けて呼ぶ流れを足す
    - 完了: 共用scriptが、既存の各流れの後に標本を続けて処理し、標本ごとに保留標本と保留位置の一致・概念IDの保持と候補検証の保持の対応を確かめ、適応記録の件数が警報と確定の回数に一致し、全体で警報が1回以上起きることを確かめて成功する。Ruff・Pyrightが成功する。
    - _Depends: 1.3_
    - _Boundary: fresh_process_smoke.py_
    - _Requirements: 1.4, 1.5_

- [ ] 2. 固定基準の全回帰と証拠を確定する
  - commitして、Windowsの基準環境で全pytestとJUnit、旧11・最終3golden、Ruff・Pyright・pip check、`spec_checks.py identity`と`progress`を実測し、検出力の証拠と、integration-validation.md（要求の対応と未検証事項）を記録する。全pytestの件数が、前specの件数に今回足したtest数を加えた数と一致することと、全結果を通ったことを確かめるtestがskipされずに成功したことを確かめる。
  - 完了: 実装していない別sessionが、コード・検出力の証拠・要求の対応・照合をまとめてレビューして承認する。再開案内を更新する。
  - _Depends: 1_
  - _Boundary: 全回帰と統合証拠_
  - _Requirements: 5.3_
