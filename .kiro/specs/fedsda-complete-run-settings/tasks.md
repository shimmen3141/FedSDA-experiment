# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. 設定の型と変換
- [x] 1.1 保存用の辞書への変換
  - 変換の規則、決定性、拒否のtestを先に書く。
  - 完了: 変換のtestと、依存境界のsuiteが成功する。
  - _Boundary: settings_serialization_
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 1.2 完全なrun設定
  - 有効な値の受理、型・組合せ・食い違いの拒否、`replace`の後の検証、全体runへ渡せることのtestを先に書く。
  - 完了: 完全なrun設定のtestと、依存境界のsuiteが成功する。
  - _Boundary: FedsdaRunSettings_
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. 最終構成の既定値
- [x] 2.1 最終構成の設定の関数
  - 旧の設定の値との照合（sine2・mnist2）、全datasetのモデルの既定、goldenの3ケースの条件との一致、保存用の辞書のtestを先に書く。
  - 完了: 最終構成の設定のtestと、依存境界のsuiteが成功する。
  - _Depends: 1.1, 1.2_
  - _Boundary: fedsda_final_configuration_run_settings_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1_

- [x] 3. 検証
- [x] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.1_

## Implementation Notes

### 実装（task 1.1〜2.1）

- 手順: 3つのtestのfileを、対応するsourceより先に書いた。ただし、変換（1.1）は、testとsourceを続けて書いてから、初めて実行した（失敗を確かめていない）。完全なrun設定（1.2）と既定値（2.1）は、sourceがない状態での収集の失敗の後、sourceを書いた。testが最初から通ったので、既定値の定数を2つ、一時的に変えて（評価標本の追加件数20→21、回復の窓200→50）、testが失敗することを確かめ、元へ戻した。
- 完全なrun設定`FedsdaRunSettings`は、既存の設定型（実行の枠、参加者の束、統合、指標）を、そのまま束ねる。機能の組合せは、既存の部分型`ValidatedExperimentRunSettingsSubset`を、部分から組み立てて検証する。全体runの実行は、変えていない。既存の設定型は、レビューの指摘への対応（下）で、clientの設定の束へ検査を1つ足した。
- 依存境界: `runtime/fedsda_run_settings.py`と`runtime/fedsda_final_configuration_run_settings.py`は、moduleごとの許可集合へ登録した。`core/settings_serialization.py`は、標準ライブラリだけを使う（coreの規則）。

### 照合の結果（Windowsの基準環境）

- 既定値: 定義のある6つのdatasetすべてで、最終構成の既定値の完全なrun設定が、旧の設定（初期値に、最終構成の固定設定を重ねて有効化した値）を、既存のtestの対応（goldenを再現する設定を作る`make_golden_condition_settings`）で写した設定と、`==`で一致した。既存の対応が、旧の設定から読まずに、決めた値で渡している項目のうち、数値と種類（検出器の賭け率、更新の反復回数、Adamの種類）は、旧の値と、個別に照合した。方式の選択（予測の結合、再較正、学習、勾配の統合、候補の方針・学習・初期化、クラスタリングの4つ、概念列）は、旧の選択肢の値と、新の方式の名前の対応の表（testの`LEGACY_AND_NEW_FINAL_CONFIGURATION_CHOICES`）で、両方の値を確かめる（名前の対応そのものは、表の記載であり、挙動の一致は、既存の全体runの対照と、goldenの照合が確かめている）。
- datasetで変わる既定は、隠れ層の幅と学習率だけ（旧のdatasetの定義と一致）。
- goldenの3ケースの条件: 旧の回帰testの条件が、既定値から変えている旧の設定の項目は、client数・標本数・事前学習の件数（mnist2はepochも）・概念の変更の最小の間隔と確率・回復の窓だけだった。既定値から、この項目だけを置き換えた設定が、既存の（goldenを再現する）設定と、`==`で一致した。
- 完全なrun設定の部分を、そのまま、全体runの実行と、指標の導出へ渡せることを、小さい条件（sea2、client 2、60件）で確かめた。

### 独立レビューと検証

- 独立レビュー（1回め、2026-10-11）: Claude Haiku 5.5（独立CLI、effort `medium`を指定）。GPT-6 Lunaは利用上限で使えない期間。session `a6ff612f-9c12-4594-a149-2ce598f1dcc1`、対象`344ff4f..a3b3756`。判定は`IMPLEMENTATION: CHANGES_REQUESTED`（Major 1件、Minor 2件、任意 1件）。レビュー担当は、対象のtest（44 passed）、依存境界suite（3043 passed）、Ruffを独立に実行し、既定値の数値を、旧の`config.py`・`data/specs.py`・`models.py`・proposed-method.mdと、独立に照合した（一致）。同じ値を2箇所で使う定数（γ、FIFOの長さ、batch、最小改善量）が、旧でも1つの設定であることも、旧のsourceで確かめた。
- 指摘の採否: (1)Major: 検出の方式（`e_sr`）と、記録に残す検出器の表示名（`detector_name`。任意の文字列を受け付けていた）の食い違いを、拒否していない。要求1.3の例（アダプタのrankほか）は、起こりえない食い違いだった→採用。testを先に足して失敗を確かめ、clientの設定の束`FedsdaRunClientSettings`の生成時の検査へ、検出の方式から決まる表示名との一致を足した（既存の、完了済みの設定型への、検査の追加）。要求1.3と設計の記載を、実際に一致を求める組へ書き直した。再利用の許容量とクラスタリングの判定の上限の一致は、既存の参加者の設定の束が、すでに確かめていた（testを足した）。(2)Minor: 方式の選択の既定値を、旧の選択肢の値と、直接照合していない→採用。旧の選択肢の値と、新の方式の名前の対応の表を、testへ足した。(3)Minor（潜在）: 保存用の辞書の型名が`__name__`だけ。「違う辞書」の比較が`==`→型名は変えず（moduleを含めると、moduleを動かしたときに、保存した条件と一致しなくなる）、完全なrun設定の中の設定型の名前が重ならないことを確かめるtestを足した。比較を、JSONの文字列で行う形へ直した。(4)任意: 検出器の候補の上限の出どころを、コメントに書く→採用。修正は`83a6271`。
- 確認のレビュー（Majorを直したので、別session、2026-10-11）: Claude Haiku 5.5。session `7e249cd0-eece-4a50-9fa5-44937c68e2c6`、対象`a3b3756..83a6271`。判定は`FIX: APPROVED`（Blocker・Majorなし。Minor 1件、任意 2件、確認のみ 2件）。採否: 対応の表の、アダプタのrankの行が、rankの値を確かめていない→採用（`929acd0`。testだけ）。既定値の比較も、JSONの文字列でも行う→採用（同）。検出の方式をfrozenを回避して不正にしたとき、理由が`detector_name`になる→変更なし（拒否はされる）。確認担当は、許可した検証コマンドのほかに、読取りだけのコマンド（`git grep`・`grep`・`sed`・`cat`・`ls`）を実行したと申告した（依頼文の範囲からの逸脱。ファイルは変更・作成していない）。
- 検証（Windows基準環境、`929acd0`。この後は、specの文書とsteeringだけを変えた）: 全pytest（並列、`-n 8 --dist loadfile`）11900 passed / 4 skipped / 3 warnings、exit 0。JUnitはfailure 0、error 0。`tests.test_regression`・`tests.test_proposed_regression`を含む。skipの4件は、POSIX bashがない3件と、旧実装の照合を既存の回帰testに任せる1件。1回めのレビューの前の`a3b3756`では 11898 passed、修正の`83a6271`では 11900 passed。Ruff（check・format）、Pyright 0 errors、`pip check`成功。固定旧`748c3aa`からの差分は空。`spec_checks.py names`は、新規のsourceで「未登録: なし」。
- WSLでは、このspecのtestを実行していない（sourceは、設定の値の組立てと検証だけで、数値計算を含まない）。
- 変異テストは実行していない（既定。既定値の定数を2つ、手で変えて、testが検出することは確かめた）。

### 未検証・残る制約

- 旧の既定の規模（client 10、標本5000）での全体runは、実行していない（既定値の照合は、設定の値の一致で行った）。
- 辞書から設定へ戻す変換は、ない。
- 既定値の一部を変える書き方は、`dataclasses.replace`の入れ子で、長い（掃引の軸と、短い書き方は、(A3)）。
- 設定の束の中の、仮の置き場所（スカラーの束ほか）は、そのまま。
