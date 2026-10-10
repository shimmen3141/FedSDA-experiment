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

- [ ] 3. 検証
- [ ] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.1_

## Implementation Notes

### 実装（task 1.1〜2.1）

- 手順: 3つのtestのfileを、対応するsourceより先に書いた。ただし、変換（1.1）は、testとsourceを続けて書いてから、初めて実行した（失敗を確かめていない）。完全なrun設定（1.2）と既定値（2.1）は、sourceがない状態での収集の失敗の後、sourceを書いた。testが最初から通ったので、既定値の定数を2つ、一時的に変えて（評価標本の追加件数20→21、回復の窓200→50）、testが失敗することを確かめ、元へ戻した。
- 完全なrun設定`FedsdaRunSettings`は、既存の設定型（実行の枠、参加者の束、統合、指標）を、そのまま束ねる。機能の組合せは、既存の部分型`ValidatedExperimentRunSettingsSubset`を、部分から組み立てて検証する。既存の設定型と、全体runの実行は、変えていない。
- 依存境界: `runtime/fedsda_run_settings.py`と`runtime/fedsda_final_configuration_run_settings.py`は、moduleごとの許可集合へ登録した。`core/settings_serialization.py`は、標準ライブラリだけを使う（coreの規則）。

### 照合の結果（Windowsの基準環境）

- 既定値: 定義のある6つのdatasetすべてで、最終構成の既定値の完全なrun設定が、旧の設定（初期値に、最終構成の固定設定を重ねて有効化した値）を、既存のtestの対応（goldenを再現する設定を作る`make_golden_condition_settings`）で写した設定と、`==`で一致した。既存の対応が、旧の設定から読まずに、決めた値で渡している項目（検出器の賭け率、更新の反復回数、Adamの種類、候補の学習と初期化の方式、概念列の方式）は、旧の値と、個別に照合した。
- datasetで変わる既定は、隠れ層の幅と学習率だけ（旧のdatasetの定義と一致）。
- goldenの3ケースの条件: 旧の回帰testの条件が、既定値から変えている旧の設定の項目は、client数・標本数・事前学習の件数（mnist2はepochも）・概念の変更の最小の間隔と確率・回復の窓だけだった。既定値から、この項目だけを置き換えた設定が、既存の（goldenを再現する）設定と、`==`で一致した。
- 完全なrun設定の部分を、そのまま、全体runの実行と、指標の導出へ渡せることを、小さい条件（sea2、client 2、60件）で確かめた。

### 未検証・残る制約

- 旧の既定の規模（client 10、標本5000）での全体runは、実行していない（既定値の照合は、設定の値の一致で行った）。
- 辞書から設定へ戻す変換は、ない。
- 既定値の一部を変える書き方は、`dataclasses.replace`の入れ子で、長い（掃引の軸と、短い書き方は、(A3)）。
- 設定の束の中の、仮の置き場所（スカラーの束ほか）は、そのまま。
