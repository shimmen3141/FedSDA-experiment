# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [ ] 1. 設定の型と変換
- [ ] 1.1 保存用の辞書への変換
  - 変換の規則、決定性、拒否のtestを先に書く。
  - 完了: 変換のtestと、依存境界のsuiteが成功する。
  - _Boundary: settings_serialization_
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 1.2 完全なrun設定
  - 有効な値の受理、型・組合せ・食い違いの拒否、`replace`の後の検証、全体runへ渡せることのtestを先に書く。
  - 完了: 完全なrun設定のtestと、依存境界のsuiteが成功する。
  - _Boundary: FedsdaRunSettings_
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 2. 最終構成の既定値
- [ ] 2.1 最終構成の設定の関数
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

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
