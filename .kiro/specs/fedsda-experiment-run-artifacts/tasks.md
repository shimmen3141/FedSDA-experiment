# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [ ] 1. 実行と表
- [ ] 1.1 設定から、計測つきの全体runと指標の導出までを行う入口
  - 別々に呼んだ場合との一致、乱数を進めないこと、同じ設定からの同じ結果、拒否のtestを先に書く。
  - _Boundary: fedsda_experiment_run_
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 1.2 記録の表
  - goldenの条件のfixture（指標の導出のtest）を使い、表だけから旧の31の配列を作って実旧と照合するtest、ラウンドごとのモデル数の照合（goldenの3ケースと、小さい条件）、表の形・型・行の数の検査を先に書く。
  - _Depends: 1.1_
  - _Boundary: fedsda_run_record_tables_
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 2. 保存
- [ ] 2.1 要約と表の保存・読込み
  - 往復、上書きの拒否、書きかけを残さないこと、非有限の値、形式の版のtestを先に書く。保存用の辞書への変換へ、非有限の値の扱いを足す。
  - _Depends: 1.2_
  - _Boundary: fedsda_run_artifact_files、settings_serialization_
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 2.2 goldenの3ケースを、設定から実行・保存・読込みして、実旧と照合する。共用scriptへ足す
  - _Depends: 2.1_
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 3. 検証
- [ ] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 2.2_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
