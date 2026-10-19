# Implementation Plan

- [x] 1. seed表記を正規化する引数操作を実装する
  - テスト先行で列挙・混在・端点・既定・複数回flagの最後優先・単独値/範囲双方のuint32拒否・上限前検査を固定する。
  - 単体parserの値/エラーが仕様どおりであることを完了条件とする。
  - _Requirements: 1, 2_
- [x] 2. seedを受け取る各CLIへ接続する
  - 4つの入口とhelpを揃え、同じ条件の列挙/範囲で計画・識別子が一致することを確認する。
  - 不正入力・print-planで実験と出力作成が起きず、再開照合も表記に依存しないことを確認する。
  - _Requirements: 1, 2_
- [x] 3. 回帰と独立レビューで完了する
  - 通常のexperiment-sweep、挙動比較の既存案内、prediction-initial-mass-diagnostic、component-ablation解析の既存案内を更新し、全task後の独立CLIレビュー・対象/研究checks・fresh・全pytest・品質検査を実測する。
  - Minor以下は修正のみで再レビュー不要、Major以上は再確認する。mainへマージしない。
  - _Requirements: 1, 2_

## Implementation Notes

初回仕様レビューのMajor1/Minor3は、単独値と範囲両端の共通uint32検証、繰返しflag、正式module名、文書対象を明記して修正。独立Luna highの再確認でPASS（指摘0）後に実装を開始した。最初のCLI再確認は利用上限で未完了でありPASSに数えていない。100000個はメモリを守るCLIの展開上限で、実験のパラメータ上限ではない。

新規31testは未実装importでRED後に実装。対象/通常掃引/再開/挙動比較/依存境界3280件成功、研究CLI専用checksは個別に19件・26件成功。両研究checksは同名moduleのため同じpytest呼出しへまとめず、既存手順どおり別processで実行した。旧の重複拒否テストは、計画構築前のargparseエラーを許容するよう調整した。型検査0 errors、Ruff/format・公開名登録検査も成功。

独立Luna highの実装レビューはPASS、Blocker/Major/Minor0。独立pytest3154件成功、tmp_pathを使う4件は既定Temp/専用basetempともfixture準備の権限拒否で未実施。主担当の3280件成功ログ（4件を含む）と型検査0 errorsを照合した。独立Ruff/formatも成功。レビュー用一時ディレクトリは主担当が空を確認して非再帰で片付けた。

共用freshのフロー数は維持して16フロー成功。通常CLIのseed入力を`0-0`に変え、保存と再開を同じ値のまま実行した。新規依存なし、pip check成功。

主担当の固定Windows全pytestは14443 passed/8 skipped、exit0、517.45秒。直前14412 passed/8 skippedに新規31件を足した想定と一致。JUnitは`../../venv/refactoring-tests/seed-range-full.xml`、ログは同`seed-range-full.log`。旧11/最終3goldenを各1testcase内で成功確認。freshログは同`seed-range-fresh.log`。

独立Luna highが完了証拠を確認しPASS、追加指摘0。JUnitの14451件・failure/error0・skip8、新規31件の成功、2回帰testcaseの成功、fresh16の成功マーカー、methods/learning/data/runtime・旧回帰test/goldenの差分なしを照合した。全pytest/fresh自体は主担当の実測であり独立再実行ではない。CLI/研究toolsの引数処理は意図した変更で、汎用identityの固定旧tools全体との空diff条件は適用できないため、保護対象を別途照合した。

全task完了。検証source/testのcommitは`37bc7e0`。tracked Python/2goldenのLF hashはspec.jsonのintegration_validationへ記録した。mainへマージしない。
