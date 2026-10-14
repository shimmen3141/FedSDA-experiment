# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `collect_behavior_baseline_mismatch_descriptions` | `sweep/experiment_sweep_execution.py` | 計画のrunのうち、出力先の済んだrunの基準版が今と違うものの理由を、計画の順で集める（何も変えない） |
| `validate_plain_directory_name` | `sweep/experiment_sweep_command.py` | 研究の問い・ラベルが、1段のディレクトリ名として使えるかを確かめる |
| `write_experiment_readme` | `sweep/experiment_sweep_records.py` | 結果のディレクトリに、目的と出どころを書いた`README.md`を、無ければ作る |
| `BEHAVIOR_FIX_COMPARISON_STUDY_NAME` | 直下の`run_behavior_fix_comparison.py` | 比較scriptの研究の問いの名前（`behavior-fix-comparison`） |
| `--study`・`--purpose`・`--label`・`--stdout` | コマンドの引数 | 研究の問い、目的、ディレクトリ名のラベル、分析の結果を標準出力へ出す |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `execute_experiment_sweep` | 基準版の確認を、中で行う | 同じ確認を、`collect_behavior_baseline_mismatch_descriptions`で行う（動作は同じ） |
| `write_sweep_record` | 目的を持たない（形式の版2） | `purpose`・`study_name`を書く（形式の版3） |
| `analyze_behavior_fix_comparison.py`の`main` | 結果を標準出力へ出す | 既定で、入力のディレクトリの中のファイルへ書く。`--stdout`で標準出力 |
