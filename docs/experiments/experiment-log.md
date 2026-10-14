# 実験の台帳（新実装）

新実装（`run_experiment_sweep.py`・`run_behavior_fix_comparison.py`）で回した実験を、1つの目的の実験につき1行で記録する。
結果のディレクトリ（`results/`）はgitの対象外なので、いつ・何のために・何を回し・どう判断したかを、ここで追えるようにする。

## 規約

- 1行は、1つの目的の実験（コマンドの1回の実行ではない。同じ目的で組合せを足した実行は、同じ行にまとめる）。
- 実験を始めたときに行を足す（結論は「未」）。結論が出たら、結論の参照を書く。結果の数値と判断の本文は、
  findingや改善候補の記録（`docs/research/`）へ書き、台帳には書かない。
- 結果のディレクトリの`README.md`（コマンドが作る）の「結論」にも、同じ参照を書く。
- 研究の問いは、結果のディレクトリの1段目（`results/<研究の問い>/<開始日時>_<ラベル>`）と同じ名前にする。
- コードは、結果の`sweep_record_*.json`・`comparison_record_*.json`の`code_revision.commit`と`behavior_baseline_version`から書く。
  記録に無いものは「記録なし」。
- 固定旧（`run_pareto_sweep.py`・`tools/run_server_sweep.sh`）の実験は、ここに載せない（`results/`の旧の結果と、各記録を参照）。

## 列

| 列 | 内容 |
|---|---|
| 日付 | 実行を始めた日（複数回に分けた実行は、最初と最後） |
| 研究の問い / 結果 | 研究の問いと、結果のディレクトリ（`results/`からの位置） |
| 目的 | 1行 |
| 規模 | 構成・組合せ、データ、seed、集約間隔、run数 |
| コード | commit（12文字）と基準版 |
| 結論の参照 | findingや判断の記録へのリンク。未定なら「未」 |

## 台帳

| 日付 | 研究の問い / 結果 | 目的 | 規模 | コード | 結論の参照 |
|---|---|---|---|---|---|
| 2026-10-13〜14 | behavior-fix-comparison / `results/behavior-fix-comparison`（このしくみより前の平らな形。3回の実行が同じ場所にある） | LEGACY-016（登録）・LEGACY-002（短い警報）の修正と、IMPROVE-006（警報を待つ）の、精度・通信量への影響を、同じ条件・同じseedで比べる | 提案構成、6データ（sea2・sea4・circle2・sine2・mnist2・mnist4）、seed 0〜9、集約間隔50・200。2026-10-13: 登録legacy・fixed×短い警報legacy・fixed（480 run）。2026-10-14: 登録waited-as-current×短い警報legacy・fixed（240 run）、登録waited-as-current×短い警報fixed×警報を待つon（120 run） | c577e52e2fe9（10-13）、ad5231049df1（10-14の2回）。基準版は記録なし（runの要約も来歴を持たない） | [LEGACY-016の記録](../research/implementation-findings/legacy-016-registration-id-consumed-without-model.md)の「本規模の比較と、扱いの見直し」「seedを増やした比較と、既定を全部登録へ戻した判断」「本規模の比較（待つ登録と待つ警報を含む8通り）と、案Yの判断」 |
| 2026-10-14 | behavior-fix-comparison / 未確認（上の出力先の記録に無い） | 全部登録の下で、警報を待つ扱いの効果を比べる（上の8通りの残りの1つ） | 登録fixed×短い警報fixed×警報を待つon、上と同じ条件（120 run） | 未確認 | 上と同じ |
