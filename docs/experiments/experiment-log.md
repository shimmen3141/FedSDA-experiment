# 実験の台帳（新実装）

新実装（`run_experiment_sweep.py`・`run_behavior_fix_comparison.py`）で回した実験を、1つの目的の実験につき1行で記録する。
結果のディレクトリ（`results/`）はgitの対象外なので、いつ・何のために・何を回し・どう判断したかを、ここで追えるようにする。
新実装の結果は、新実装のworktreeの`results/`に置く（2026-10-15、ユーザーの指示で、旧実装のcheckoutの`results/`から、`behavior-fix-comparison`・`method-baselines`・`fedsda-training-recency`を移した）。旧実装の`results/`には、旧実装の結果だけを残す。

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
| 2026-10-13〜14 | behavior-fix-comparison / `results/behavior-fix-comparison/20261013_120701_registration-short-alarm-wait`（2026-10-14に、もとの平らな形から移した。READMEは記録から後で作った） | LEGACY-016（登録）・LEGACY-002（短い警報）の修正と、IMPROVE-006（警報を待つ）の、精度・通信量への影響を、同じ条件・同じseedで比べる | 提案構成、6データ（sea2・sea4・circle2・sine2・mnist2・mnist4）、seed 0〜9、集約間隔50・200、8組合せ・960 run（4回の実行。2026-10-13: 登録legacy・fixed×短い警報legacy・fixed、480 run。2026-10-14: 登録waited-as-current×短い警報legacy・fixed、240 run。登録waited-as-current×短い警報fixed×警報を待つon、120 run。登録fixed×短い警報fixed×警報を待つon、120 run） | c577e52e2fe9（10-13）、ad5231049df1（10-14の2回）、0dbdf70a7dec（10-14の最後）。基準版は記録なし（runの要約も来歴を持たない） | [LEGACY-016の記録](../research/implementation-findings/legacy-016-registration-id-consumed-without-model.md)の「本規模の比較と、扱いの見直し」「seedを増やした比較と、既定を全部登録へ戻した判断」「本規模の比較（待つ登録と待つ警報を含む8通り）と、案Yの判断」 |
| 2026-10-14 | method-baselines / `results/method-baselines/20261014_141259_feddrift-fedavg-n5000`（2026-10-14に`results/20261014_141259_baselines-n5000`から移した。READMEは記録から後で作った） | FedDriftとFedAvgの新実装での基準値を、本規模で取り、FedSDAの既定（案Y）と比べる | FedDrift・FedAvg、6データ、seed 0〜9、集約間隔50・200、240 run | a30d8bd1e512。基準版は記録なし | 比較の表は、結果のディレクトリの`analysis_20261014_method_comparison.md`。判断は未 |
| 2026-10-14 | fedsda-training-recency / `results/fedsda-training-recency/20261014_202611_local-probes`（手元のWindows） | FedSDAがsea2で弱いことと、定常期の精度がFedDriftに届かないことの原因を切り分ける（計算量の内訳、構造の比較、直近K件の学習の試作） | 提案構成と構造の2派生、sea2・circle2・sine2、seed 0〜2、間隔50（sea2は200も）、標本5000。構造の比較18 run、直近K件の試作36 run、計算量の内訳2 run（標本2000） | 426b6f7（構造の比較と計算量の内訳。未commitの変更は文書だけ）、8409296（試作）。差は文書だけ。基準版5 | [研究バックログの7](../research/research-backlog.md) |
| 2026-10-14〜15 | fedsda-training-recency / `results/fedsda-training-recency/20261015_selection-probe`（手元のWindows） | 直近K件と指数減衰の学習batchの抜き出しを、実装した設定で手元で比べる（試作と同じ傾向かの確認） | 提案構成、`uniform`・`recent:300`・`decay:0.997`、sea2・circle2・sine2の間隔50とsea2の間隔200、seed 0〜2、標本5000（36 run）。2026-10-15に、同じ条件で`alarm`・`alarm-decay:0.99`を足した（24 run） | 7831e74f5042、8cbc306（追加分）。基準版5 | [研究バックログの7](../research/research-backlog.md) |
| 2026-10-15 | fedsda-training-recency / `results/fedsda-training-recency/20261015_000543_selection-n5000`（研究室サーバ） | 学習batchの抜き出し方（全履歴・直近300件・指数減衰0.99と0.997）を本規模で比べる | 提案構成、`uniform`・`recent:300`・`decay:0.99`・`decay:0.997`、6データ、seed 0〜4、集約間隔50・200、240 run | fddd5d29ced9。基準版5 | [研究バックログの7](../research/research-backlog.md) |
| 2026-10-15 | fedsda-training-recency / `results/fedsda-training-recency/20261015_circle2-diagnosis`（手元のWindows） | circle2ほかで、FedSDAの定常期の精度がFedDriftに届かない原因を切り分ける（学習データの純度、adapterのrank、round内の位置ごとの精度、FedAvgの局所更新の時機、FedSDAで予測を配布モデルで行う診断） | 診断ごとに異なる（README）。主なものは、FedAvg 30 run・FedSDA 30 run（circle2・sine2・sea2、間隔50、seed 0〜4、標本5000）と、serverの既存の結果の読取り | 2d352cf〜56e6001の作業ツリー（sourceは7831e74と同じ。FedSDAの診断は、実行時に差し替え）。基準版5 | [研究バックログの8](../research/research-backlog.md) |
| 2026-10-15 | method-baselines / `results/method-baselines/20261015_043929_fedavg-local-updates-per-aggregation-n5000`（研究室サーバ。コピー先の入れ子を直した） | 局所更新を集約ごとにまとめるFedAvg（元論文に沿った形）の基準値を、既存のFedDrift・FedAvgと同じ条件で取る | `fedavg-local-updates-per-aggregation`、6データ、seed 0〜9、集約間隔50・200、120 run | 340f191c6234。基準版5 | [研究バックログの8](../research/research-backlog.md) |
| 2026-10-15 | fedsda-prediction-timing / `results/fedsda-prediction-timing/20261015_060725_synchronized-vs-local-n5000`（研究室サーバ） | FedSDAの予測を、最後の同期の時点のモデルで行う構成と既定を、本規模で比べる | `fedsda`・`fedsda-synchronized-model-prediction`、6データ、seed 0〜9、集約間隔50・200、240 run | 0e32c7b989e9。基準版5 | [研究バックログの8](../research/research-backlog.md)（既定にするかは未決） |
| 2026-10-15 | fedsda-prediction-timing / `results/fedsda-prediction-timing/20261015_114326_synchronized-selection-n5000`（研究室サーバ） | 同期の時点の予測と、学習batchの抜き出し方（指数減衰0.997・直近300件）の組合せを、全履歴と比べる | `fedsda-synchronized-model-prediction`、`decay:0.997`・`recent:300`、6データ、seed 0〜9、集約間隔50・200、240 run（全履歴は`20261015_060725_synchronized-vs-local-n5000`の結果を使う） | fbf41a5b03c0（srcは0e32c7bと同じ）。基準版5 | [研究バックログの7・8](../research/research-backlog.md) |
