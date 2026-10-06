# 実測: モデル別評価標本の保持

## 環境と範囲

2026-10-06、共有../../venvのWindows CPU基準。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版と手順はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。
評価標本の保持/抽出/ID対応のみ。学習標本・評価fallback・全登録/client/runは後続。

## Task1

- RED: 新evaluation package未実装のModuleNotFoundError、1 collection error/2.73秒、exit1。
- 初回GREEN途中で旧oracleのcurrent ID=999と欠落ID対応表999→12が衝突し、評価owner以外のmappingイベントに入ってfixture不足の9失敗となった。current IDを対応表外9999へ直し、Ruffのloop変数上書きを修正。production挙動の失敗ではない。
- GREEN: 290 passed/1.99秒、Ruff/format成功。144追加条件・18付替え条件・72再編条件、固定入力拒否13・操作拒否32・payload/snapshot3・empty/no-op Random検査8。
- 実旧処理を同じ開始Random stateで実行し、全ID順/標本順/Tensor identityと終端Random stateを照合。旧oracleは共有Python RNGをfinallyで復元する。
- Pyrightはsandbox実行で基準venvのnumpy/torchを解決できず失敗したため、従来の基準手順と同じ権限設定で再実行する。
- 通常権限での基準Pyright再実行は0 errors/0 warnings、exit0。
- 実Luna Task1 APPROVED。独立290passed・静的検査/scan成功、指摘なし。主担当も実diffと上記実測を確認。
