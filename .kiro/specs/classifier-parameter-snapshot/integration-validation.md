# 実測: 分類器parameter snapshot

## Scope
一分類器の全parameter現在値の独立コピー。候補選択/平均/復元はtest-only接続で、登録/送信保留/新client/新全体runは未移植。要求/設計rev1、命名rev2、tasksが正本。

## Task1
- 未実装module先行RED: ModuleNotFoundError、1 collection error/2.09秒/exit1。
- GREEN: 16 passed/3.37秒/exit0。class2/4/10の実旧get_paramsと全native key順/shape/値/独立storageを照合。
- 隠れ層空/一層/二層、非連続parameter、相互変更と再取得、共有参照・既存grad・混合training flags保持、forward0、Python/NumPy/Torch乱数、default dtype/device/gradmodeとinference_modeを確認。
- 型/subclass、NaN/Inf/float64/meta/sparse拒否と不正parameterの値・grad参照/値保持。Ruff成功、対象format成功。
- 主担当の最終16 passed/3.62秒、Pyright0 errors/0 warnings。独立Luna実行も16passed・品質成功、Task1 APPROVED。

## 再実行環境
2026-10-06、既存Windows CPU基準、共有../../venv。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版と基準はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。goldenは固定旧実装を実行し、新全体run完成を意味しない。
