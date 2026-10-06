# 実測: 損失統計のモデルID付替え

## 範囲・環境

既存統計storeの単一ID付替えだけ。モデル/標本/counterの付替え、現在帰属ID/保留解除を組み立てる正式登録全体、新client/全体runは後続。
2026-10-06、共有../../venvのWindows CPU基準。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版と基準はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。

## Task1

- RED: API未実装のAttributeErrorで63 failed/2.88秒/exit1。
- GREEN: 63 passed/1.94秒/exit0、対象Ruff/format成功。
- 実旧正式登録18ケース（負元/先既存・未登録/欠落/同ID/元0件）、汎用signedと同ID4ケース、両IDの不正型/元先有無40ケース、独立取得値と後続Welford更新1ケース。
- 内部ownerのみ更新、既存先位置維持/新先・同ID末尾、元欠落no-op、全クラス/全field、過去snapshotと返却値改変の非波及を確認。
- Pyright0 errors/0 warnings。独立Lunaも63 passed/2.11秒・Ruff/format/diff成功、Task1 APPROVED。指摘なし。主担当も実測GREENと型/境界を照合して完了判定。
