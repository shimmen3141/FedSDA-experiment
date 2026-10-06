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

## Task2

- test-only接続なのでproduction RED N/A。追加時に既存test末尾を新関数の後へ挿入した編集ミスでRuff未定義変数と4 failed/63 passedを観測。既存assertを元の関数へ戻して修正。production変更なし。
- 67 passed/2.63秒/exit0、対象Ruff/format成功。
- class2/4×delay1/2の4条件で実旧register→保留→生統計更新→ready→confirmと、新producer/loss/initializer/store/pending→現在統計取得→付替え→明示clear→変更先統計更新を照合。
- 全native parameter snapshot値を旧prefix明示対応で比較し、登録後モデル変更はsnapshotへ非波及。付替え後も保留record/snapshot/対応ID/ready状態を保持し、明示clearのみ解除。
- 統計全field/モデル順、parameter/grad参照と値、Python/NumPy/Torch乱数、既定dtype/device/gradmode保持を確認。
- 独立Lunaも67 passed/3.23秒・Ruff/format/diff成功、Task2 APPROVED。指摘なし、主担当も証拠と実diffを照合して完了判定。
