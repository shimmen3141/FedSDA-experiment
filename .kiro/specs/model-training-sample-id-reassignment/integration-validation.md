# 実測: モデル学習標本のID付替え

## 範囲・環境

学習標本storeの単一ID付替えのみ。評価用stored_data/容量、counter、正式登録全体・client/通信/新全体runは後続。
2026-10-06、共有../../venvのWindows CPU基準。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版と基準はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。

## Task1

- RED: 新API未実装のAttributeError、-xで1 failed/2.51秒/exit1。
- GREEN: 新92件＋既存storage/sampler=242 passed/4.76秒/exit0。対象Ruff/format成功。
- 旧confirm36条件（空/非空元、先既存/新先/同ID、元欠落/各ID順）、signed4、両引数不正6型×元先有無48、過去snapshot/後続追加1、opaque/float64/meta payload3。
- 実旧train_data_storeのTensor参照/標本順/モデル順一致、空列先上書き・重複保持、列連結なし、過去snapshot構造保持と借用payloadの変更共有を確認。
- Pyright0 errors/0 warnings。実Luna独立242passed/3.72秒、静的検査/scan成功、Task1 APPROVED。主担当も実測/型/実diffを照合し完了。

## Task2

- test-only接続。production変更なし、RED N/A。対象104 passed/3.06秒、Ruff/format成功。
- class2/4×Adam標準/AMSGrad/SGD×共有更新有無の12条件各3回。初回更新後に実旧confirmと新標本storeのID付替えを実行し、上位でbinding IDを対応する。
- 毎回の抽出Tensor、loss、全parameter/grad、個別・共有optimizer state、明示Random終端をexact照合。previewはRandomのdeepcopyを使い、実更新の乱数を追加消費しない。3共有RNGとtorch defaultsの不変、過去snapshotの旧IDも確認。
- 実Luna独立348passed/6.61秒、静的検査/scan成功、Task2 APPROVED。指摘なし。
