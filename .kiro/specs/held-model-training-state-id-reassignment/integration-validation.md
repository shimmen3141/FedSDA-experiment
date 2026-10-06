# 実測: 保有モデル学習状態のID付替え

## 範囲と環境

既存registryの一モデルID付替えだけ。正式登録全体/標本/counter移動、client/通信/新全体runは後続。
2026-10-06、共有../../venvのWindows CPU基準。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版と基準はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。

## Task1

- RED: 新API未実装で71 failed/2.87秒/exit1。代表を-x/shortで再確認しAttributeError（1 failed/1.70秒）。
- GREEN: 新71件＋既存registry32件=103 passed/3.43秒/exit0。対象Ruff/format成功。
- 実旧confirmのpop/代入18ケース。unitでは新NNを旧modelsの値として渡し実メソッドの参照保持/辞書順を観測（旧NN内部数値の対照はtask2）。汎用signed/同ID4ケース、両引数×不正6型×元先有無48ケース、古い記録/蓄積optimizer/通常reset1ケース。
- 変更先の新wrapper、元/先の別NNとownerのidentity、旧record/binding/一覧保持、parameter/grad参照と値、optimizer蓄積値保持・reset後現在bindingを確認。
- Pyright0 errors/0 warnings。実Luna独立71 passed/3.07秒、静的検査/scan成功、Task1 APPROVED。主担当もGREEN103件と型/契約/実diffを照合して完了。
