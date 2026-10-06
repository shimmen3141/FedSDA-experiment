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

## Task2

- test-only接続なのでproduction RED N/A。class2/4×Adam標準/AMSGrad/SGD×共有更新/凍結の12条件で各3共同更新。
- 最初の更新後、実旧BaseClient.confirmで旧NNのIDを付替え、新registryでも同じタイミングに付替える。旧samplerのIDだけをtest側で対応し、標本/反復順は維持する。
- 全loss/parameter/grad/個別と共有optimizer stateを各更新でexact照合。付替え直前直後のparameter/grad/個別optimizer蓄積不変・wrapper/owner参照・統計/保留不変を確認。統計は上位で別途付替える。
- 3乱数/default dtype/device/gradmode、保留record/残回数/固定snapshot保持を確認。固定snapshotの独立比較基準をtest-only deepcopyし、その局所名を命名revision3へ補足してレビューに戻す。production/API命名はrevision2から不変。
- 83 passed/2.87秒/exit0、対象Ruff/format成功。
- 実Luna独立83 passed/2.87秒、Task2/命名revision3 APPROVED。指摘なし、主担当も実測/実diffを照合し完了。

## Task3

- 対象＋既存AST: 747 passed/5.80秒/exit0。依存追加なし、guard変更/新RED N/A。
- fresh新CPU processでclass2/4のproducer/loss/store/pendingとregistry/蓄積optimizerを接続。単一ID付替え→現在binding→上位統計付替え→明示clear、古いrecord IDとNN/owner/optimizer参照保持、旧package非importを確認。exit0。
- Ruff成功/format116 files already formatted/pip成功/diff-check成功、Pyright0 errors/0 warnings。固定748c3aaとの差分（旧production/両golden/旧回帰test）は空。
- source216 Python/両goldenのSHA256: `cd2ff4277d326f4de5af825f9dc19e3e179363f64a3c730e6a2f5f4d3e594a2b`。追跡path sort→UTF8相対path+NUL+LF正規化内容+NUL。要求/設計revision1・命名revision3の承認hash一致。
- 全pytest: 4642 passed/3 skipped/1既存warning/123.09秒/exit0。旧11条件/最終3条件goldenも成功。golden値・許容差は変更していない。skipは既存Windows wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。
- JUnit training-state-id-full.xml: 4645 tests/0 failures/0 errors/3 skipped。
- 実Luna Task3 APPROVED。独立747 passed/4.07秒・fresh/全回帰/品質/固定旧差分空を確認。承認後の主担当gateも747 passed/3.62秒・fresh/diff成功。全3task完了後の別featureレビューへ進める。
- 今回、新たな旧正常実装の不具合は観測していない。

## 全8要件trace

|要件|証拠|
|---|---|
|1.1|実旧18ケース、NN/owner identity、12条件の旧NN確認と学習継続|
|1.2|独立NN/ownerの既存先を上書き、先位置/他ID順維持|
|1.3|新先/同ID末尾、汎用signed4ケース|
|1.4|空/元欠落×先有無no-op、生成なし|
|2.1|両ID×不正6型×元先有無48拒否、一覧/parameter/grad/optimizer不変|
|2.2|旧state/binding/一覧のIDと参照保持、新wrapperの先ID|
|2.3|蓄積state/parameter/grad保持、reset後現在binding、12条件3共同更新の全loss/値/状態|
|2.4|上位test統計付替え/registry単独の統計と保留不変、3乱数/defaults、既存AST/fresh/品質/固定golden gate|

```text
python -m pytest tests/refactoring/test_held_model_training_state_id_reassignment.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/training-state-id-full-20261006 --junitxml=../../venv/refactoring-tests/training-state-id-full.xml
python ../../venv/refactoring-tests/held_model_training_state_id_smoke.py
python -m ruff check .
python -m ruff format --check .
python -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
python -m pip check
```
