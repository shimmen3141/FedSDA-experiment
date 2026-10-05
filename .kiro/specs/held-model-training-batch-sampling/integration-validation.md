# 統合検証: 保有モデルの学習バッチ抽出

## 判定範囲
Task1–3の抽出/契約/上位接続とTask4の依存境界・全回帰を検証した。最終feature判定はLuna統合レビュー待ち。
現在地・承認hashの正本はspec.json/review.md。抽出器は標本ストアや乱数生成器を所有しない。
標本追加/帰属/ID統合、optimizer所有/reset、学習反復/counter/候補進行/通信、新全体runは本specの完成範囲に含めない。

## 環境とコマンド
Windows CPU、共有../../venv: Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1。
固定版/再構築はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpuを参照。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache。
対象: python -m pytest tests/refactoring/test_held_model_training_batch_sampling.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider。
全体: OMP_NUM_THREADS=1/MKL_NUM_THREADS=1/FDE_MNIST_DATA_DIR=../../data/mnist、python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/held-model-training-batch-sampling-final-20261005a。
fresh smokeはPYTHONPATH=srcの別processで抽出→分類器/optimizer対応→実共同更新を実行。

## 実測
- Task1 missingmodule実RED: 1 collection error/1.70s/exit1。GREEN7 passed。6条件×3callでpopulation5/100、非昇順/負ID、B1/B=N、同参照別位置、空/skipの全ID/Tensor/抽出順/終端RNGを実旧samplerへ完全照合。参加者ごと一回drawの順序をwrapで観測。
- Task2対象44 passed。31拒否条件で後段未抽出標本まで事前検査しsample未呼出/借用RNG/全参照/値/既存grad保持を確認。未保有payload field未読、不足内容skipと保有collection構造拒否を区別。records frozen/kwonly/defaultなし、Parameter入力/非contiguous/異なるモデル間D/finite softと範囲外label受理、B1/B2独立storage、float64/meta/gradmodeとglobalPython/NP/Torch保持を検証。
- Task3対象56 passed。12条件×3stepのtest-onlyFIFO release/drain→上位観測解決→model列→抽出→ID対応→実共同更新で、旧joint内部の実samplerをstepごと一回だけ観測。二値/4クラス、Adam standard/AMSGrad/SGD、共有更新/凍結の抽出全Tensor/loss/全Parameter/grad/optimizerstate/終端RNGが完全一致。数値本体のmockなし。
- ASTは34禁止/8許可の追加を先行。実RED16 failed/443 passed/4.55s/exit1から、exact二module/publicsymbolsのguardを一般stdlib許可の前へ配置してGREEN459 passed/4.12s/exit0。主担当fresh対象＋ASTも459 passed/4.50s/exit0。
- 全tests: 3591 passed/3 skipped/1 warning/115.24s/exit0。旧11/最終3goldenを含む。3skipは既存Windows非対応wrapper、1warningは既存qint8fixture deepcopyのTypedStorage非推奨。
- fresh CPU: HELD_MODEL_TRAINING_BATCH_SAMPLING_SMOKE_PASS/exit0。通常/共有凍結/空共有の抽出→共同更新、optimizerstate更新/共有凍結値保持、抽出と学習のTorch RNG保持、旧packageがsys.modulesに存在しないことを確認。
- 全回帰実測時のtracked Pythonと両goldenの内容hashは392dbd7441b9ab4ff2edd119fd67d4bf87d530e05ba76d6bdf61ba23d0e20538。文書更新とレビュー後も同状態との一致を確認する。
- git diff 748c3aa -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json tests/test_regression.py tests/test_proposed_regression.pyは空。旧productionとgolden/許容差は未変更。
- 全承認hashはLF正規化で一致、UTF8/配置を確認した。新二productionと対象/AST testはdesignのFile Structure Planに一致する。

## 12要件の証拠
| 条件 | 実装/確認 |
|---|---|
| 1.1 | collection入力順/held・件数filter、非昇順/負IDの実旧一致 |
| 1.2 | standardRandom.sample一回/抽出順cat、同参照別位置/B1/B=N |
| 1.3 | 未保有payload未読/不足内容skip、drawなし |
| 1.4 | empty/noeligibleで()とRNG不変、held-onlyID非生成 |
| 2.1 | exact型/ID/bool/重複/正count/generatorのdraw前拒否 |
| 2.2 | 後段/未抽出標本のfinite/shape/列内D違反でもdrawゼロ |
| 2.3 | 全参加populationとskip対象の異なる検査範囲 |
| 2.4 | tuple/record/Tensor参照・値/grad保持、cat独立storage |
| 2.5 | 借用Randomだけ進む、globalPython/NP/Torch/default/grad保持 |
| 3.1 | actual旧抽出全Tensor/ID/order/複数call/終端RNG exact |
| 3.2 | test-onlyFIFO位置解決→抽出→ID/NN/optimizer対応→実共同更新 |
| 3.3 | exactAST/禁止許可注入/freshCPU/全tests旧11最終3golden不変 |

## 旧実装の記録と残る境界
本specで通常経路の新たな旧不具合は観測していない。既存findingsの修正状態は変えない。
旧sampleのconcept metadataとNN/optimizer対応はtest-onlyで明示し、新通常APIへ互換窓口を持ち込まない。
上位接続の検証は標本ストア/FIFO帰属判断やclient進行をproductionへ移植済みという意味ではない。
クラス上限/共有モデル入力幅は後続共同更新が検査する。資源不足や外部改変の抽出開始後rollback、新全体run/全研究条件同値性は本検証の主張に含めない。

