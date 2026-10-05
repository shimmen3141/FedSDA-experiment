# 統合検証: 一回の共同モデルパラメータ更新

## 判定範囲
Task1/2/3の承認・現在地の正本はspec.json/review.md。本証拠は一回共同更新部品に限定する。
参加抽出/反復、optimizer生成・共有所有/reset/復元、候補進行、計算/診断counter、通信、PCGrad、新全体runは含めない。

## 環境とコマンド
Windows CPU、共有../../venv: Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1。
固定版/再構築はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpuを参照。
TMP/TEMPは../../venv/refactoring-tests、MPLCONFIGDIRは../../venv/matplotlib-cache。
対象: python -m pytest tests/refactoring/test_joint_model_parameter_update.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider。
全体: OMP_NUM_THREADS=1/MKL_NUM_THREADS=1/FDE_MNIST_DATA_DIR=../../data/mnist、python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/joint-model-parameter-update-final-20261005a。
fresh smokeはPYTHONPATH=srcの別processで実行した。

## 実測
- Task1実REDはmissingmoduleの1 collection error/exit1。GREEN26条件。24条件×3stepの全Parameter/grad/optimizerstate/損失を実旧_train_heads_togetherへtorch.equalで照合、2条件でzero/共有forward一回/backward一回/step入力順を観測。
- Task2の51追加条件を含む対象77 passed。41事前拒否はwarm Adamstate/sentinelgrad下のzero/step未呼出と全値/grad/state/input/settings/RNG保持。空列no_grad no-op、softbinary/非contiguousの実旧照合、複数group、warm共有凍結、frozen/kwonly、ambientfloat64/metaとPython/NP/Torch RNG保持を検証。
- 空共有は共有optimizer=Noneで独立同値NNの標準BCE/backward/Adamstepと2step全値/grad/state一致。共有更新有効/無効両方と空共有へoptimizer誤指定の拒否を検証。
- ASTは34禁止/18許可の追加を先行。実RED21 failed/417 passed/exit1、GREEN438 passed/exit0。exact二module/publicsymbolsのguardを一般stdlib許可の前へ配置。
- 主担当fresh対象＋AST: 438 passed/exit0/4.44s。
- 全tests: 3493 passed/3 skipped/1 warning/exit0/125.24s。旧11/最終3goldenを含み値/許容差は更新していない。3skipは既存Windows非対応wrapper、1warningは既存qint8fixture deepcopyのTypedStorage非推奨。
- fresh CPU: JOINT_MODEL_PARAMETER_UPDATE_SMOKE_PASS/exit0。通常/共有凍結/空共有/空列、optimizerstate更新/RNG維持、legacypackageがsys.modulesに存在しないことを確認。
- git diff 748c3aa -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json tests/test_regression.py tests/test_proposed_regression.pyは空。旧production未変更。
- 全approval hashはLF正規化、UTF8の置換文字や文字化けなし、File Structure Plan二production/二testと一致。

## 17要件の証拠
| 条件 | 実装/確認 |
|---|---|
| 1.1 | 明示APIと参加record、既存設定publiccopy |
| 1.2 | empty participationでNone、grad/state保持 |
| 1.3 | exact型/forgedsettings/nonmean拒否 |
| 2.1 | 正N/finite/共有identity/入力D/labelsN1のzero前検査 |
| 2.2 | 二値[0,1]、多クラス整数範囲 |
| 2.3 | 分類器/optimizer/Parameter重複・参照列/順序・requiresgrad |
| 2.4 | 非空共有でoptimizer欠落をtruefalse拒否、空共有None独立学習 |
| 3.1 | shared/個別zero順、cat/共有forward一回 |
| 3.2 | BCE/CE、標本数積sum入力順/総数と旧loss一致 |
| 3.3 | backward一回→sharedstep→個別step順、返却float/全state |
| 3.4 | sharedno_grad/step省略/warmstate維持/gradNone/個別更新 |
| 4.1 | 全検査zero前/後段不正の全state保持/外側no_grad拒否 |
| 4.2 | borroweddata/structure/groups/RNG/dtype/device/gradcontext保持 |
| 4.3 | designのrollback非保証と実装にsnapshotrollbackなしを確認 |
| 5.1 | 実旧oracleの24数値条件×3step、追加softbinary/strided |
| 5.2 | 空共有正常拡張を独立標準loss/stepで検証 |
| 5.3 | exactAST/freshCPU/fulltests/golden無変更 |

## 旧実装の記録と残る境界
[LEGACY-010](../../../docs/research/implementation-findings/legacy-010-empty-shared-feature-optimizer.md)の旧空共有optimizer生成失敗は未修正。通常/過去実験への影響は未確認。新一回更新入口のNone対応を、上位optimizer寿命/所有まで移植済みとは扱わない。
0行batch・重複参加・多クラス小数の旧挙動はresearch.mdの契約外調査に記録。通常サンプリングで発生する不具合との実証はなく、旧正常挙動の修正として混ぜない。
新全体run接続、全研究条件の同値性、学習中の数値安定性/演算失敗rollbackは本検証の主張に含めない。
