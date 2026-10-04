# 統合検証: Residual Adapterモデル構造

## 完成範囲
共有特徴抽出部・非線形Residual Adapter・分類層のCPU float32構築/forwardと、共有参照・勾配経路を完成した。
optimizer/loss/学習更新、生成後共有付替え、候補進行、データ名解決、ID/通信、新全体FedSDA run、baseline構造は後続。
旧名alias/state readerは導入していない。旧prefixへのmappingはtest-only。

## 要件と証拠
| 条件 | 観測済み契約 |
|---|---|
| 1.1 | hidden幅順/空列identityを実旧へ照合 |
| 1.2 | 実効rank制限、初期恒等、Linear draw後zero化 |
| 1.3 | binary1列Sigmoid、多クラスK列logits |
| 1.4 | 抽出済forward一致/再抽出なし |
| 1.5 | 全state・forwardとzero/nonzero展開の勾配旧一致 |
| 2.1 | forged/型値/共有実構造/未登録Parameterの拒否、事前CPU RNG/共有不変 |
| 2.2 | vector/batch/emptyとdtype/device/layout/rank/幅拒否、入力不変 |
| 2.3 | 後構築CPU RNG旧一致、forward無乱数、既定dtype/meta/grad/PythonNP保持 |
| 3.1 | extractor object/storage共有、adapter/classifier独立 |
| 3.2 | 入力/全Parameter勾配旧一致、forward/backwardで重み未更新 |
| 4.1 | 実旧oracle＋既存確率/観測meanlossとsnapshot選択→標準load接続 |
| 4.2 | exact依存・fresh起動・全tests旧11/最終3golden、旧無差分 |

## 実配置と依存
src/federated_learning_experiments/learning/models/内のshared_feature_extractor.py、
nonlinear_residual_adapter.py、residual_adapter_classifier.py。設計の三部品と一致。
対象testはtests/refactoring/test_residual_adapter_model_architecture.py。
ASTはtests/refactoring/test_single_run_dependency_boundaries.py。exact module/public symbolで固定し、一般stdlib許可より前に新3module規則を適用する。
本specのLOCAL境界外の学習/method orchestrationを隠れて実装していない。

## 検証結果
- 初回Task1 missing-module RED1error/exit1→11件成功。
- Lunaが未登録weight/biasの受理を発見。先行RED2 failed/11 passed→登録identity guardで13 passed、再reviewAPPROVED。
- Task2 test-only追加後53 passed。LunaAPPROVED、主担当fresh53 passed/2.31s/exit0。
- AST禁止28/許可21を追加。RED22 failed/297 passed/2.68s/exit1→GREEN319 passed/2.08s/exit0。主担当fresh319 passed/2.27s/exit0。
- 全tests: 3284 passed / 3 skipped / 1 warning / 115.87s / exit0。skipは既存Windows非対応wrapper、warningは既存qint8 fixture deepcopyのTypedStorage非推奨。
- fresh process: RESIDUAL_ADAPTER_MODEL_ARCHITECTURE_SMOKE_PASS / exit0、旧package importなし。
- 旧production/golden/旧回帰testは基準748c3aaから差分なし。golden更新・許容誤差変更なし。
- 追加の旧不具合は観測していない。今回のregistered guardは新実装のreview修正で、旧実装を修正した記録ではない。

## 実行環境・コマンド
共有venv Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1。golden基準手順はdocs/experiments/refactoring-baseline.md。
```powershell
$env:TMP=(Resolve-Path ../../venv/refactoring-tests).Path
$env:TEMP=$env:TMP
$env:MPLCONFIGDIR=(Resolve-Path ../../venv/matplotlib-cache).Path
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:FDE_MNIST_DATA_DIR=(Resolve-Path ../../data/mnist).Path
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/residual-adapter-model-architecture-final-20261004a
```
CPU fresh smokeは別processでPYTHONPATH=srcを設定し、共有抽出部注入の二分類器（二値/多クラス）を構築して実forward/抽出済経路/backward/CPU32/forward RNG不変/旧importなしを確認した。
```python
import sys
import torch
from federated_learning_experiments.learning.models.model_architecture_settings import ModelArchitectureSettings
from federated_learning_experiments.learning.models.shared_feature_extractor import SharedFeatureExtractor
from federated_learning_experiments.learning.models.residual_adapter_classifier import ResidualAdapterClassifier
settings=ModelArchitectureSettings(model_architecture_name="shared_backbone_residual_adapter",residual_adapter_requested_rank=8)
extractor=SharedFeatureExtractor(input_feature_count=2,hidden_layer_widths=(4,))
first=ResidualAdapterClassifier(model_architecture_settings=settings,input_feature_count=2,hidden_layer_widths=(4,),class_count=2,shared_feature_extractor=extractor)
second=ResidualAdapterClassifier(model_architecture_settings=settings,input_feature_count=2,hidden_layer_widths=(4,),class_count=3,shared_feature_extractor=extractor)
assert first.feature_extractor is second.feature_extractor is extractor
assert first.residual_adapter is not second.residual_adapter
assert first.residual_adapter.effective_rank==4
features=torch.ones(2,2,device="cpu",dtype=torch.float32,requires_grad=True)
rng=torch.get_rng_state().clone()
output=first(features)
assert output.shape==(2,1)
assert second(features).shape==(2,3)
assert torch.equal(first.forward_from_shared_features(first.extract_shared_features(features)),output)
assert torch.equal(rng,torch.get_rng_state())
output.sum().backward()
assert features.grad is not None
assert all(p.grad is not None for p in first.parameters())
assert all(p.dtype==torch.float32 and p.device.type=="cpu" for p in first.parameters())
assert not any(n.startswith("federated_drift_experiment") for n in sys.modules)
print("RESIDUAL_ADAPTER_MODEL_ARCHITECTURE_SMOKE_PASS")
```

## 最終判断
最終taskreviewとfeature統合GOは別gateで記録する。この文書の成功範囲はモデル構造・test-only接続であり、新全体runの完成ではない。
