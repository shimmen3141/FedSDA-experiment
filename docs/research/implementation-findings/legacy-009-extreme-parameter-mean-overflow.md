# LEGACY-009: 極大parameter値の単純平均が非有限になる
## 種別・状態
2026-10-04発見。再現済み・未修正の境界入力不具合候補。基準commit748c3aa、対象clients/fedsda.py:574–586 _average_model_paramsとその初期化呼出。
発見spec: candidate-parameter-initialization。

## 再現条件と観測
CPU float32でtorch.finfo(torch.float32).maxを同じ1要素parameterとして2モデルに設定する。
入力は双方finiteだが、実旧FedSDAClient._average_model_paramsのtorch.stack(values).mean(dim=0)がinfを返す。
モデル生成なしget_params deepcopy stubで実旧unbound helperを直接呼び、同入力の新APIがValueError/入力不変となることを対照する。
再現test: tests/refactoring/test_candidate_parameter_initialization.py のtest_candidate_parameter_initialization_rejects_nonfinite_average_results。
コマンド: ../../venv/Scripts/python.exe -m pytest tests/refactoring/test_candidate_parameter_initialization.py -q -p no:cacheprovider
TMP/TEMPとMPLCONFIGDIRはrefactoring-baselineの既存検証先を指定する。

## 期待・影響
有限な入力でも算出結果が非有限なら、初期値として黙って返さず問題を報告できること。
通常のモデル値・過去成果・実際の後続学習や採否への影響は未確認。単純平均を使わない最終標準の最小損失選択を、この再現だけで不具合ありとは扱わない。

## 今回の扱い
旧productionは変更しない。新APIはCPU dtypeと旧stack/mean演算順を維持し、算出結果のfinite検査でValueErrorを返す。
clip、float64化、安定平均の別式への変更は行わない。正常域の数値は旧oracleで照合する。
新APIの拒否を旧production修正済みと表現しない。goldenは更新しない。

## 将来の修正
初期parameter平均の責務で、旧経路に算出結果検査を追加するか、演算式の変更が必要かを別変更で判断する。
演算式を変えるなら通常値の数値/演算順・学習結果・研究条件への影響と再実験範囲を検証する。現時点では未定。

