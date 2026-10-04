# LEGACY-008: 極大観測件数のサーバ損失平均が数値契約を外れる
- 発見日: 2026-10-04
- 基準: 748c3aa
- 発見spec: server-loss-mean-aggregation
- 対象: federated_drift_experiment/servers/base.py:168–180。共有バックボーンサーバも同じ通常加算/除算式。
- 種別/状態: 数値境界の不具合候補。再現済み・未修正。

## 観測と再現
BaseServer.update_global_modelsをモデル実体なしSimpleNamespaceへ直接呼ぶ。各clientは同一modelを所有、training件数1、get_paramsがPythonfloat dummy辞書を返し、通信hookはno-op。
1. n=(2**53,3,3)、各mean=1.0/M2=0.0をclient順に渡すと、n=9007199254740998、mean=1.0000000000000002、M2=0.0をglobal_statsへ保存。通常加算の丸めにより有界損失平均の上限1を超える。
2. n=(10**308,10**308)、各mean=1.0/M2=0.0では、合計intのfloat除算時にOverflowError。global_modelsは先に更新済みでglobal_statsは既存値のまま。
test_server_loss_mean_aggregation.pyのtest_server_loss_mean_aggregation_extreme_counts_reproduce_legacy008へ2入力の旧/新直接対照を追加した。
```powershell
../../venv/Scripts/python.exe -m pytest tests/refactoring/test_server_loss_mean_aggregation.py -q -p no:cacheprovider
```

## 影響と今回の扱い
極大件数の直接入力でのみ再現した。通常client/過去の研究成果に影響した証拠はない。
各入力件数は新BoundedLossMomentsで受理できるが、合計・結果は契約外になる。新pure関数はclipせず理由付きValueErrorで拒否し、入力/外部状態を更新しない。
これは旧サーバproductionの修正ではない。正常な実験規模の旧演算順・M2zeroは保持し、golden更新をしない。

## 将来修正案
算出結果/合計の検査をモデル・統計の保存前に行い、失敗時に部分更新を残さない。具体仕様と採否は未定。
fsum/clip/分散合成へ変更する場合は独立spec/commitで数値・過去成果への影響を検証する。旧修正commitはなし。

## 追跡検証
server-loss-mean-aggregation task2で両入力を直接再現し、新ValueError拒否と入力不変を確認。対象46テスト通過、独立LunaレビューAPPROVED。
通常client/過去成果への影響は依然未確認、旧productionは未修正。
