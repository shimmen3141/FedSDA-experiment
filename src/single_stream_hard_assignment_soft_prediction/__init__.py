"""単一のデータストリームで、FedHASPの仕組み（変化の検出、MDLによる再利用・新規作成、固い割り当て、Fixed-Shareの
混合予測）をHoeffding treeの学習器で動かす手法。

製品の`federated_learning_experiments`とは別のパッケージ。製品のtorchを使わない判断の部品だけをimportし、製品からは
importされない（spec `single-stream-hard-assignment-soft-prediction`）。
"""
