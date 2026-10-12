# Requirements Document

## Introduction
FedDriftの観測済みストリームを、clientの未完了検出batchの確定から条件付き最終回収・集約・配布まで接続する。通常同期とは別の旧終端経路の数値・更新順・乱数・通信を維持する。全体run指標・成果物・CLI・golden移植は後続。

## Requirements
### Requirement 1: 全clientの検出buffer確定
1.1 When 終端が呼ばれた時, the FedDrift終端処理 shall 全clientの観測済み検出bufferだけをclient順に確定し、未観測標本・予測・観測件数を追加しない。
1.2 When 全bufferが空かつreadyモデルが無い時, the FedDrift終端処理 shall モデル・optimizer・通信・診断・乱数を変更しない。
### Requirement 2: 条件付き最終回収と配布
2.1 When buffer確定後にreadyモデルが一つ以上ある時, the FedDrift終端処理 shall client順にready snapshotと統計を正式IDへ登録し、lineageと確認を反映し、repositoryのID順に1回だけ実旧同様の集約・全client配布を行う。
2.2 While 終端を進める間, the FedDrift終端処理 shall ローカル学習・クロス評価・クラスタリング・隔離期限追加・状態要約通知を行わず、既存の集約重みを維持し、新登録の重みだけ初期値で追加する。readyが無い場合はserver処理を行わない。
2.3 When 最終配布が完了した時, the FedDrift終端処理 shall 実旧同様に残る待機中の保留をreadyへ昇格する。server処理を行わなかった場合は昇格しない。
### Requirement 3: 接続境界と検証
3.1 If 時刻・設定・client/serverの入力契約または検出bufferが不正な場合, then the FedDrift終端処理 shall 入力検査で判別できる不正を全clientのflushに先立って拒否し、状態・通信・3乱数を保つ。各数値生成は既存部品と集約の更新候補内で検査し、予期しない実行時障害の全client rollbackは提供しない。
3.2 The FedDrift終端処理 shall 勾配記録を無効にした呼出しにも対応し、factoryで準備した実体から観測・同期・終端まで接続でき、新実装だけのfresh processと実旧対照で検証できる。
