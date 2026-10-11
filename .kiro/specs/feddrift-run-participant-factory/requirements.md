# Requirements Document

## Introduction
研究者がFedDriftの新実装で実験を進めるため、移植済みの初期学習・client・serverを、手動で組み立てずrunごとの参加者集合として準備する。現在は個別部品の照合まで完了している。旧の_setup_server_and_clientsと同じ初期値・乱数順序を持つ集合を返し、そのまま観測と時刻同期に使えるようにする。

## Boundary Context
- In scope: 概念0の事前学習から全clientとserverの準備、入力検査、独立したrunの再準備、観測・同期への接続。
- Out of scope: stream実行ループの契約変更、終端run_round、精度評価、指標/成果物/CLI、全体run/golden移植。
- Adjacent expectations: 既存の事前学習とFedDrift client/serverを再利用し、数値・判断を複製しない。

## Requirements

### Requirement 1: 参加者の初期準備
1.1 When runの条件と設定を渡して準備を依頼する, the FedDrift participant factory shall 概念0で初期モデルを一度だけ事前学習し、ID 0の全clientを0から順に作り、同じ初期parameterと統計を持つserverを返す。
1.2 The FedDrift participant factory shall client間およびserverとの可変モデル/optimizer/統計/ownerを共有せず、全参加者がrunの同じPython乱数源を借用する。
1.3 When 初期準備を実行する, the FedDrift participant factory shall 初期化・標本生成・shuffleの旧順序を維持し、値・optimizer・統計・server記録・Python/NumPy/torchの乱数状態を実旧へ一致させる。

### Requirement 2: 検査と再準備
2.1 If 設定・run条件・乱数源・標本生成器またはその組合せが契約外である, the FedDrift participant factory shall モデル生成・標本生成・乱数消費・直前の参加者の置換より前に拒否する。
2.2 When 準備を繰り返す, the FedDrift participant factory shall 新しい参加者とownerを返し、直前の成功した集合だけを公開する。検査だけでは参加者を生成しない。

### Requirement 3: 接続と境界
3.1 When 準備した集合へ観測処理と時刻同期を依頼する, the FedDrift participant factory shall 既存client/serverへそのまま接続でき、実旧の初期準備後の同じ操作と状態・通信・乱数が一致する。
3.2 The FedDrift participant factory shall 旧実装やtestを実行時に読み込まず、固定旧/goldenを変更しない。新実装だけの別processで準備から観測・同期まで実行できる。
