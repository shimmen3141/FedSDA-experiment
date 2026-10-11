# Requirements Document

## Introduction
連合学習の実験開発者は、FedDriftの観測列を新実装へ移植したが、ラウンド内の局所学習は未接続である。旧local_trainの更新回数・モデル順・抽出・optimizer状態と同じ学習を、既存ownerを借用するruntimeから呼べるようにする。

## Boundary Context
- In scope: 一ラウンドの全保有独立モデルの局所学習と完了した学習量の計数。
- Out of scope: 観測時の学習要求、client factory、サーバ、Rラウンド進行、全体run/golden移植、時間/演算量の全計数、初期学習方式。
- Adjacent expectations: 観測列と同じregistry/store/Randomを外側が渡す。同期中のowner変更はしない。

## Requirements
### Requirement 1: 単一モデルの学習
**Objective:** As an 実験開発者, I want 旧の更新と同値な計算, so that 数値基準を維持できる。
#### Acceptance Criteria
1. When 正常な独立モデルのbatchを受けたとき, the 学習部品 shall 二値BCE/多クラスCrossEntropyの平均損失から一回更新し、更新前損失をfloatで返す。
2. The 学習部品 shall 同じparameter/optimizer参照を保ち、既存Adam状態を継続して更新し、モデルmodeを変えずに旧updateと同じ値/勾配を得る。
3. If モデル構造・parameter/optimizer対応・学習標本・grad contextが不正なら, the 学習部品 shall forward/zero_gradより前に拒否する。

### Requirement 2: ラウンドの更新順
**Objective:** As an 実験開発者, I want 旧のlocal_trainと同じ順, so that 乱数と学習量を維持できる。
#### Acceptance Criteria
1. When ラウンド学習を呼んだとき, the runtime shall 標本storeのモデル順に、保有していてbatch件数以上あるモデルだけを、updates_per_sample×k_steps回ずつ続けて更新し、各回でRandom.sampleに相当する非復元抽出を行う。
2. When 一更新が成功したとき, the runtime shall そのモデルの学習標本数にbatch件数、更新数に1を加え、損失を更新順のtupleで返す。
3. If 予算が0または参加モデルがないなら, the runtime shall 学習/抽出/計数を行わず空tupleを返す。The runtime shall registry・標本列・他のclient状態を変更しない。

### Requirement 3: 検査と接続
**Objective:** As an 実験開発者, I want 不正入力の事前拒否と新実装での実行, so that client組立へ安全に接続できる。
#### Acceptance Criteria
1. If owner型・予算/件数が不正なら, the runtime shall 最初の状態/RNG更新前に拒否する。参加モデルの全保持標本と学習対応の不正も、他モデルの抽出/更新前に拒否する。
2. The runtime shall 非保有/件数不足モデルのpayloadを学習に使わず、観測処理やサーバ同期を発行しない。
3. The 学習部品とruntime shall 旧実装・実験scriptへ依存せず、新実装だけのfresh smokeから実学習できる。
