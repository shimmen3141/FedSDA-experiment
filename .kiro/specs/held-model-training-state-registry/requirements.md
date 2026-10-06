# Requirements Document

## Introduction
研究実装の開発者が、採用候補を正式登録する前提となるモデル別保有一覧を新APIで扱えるようにする。現在の新実装にはNN、個別optimizer管理器、借用学習記録があるが、それらをモデルIDで保持して現在の学習記録を提供する管理がない。
モデルと個別optimizer管理器を組として保持し、旧モデル一覧の初出順・置換順を維持する。

## Boundary Context
- In scope: ID指定の保有状態登録/置換、取得、順序付き参照一覧、現在optimizerによる借用学習記録。
- Out of scope: モデル生成/clone、共有値反映/接続/reset、ID採番/サーバ対応/削除、標本/統計/送信状態、候補判定、学習実行、client/全体run。
- Adjacent expectations: 上位が準備済みモデルと対応する個別optimizer管理器を渡す。外部のモデル構造改ざん・並行操作は通常契約外。既存の共有接続は概念固有parameterを交換しない。
- モデルIDは負の一時IDと非負のサーバIDを含むPython組み込みintの完全一致とし、範囲による制限は付けない。真偽値・int派生型・他ライブラリの整数型は受理しない。IDの採番や意味の変換は上位が行う。

## Requirements

### Requirement 1: ID別の登録と参照
**Objective:** 開発者として、モデルと個別optimizer管理器の対応と初出順を一箇所で扱いたい。

#### Acceptance Criteria
1. The 保有状態管理 shall 空の保有一覧から開始する。
2. When 正しいモデルIDと対応するモデル/個別optimizer管理器が登録される, the 保有状態管理 shall 負/非負IDを同等に扱って渡された実体への参照を保持し、未登録IDを末尾へ追加する。
3. When 既存IDへ新しい対応組が登録される, the 保有状態管理 shall そのIDの初出位置を維持したまま対応組を置換し、古い取得済み記録を変更しない。
4. When 登録済みIDの状態または順序付き一覧が取得される, the 保有状態管理 shall 登録された対応組を返し、取得によって登録を増やさず、一覧構造を後続登録/置換から独立させる。
5. If IDの型、モデル/管理器の型、個別parameterとの対応が不正である, then the 保有状態管理 shall 登録前に明示的に拒否し、既存一覧と渡された学習状態を変更しない。対応とはモデルの概念固有parameter列と管理器の現在optimizerが同じparameterを同順序で対象とすることを指す。
6. If 未登録IDの取得が要求される, then the 保有状態管理 shall 未登録を明示的に報告し、空の状態を自動生成しない。

### Requirement 2: 現在optimizerによる学習記録
**Objective:** 開発者として、個別optimizerのreset後も正しい現在参照を学習へ渡したい。

#### Acceptance Criteria
1. When 学習記録の一覧が要求される, the 保有状態管理 shall 登録順のモデルID・モデル・その時点の個別optimizerを借用する記録を返す。
2. When 登録済み個別optimizer管理器が外側でresetされる, the 保有状態管理 shall 次に取得する学習記録へ現在optimizerを反映し、既に取得済みの借用記録は古いoptimizer参照を保持する。
3. The 保有状態管理 shall 登録/取得/記録生成によりモデル値・grad・optimizer学習state・乱数を変更せず、共有optimizerを所有/生成しない。

### Requirement 3: 独立した移植と検証
**Objective:** 開発者として、保有一覧と既存の共同学習を接続し、旧実装との対照を残したい。

#### Acceptance Criteria
1. When 新保有一覧の学習記録が既存の共同学習へ明示接続される, the 移植検証 shall 同一条件の旧保有一覧と初出/置換順・学習結果・optimizer state・乱数消費を一致させる。
2. The 保有状態管理 shall 旧実装、グローバル設定、候補採否、標本/統計/送信状態、学習実行へ依存せず、旧productionとgoldenを変更しない。
