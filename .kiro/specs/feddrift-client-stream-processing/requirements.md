# Requirements Document

## Introduction
FedSDAとFedDriftを比較する実験の開発者は、FedDriftの一バッチ判定を移植済みだが、旧clientの観測列の進行へまだ接続できない。切替前の予測、時刻をまたぐ蓄積、検出・切替位置と部分バッチの終端処理を、新実装のruntimeで旧と同じ順に実行できるようにする。

## Boundary Context
- **In scope**: 0から連続した位置を持つ単一標本の時刻ブロック、予測記録、検出バッファと完了結果、終端flush。
- **Out of scope**: 局所学習、初期学習方式の追加、サーバ同期、通信・時間・演算量計数、client factory、全体run/golden。
- **Adjacent expectations**: 検出バッチ処理のownerは外側でも共有する。局所学習は観測の副作用にせず後続のラウンド処理で行う。現在モデルのサーバ側再割当は次の予測で読み直す。

## Requirements

### Requirement 1: 観測列と予測
**Objective:** As an 実験開発者, I want 切替前の予測を保存したい, so that 旧のオンライン精度と一致する。
#### Acceptance Criteria
1. When 正常な時刻ブロックを受けたとき, the stream処理 shall 入力位置の順に現在モデルだけで予測し、位置・真の概念ID・使用モデルID・予測クラス・観測クラス・正否を記録する。
2. When 検出と割当が起きる標本を観測したとき, the stream処理 shall その標本の予測を割当前のモデルで行い、次の標本から割当先を使う。
3. The stream処理 shall 観測だけで既存parameter/optimizer・Torch乱数を更新せず、概念IDを判断に使わない（検出時の新規モデル構築によるTorch乱数消費は既存バッチ処理の契約）。

### Requirement 2: 蓄積と位置
**Objective:** As an 実験開発者, I want 時刻境界を越えて検出したい, so that 検出間隔と時刻規模を独立にできる。
#### Acceptance Criteria
1. When 検出件数に達したとき, the stream処理 shall そのバッファだけを既存バッチ処理へ渡して消去し、複数回を含め一度でも完了した時刻ブロックにはtrueを返す。
2. While 検出件数に満たない間, the stream処理 shall バッファを次のブロックへ保ち、空ブロックはfalseを返して状態・乱数を変えない。
3. When 一バッチの処理が完了したとき, the stream処理 shall 完了結果を順に保持し、検出先頭と切替末尾の位置をその結果から区別できるようにし、通常処理では発火標本の予測記録にだけ結果を付ける。

### Requirement 3: 終端と検査
**Objective:** As an 実験開発者, I want 部分バッチと異常入力を扱いたい, so that 再開と全体接続を検証できる。
#### Acceptance Criteria
1. When 終端flushを呼んだとき, the stream処理 shall 残った標本を一度だけ処理し、完了結果と位置を保持してバッファを空にするが、標本数・既存予測記録を変えない。残りがなければNoneを返して何も変えない。
2. If 設定・入力型・非連続位置・概念ID型・標本形状/値・保有モデルが不正なら, the stream処理 shall 予測forwardと最初の状態更新より前に拒否する。時刻ブロック内の後方の不正標本も全ブロック更新前に拒否する。
3. The stream処理 shall 新実装だけで実行でき、runtimeから既存learningと検出バッチ処理へ依存し、旧clientや実験スクリプトへ依存しない。
