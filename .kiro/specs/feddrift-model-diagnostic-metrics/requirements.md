# Requirements Document

## Project Description (Input)
FedSDAとFedDriftを比較する研究者向け。FedDriftの全体run・基本指標・計算量は接続済みだが、モデル数の系列・学習量の断片化・クラスタリングと真の概念の一致が指標へ未接続である。既存の記録と実同期から、FedSDAと同じ定義の適用可能なモデル診断を得られるようにする。

## Boundary Context
モデル診断の取得から全体run指標への接続までを対象とする。候補検証・予測混合・再較正・予測相補性の新規評価・保存/CLIは対象外。共通評価算式と固定旧748c3aa、golden、数値/乱数/通信/基本指標を維持する。

## Requirements
### Requirement 1: 同期後のモデル数
1. When 通常時刻の準備とR回の学習・集約・配布が成功したとき, the FedDrift server shall その時刻の最後の実globalモデル数とclient順の実保有モデル数を1件記録する。一時モデルを保有数へ含める。
2. If 同期が失敗したとき, the record shall その時刻を追加しない。初期準備、準備のみ、単独集約、終端は通常同期の系列へ含めない。
3. The diagnostic shall 同期後のglobal数の系列と平均・最大・延べ合計を返す。同期がなければ空系列と0を返す。
### Requirement 2: モデル別の学習量とクラスタリング
1. The diagnostic shall 終了時の実保有モデルについてclient順/保有順に、保持学習標本数・学習に使った延べ標本数・更新step数を返す。一時IDを含め、欠落計数は0とする。
2. The diagnostic shall 同じmodel IDを全clientで足してから、共通ValueDistributionSummaryで3種類の学習量を要約する。統合と正式ID確認後のownerの計数を使う。
3. The diagnostic shall 真の概念の一致が既知のモデル対だけで、共通のクラスタ割当適合率/再現率/F1と、損失距離/parameter距離の同概念AUCを返す。未知を除外し、AUC未定義はNoneとする。
### Requirement 3: 全体runへの接続と検証
1. When 全体run指標を導出したとき, the result shall モデル診断を含み、通常同期の記録位置/件数とrun結果の同期数/client数の対応を検査する。
2. If 記録の型・整数・件数・順序が不正なとき, the derivation shall 状態と乱数を変えずに拒否する。診断は推論/学習/クラスタリングの判断へ戻さない。
3. The verification shall 実旧の学習量/同期model数/クラスタ診断との対応、0同期・R・統合・一時ID・破損記録、共用fresh/exact依存境界、全pytest/品質/固定旧goldenの無干渉を確かめる。
