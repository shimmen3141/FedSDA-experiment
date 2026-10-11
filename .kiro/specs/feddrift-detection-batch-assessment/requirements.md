# Requirements Document

## Project Description (Input)
研究者がFedSDAとFedDriftを新実装で比較するため、固定旧実装にしかないFedDriftの検出バッチの判定を移植する。全保有モデルの標本別有界損失から、保有順で最小平均損失のモデルを選び、前回の最小平均損失に対する厳密な閾値超過を判定する。数値の一致が繊細な箇所を小さいspecへ切り出す共通引継ぎ手順に従う。モデル生成・割当・学習・サーバ同期は後続のspecが扱う。

## Requirements
### Requirement 1: 検出バッチの損失評価とモデル選択

**Objective:** 研究者として、旧FedDriftと同じ最小損失のモデルを選び、比較実験の判定を維持したい。

#### Acceptance Criteria
1. When 非空の検出バッチを評価するとき, the FedDrift判定 shall 保有順で全モデルを1回ずつ評価し、標本別有界損失の平均を、旧実装と同じ数値で返す。
2. When 最小平均損失が同率になるとき, the FedDrift判定 shall 保有順で最初のモデルを選ぶ。
3. The FedDrift判定 shall 選んだモデルID、最小平均損失、保有順のモデル別平均損失、評価したバッチの標本数を返す。

### Requirement 2: 前回の最小平均損失との比較

**Objective:** 研究者として、閾値境界を含む検出の挙動を旧実装から変えずに移植したい。

#### Acceptance Criteria
1. While 前回の最小平均損失が無いとき, the FedDrift判定 shall ドリフトを検出しない。
2. When 今回の最小平均損失が「前回の値＋非負の閾値」を厳密に超えるとき, the FedDrift判定 shall ドリフトを検出し、等しい場合または小さい場合は検出しない。
3. When 終端の部分バッチを評価するとき, the FedDrift判定 shall 通常のバッチと同じ規則を、渡された標本の全部に適用する。

### Requirement 3: 状態と依存の境界

**Objective:** 研究者として、判定を状態更新から分け、無効な入力で途中までモデルを評価しないようにしたい。

#### Acceptance Criteria
1. If モデルの列が空・IDが重複・入力やラベルが不正・前回値または閾値が不正であるとき, the FedDrift判定 shall 全モデルの順伝播より前に入力を拒否する。
2. The FedDrift判定 shall モデルのパラメータ・既存の勾配・学習モード・乱数の状態を変更せず、モデルの生成・切替・学習・データ割当を行わない。
3. The FedDrift判定 shall 旧実装とtest moduleを読み込まずに新実装だけで実行でき、既存のFedSDAの結果を変えない。

## Boundary Context

- 対象は、検出バッチの数値評価と比較判断の全体。真の概念IDを受け取らない。
- バッファへの蓄積、予測記録、判定後の新規モデル登録と割当、前回値の更新、通信と計数、全体run・掃引・保存は対象外。
- 後続のclient処理がモデル保有順を渡し、返された判定を使って、旧と同じ順で状態を更新する。全体runの旧対照とFedDriftのgoldenは後続の完了条件であり、本specの完了で代替しない。
