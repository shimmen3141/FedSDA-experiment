# Requirements Document

## Introduction
研究者がFedSDAの構成間で通信量を公平に比較するため、独立モデル構成のクロス評価で同じ候補を同じclientへ複数回送ったと数えるLEGACY-018を修正する。2026-10-11に合意済みの重複除外を、全構成goldenの作成前に適用する。実施順は主担当の判断であり、新たなユーザー決定として記録しない。

## Boundary Context
- 対象: 新FedSDAのクロス評価の下りモデル・パラメータ値・バイトの計数と照合。
- 対象外: 固定旧748c3aa、既存golden、他LEGACY、FedDriftのモデルを送らないクロス評価、学習・検出方式、旧実装削除。

## Requirements
### Requirement 1: 1回のクロス評価内の重複除外
**Objective:** As a 研究者, I want 構造によらず同じ転送規則, so that 通信量を構成間で比較できる。
#### Acceptance Criteria
1. When 同じ候補を同じclientの複数の保有モデルで評価する, the FedSDA system shall その候補の概念固有部（共有部が無ければ全パラメータ）の下りモデル数・値数・バイト数を1回だけ数える。
2. When 候補またはclientが違う, the FedSDA system shall それぞれの転送を数える。
3. When 次のクロス評価を実行する, the FedSDA system shall 前回の重複除外の履歴を引き継がずに数える。
4. Where 共有部がある, the FedSDA system shall 共有部をclientごとに1回、概念固有部を候補とclientごとに1回数える既存規則を維持する。
5. When 評価先clientが無い, the FedSDA system shall その組の転送を増やさない。

### Requirement 2: 評価の動作と他の計数の維持
**Objective:** As a 研究者, I want 修正の影響を下り転送だけに限定, so that 移植の一致を維持できる。
#### Acceptance Criteria
1. The FedSDA system shall 評価するclientの抽出・乱数消費・評価順・損失・正誤の統計・診断の記録・学習状態を維持する。
2. The FedSDA system shall 組ごとの依頼・返信メッセージ数、上り転送、登録・集約・配布の通信量を維持する。
3. The FedSDA system shall 不正入力の更新前拒否と途中の評価失敗で済んだ記録が残る契約を維持する。

### Requirement 3: 修正後の全体runの照合
**Objective:** As a 研究者, I want 既知の差だけを説明する照合, so that 比較を無条件に除外しない。
#### Acceptance Criteria
1. When 独立モデル構成を実旧と照合する, the validation shall 実旧の転送イベントから重複分を独立に算出し、下りの3計数が旧から重複分を引いた値と一致することを確かめる。
2. The validation shall 独立モデル構成でも通信量以外の既存指標・離散列・モデル状態の新旧一致を維持する。
3. The validation shall 重複のある条件と無い条件、合成データとMNIST、保存された通信量、共有部を持つ構成とFedDriftの回帰を検証する。
