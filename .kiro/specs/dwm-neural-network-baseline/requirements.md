# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedHASPの1クライアント版を検討している研究者（ユーザー）。
- 今の状況: 1クライアントの予備実験で、FedHASPはRiverのストリーム学習の手法（ARF、SRP、Leveraging Bagging、Hoeffding Adaptive Tree、自前のDWM）を上回ったが、差には基本の学習器の違い（事前学習した神経網と木）が混ざる。Riverの手法にFedHASPの事前学習の表現を固定で与えても、表現を学習し続ける単一の神経網（構成`fedavg`）より低く、ドリフトへの対応の仕組みだけの差が分からない（`docs/research/fedhasp-single-client-pilot-v16.md`）。
- 何を変えるか: 単一モデルFedAvg（構成`fedavg`）と同じ事前学習済みの神経網を専門家（エキスパート）にしたDynamic Weighted Majority（Kolter & Maloof, JMLR 8, 2007の図1）を、製品の構成`dwm-neural-network`として追加する。各clientが独立にDWMを動かし、サーバは集約しない（通信なし）。評価は予測→学習の順で、評価に使う予測は学習の前の重みで行い、学習の手順（重みの減衰・正規化・削除・追加・全専門家の学習）は図1のとおりにする。β=0.5、θ=0.01、p=50（論文の大きい・雑音のある問題の設定）。新しい専門家は事前学習の直後の初期モデルの複製で、作られた後の標本だけを保持し、`fedavg`と同じく32件のミニバッチで標本ごとに1回更新する。既定・既存の構成の値は変えない。作業場所は`.worktrees/single-client`（ブランチ`research/single-client`）。

## Introduction
FedHASPと同じ学習器（`fedavg`と同じ事前学習済みの神経網と学習の手順）を使う比較手法として、DWMを新しい構成`dwm-neural-network`で実行・保存・掃引できるようにする。これにより、`fedavg`（専門家1つ、ドリフトへの対応なし）、`dwm-neural-network`（DWMの専門家の追加・削除と重み付き投票）、FedHASPを、同じ入力の系列・同じ学習器で比べられる。

## Boundary Context
- **In scope**: 構成`dwm-neural-network`の、clientごとのDWMの逐次の予測と学習、全体run、共通の指標、成果物の保存と読込み、既存の掃引コマンドへの接続、構成の一覧の文書。
- **Out of scope**: 既存の構成（FedHASP系、FedDrift系、`fedavg`ほか）の設定・値・runの識別子・goldenの変更。DWMの専門家の連合学習での集約。β・θ・pの掃引軸。DWM以外の既存手法（AddExp、Learn++.NSE、再来するコンセプトの専用の手法）。Riverの研究ツールの変更。
- **Adjacent expectations**: 系列の生成・事前学習・学習器の部品（分類器、optimizer、ミニバッチの更新）は既存の製品のものを使い、`fedavg`と同じ条件で動く。

## Requirements

### Requirement 1: 専門家の学習器
**Objective:** As a 研究者, I want DWMの各専門家を`fedavg`と同じ学習器にする, so that DWMの仕組みだけの効果を`fedavg`・FedHASPと比べられる

#### Acceptance Criteria
1. The system shall 各専門家に、`fedavg`と同じ構造（共有部を持たない独立の神経網）、同じ事前学習の条件、同じoptimizerの設定、同じミニバッチの件数（32）、同じ標本ごとの更新の回数（既定1）を使い、図1のとおり毎標本で学習する（`fedavg`の局所更新をまとめる間隔に当たる設定は持たない）。
2. When runを準備するとき, the system shall `fedavg`と同じく事前学習を1回だけ行い、その直後の分類器と、`fedavg`が使うのと同じoptimizerの状態を、各clientの最初の専門家（重み1）として複製する。
3. When 新しい専門家を作るとき, the system shall 事前学習の直後の分類器とoptimizerの状態の複製を、重み1で足す（学習済みの別の専門家の状態は使わない）。
4. The system shall 各専門家に、その専門家が作られた時点の標本（作られるきっかけになった標本を含む）以降の標本だけを保持させ、保持した標本の数がミニバッチの件数に達するまでは、その専門家を更新しない。
5. Where DWMが専門家を一度も足さず消さない条件（周期pが処理する標本の数より大きい）で動くとき, the system shall 同じseedの`fedavg`のclientと、予測の列・学習後のパラメータ・optimizerの状態・乱数の消費が一致する。

### Requirement 2: 標本ごとの予測と学習（Kolter & Maloof 2007の図1）
**Objective:** As a 研究者, I want 図1のDWMを、予測→学習の評価でラベルを漏らさずに動かす, so that 他の構成と同じ定義の精度で比べられる

#### Acceptance Criteria
1. When 標本が到着するとき, the system shall 学習の前に、各専門家の予測クラス（2クラスは確率0.5より大きければ1、それ以外は最大のスコアのクラス）と、その時点の重みによる重み付き投票（各専門家が自分の予測クラスへ重みを足す）で全体の予測を決めて記録し、この予測を精度の評価に使う。投票の最大が同点のときは、クラスの番号が最も小さいクラスにする。
2. When 標本の番号iが周期pの倍数のとき（iは1から数える）, the system shall 予測を誤った専門家の重みにβを掛け、その重み（減衰の直後、正規化と削除の前）で投票し直した全体の予測を求め、全重みを最大の重みが1になるよう同じ比で割り、重みがθ未満の専門家を消し、投票し直した全体の予測が誤っていれば重み1の新しい専門家を足す（この順）。
3. When 標本の番号iが周期pの倍数でないとき, the system shall 重みを変えず、専門家を足しも消しもしない。
4. When 重みと専門家の更新が済んだとき, the system shall すべての専門家に標本を保持させ、Requirement 1の条件でそれぞれ1回更新する。
5. The system shall β=0.5、θ=0.01、p=50を固定値として設定に持ち、runの記録に残す。
6. The system shall 真のコンセプトのIDを記録と評価にだけ使い、予測・重み・専門家の追加と削除・学習には使わない。
7. If 不正な標本・標本の位置・設定が与えられたとき, the system shall 状態の更新と乱数の消費より前に拒否する。

### Requirement 3: clientとサーバ
**Objective:** As a 研究者, I want 各clientが独立にDWMを動かし、サーバが何も共有しない, so that 1クライアントの比較手法として使え、複数clientでもclientごとの独立な実行になる

#### Acceptance Criteria
1. The system shall clientごとに専門家と重みを独立に持ち、他のclientの専門家・重み・標本を使わない。
2. When 同期の時点が来たとき, the system shall パラメータを集約・配布せず、通信量を0として記録する。
3. The system shall 集約の間隔を、処理する標本の区切り（完全な区間だけを処理する）にだけ使い、DWMの予測と学習には使わない。

### Requirement 4: 記録・指標・成果物
**Objective:** As a 研究者, I want 他の構成と同じ定義の指標と、DWMの動きを追える記録を保存する, so that 既存の集計に載せ、専門家の増減も確かめられる

#### Acceptance Criteria
1. When runを評価するとき, the system shall 他の構成と同じ定義で、予測精度、定常期の精度、切替の検出の指標（DWMはモデルの切替を持たないので、`fedavg`と同じく検出0件として計算する）、通信量、最後のパラメータの量、モデルの計算の計数とその正規化、処理した標本の数を導出する。
2. The system shall 標本ごとに、全体の予測、観測のクラス、真のコンセプト、正誤、予測の時点の専門家の数を記録し、専門家ごとに、作られた標本の位置（初期の専門家は-1）と消された標本の位置（最後まで残ったものは-1）を記録する。
3. When 成果物を保存するとき, the system shall 完全な設定・指標・環境のJSONと、記録の名前付きの配列のNPZを保存し、既存のファイルがあれば上書きせず、pickleを使わずに読み戻せるようにする。
4. The system shall 最後のモデルの数として、全clientの最後に残った専門家の数の合計を記録する。

### Requirement 5: 掃引への接続と既存の構成の維持
**Objective:** As a 研究者, I want 既存の掃引コマンドで`dwm-neural-network`を選べる, so that uraniumで他の構成と同じ手順で実行・集計できる

#### Acceptance Criteria
1. When 構成`dwm-neural-network`を選ぶとき, the system shall 既存の掃引コマンドの計画の表示、並列の実行、保存、runの一覧のCSV、再開と競合の拒否まで、既存の構成と同じ手順で動く。
2. The system shall FedDrift系・FedHASP系の専用の掃引軸（閾値、FIFOの容量など）で`dwm-neural-network`のrunを増やさない。
3. The system shall 既存の全構成の名前・順・設定・runの識別子・保存の形式・goldenを変えない。
4. The system shall 構成の一覧の文書に、`dwm-neural-network`の目的と、`fedavg`・論文のDWMとの違い（評価の予測を学習の前の重みで行うこと、専門家が神経網であること）を書く。
