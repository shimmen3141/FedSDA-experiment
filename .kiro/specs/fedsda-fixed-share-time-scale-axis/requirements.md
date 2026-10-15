# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの精度を、FedDrift・FedAvgと比べている研究者（ユーザー）。
- 今の状況: 帰属の保留（FIFO）の容量と、Fixed-Shareの時間尺度T（モデルの重みを分け直す速さ。混ぜる予測の重みの分け合いの割合1/Tも兼ねる）は、設定の検査で同じ値を求められ、掃引の軸`--fedsda-pending-assignment-fifo-capacities`は両方を同じ値に替える。手元の診断（2026-10-16、研究バックログの8）で、3つの役割を分けると、容量はデータによって得失が割れ（circle2・50で上がり、sine2・200で下がる）、Tと混ぜる分け合いは60で揃って上がった。2つを別々に指定できないので、「容量30・T60」を本規模で試せない（IMPROVE-015）。
- 何を変えるか: 2つの値の同値の検査を外し、Tを別の掃引の軸で並べられるようにする。容量の軸は容量だけを替える。既定の構成の動作と値は変えない。

## Introduction
2026-10-16の会話で、ユーザーは次を決めた。容量の軸は容量だけを替え、新しい軸でTを替える（指定しない側は構成の値のまま）。混ぜる重みの分け合いの割合は、Tとの連動（1/T）のまま、別のハイパーパラメータにしない（どちらも、最良の候補が入れ替わる見込みの間隔Tを表す、同じ概念のため）。

## Boundary Context
- 対象: 2か所の同値の検査（clientの設定の組の検査と、全体の設定の検査）、掃引の計画（新しい軸と、容量の軸の意味）、コマンドの引数、一覧の表の列、掃引の記録の形式の版、文書、IMPROVE-015の状態。
- 対象外: 帰属の保留・Fixed-Share・混ぜる予測の処理、既定の値（どちらも30）、FedDrift・FedAvg、run設定の保存の形。
- 仮定: FedSDAの2つの軸（容量とT）を両方指定したときは、全部の組を作る。

## Requirements
### Requirement 1: 2つの値を別々に設定できる
**Objective:** As a 研究者, I want 容量とTを違う値にしたrun設定を作れること, so that 2つの効果を分けて比べられる。
#### Acceptance Criteria
1. The run settings validation shall `pending_assignment_buffer_capacity_samples`と`fixed_share_weight_redistribution_time_scale_samples`が違う値でも、ほかの検査を満たせば受理する（clientの設定の組の検査と、全体の設定の検査の両方）。
2. The run settings validation shall 2つの値それぞれの、今の個別の検査（型、Tの下限2、警報を待つ扱いでの容量の下限＝変化区間の最小件数）を変えずに保つ。
3. The run settings shall 既定の構成の2つの値（どちらも30）と、既存の構成のrun設定・runの識別子を変えない。

### Requirement 2: Tを軸として並べられる
**Objective:** As a 研究者, I want 掃引のコマンドで、Tを複数並べて実行できること, so that 研究室サーバで、1回の掃引で比べられる。
#### Acceptance Criteria
1. The sweep command shall `--fedsda-fixed-share-time-scales`で、正の整数の一覧を受け、FedSDAの構成の各runを、Tの値ごとに作る。各runの`fixed_share_weight_redistribution_time_scale_samples`だけがその値になる（容量ほかの設定は変えない）。
2. The sweep command shall `--fedsda-pending-assignment-fifo-capacities`では、各runの`pending_assignment_buffer_capacity_samples`だけを替える（Tは変えない）。
3. When 容量の軸とTの軸の両方を指定したとき, the sweep command shall 全部の組（容量の順の中で、Tの順）を作る。
4. If 値が重複する・正のbuiltin intでない、またはFedSDAの構成を含まない計画, the sweep command shall 引数の誤り（終了コード2）として拒否する。作ったrun設定の検査が拒否する値（例: T＝1）も、終了コード2になる。
5. When Tの軸を指定しないとき, the sweep command shall Tを構成の値のままにする。既定の値（30）を指定したrunは、指定しないrunと同じ設定・同じ識別子になる。

### Requirement 3: 一覧の表と記録
**Objective:** As a 研究者, I want 結果の表で、各runのTが分かること, so that 分析で比べられる。
#### Acceptance Criteria
1. The run summary table shall FedSDAの構成の行に、列`fixed_share_time_scale`（そのrunのT）を持つ。値の無い要約では空欄にし、行を外さない。
2. The sweep record shall 計画に新しい項目を持ち、掃引の記録の形式の版を7へ上げる（動作の基準版とは別の番号）。

### Requirement 4: これまでの照合を保つ
**Objective:** As a 開発者, I want 既存のgoldenと照合が値を変えずに通ること, so that 既定の動作が変わっていないと確かめられる。
#### Acceptance Criteria
1. The validation shall 基準版1〜5のgoldenを、値を変えずに通す。既存のtestは、同値の検査を確かめるtest（受理へ変える）、容量の軸がTも替えることを期待するtest（Tを変えないへ変える）、掃引の記録の形式の版（6→7）と記録の計画の新しい項目、コマンドの引数の一覧の期待だけを変え、ほかの期待は変えずに通す。

### Requirement 5: 文書
**Objective:** As a 研究者, I want 文書が新しい軸と、容量の軸の意味の変化を説明すること, so that サーバのコマンドを正しく書ける。
#### Acceptance Criteria
1. The documentation shall 実験の入口の文書に、Tの軸、Tが混ぜる重みの分け合い（1/T）も決めること、容量の軸がTを替えなくなったこと（記録の形式の版6以前の掃引で容量を替えたrunは、Tも同じ値だった）、目安（研究バックログの8の分ける診断）、2つの軸の組合せの順、記録の形式の版7を書く。IMPROVE-015の状態を実装済みにする。
