# Requirements Document

## Project Description (Input)
- 誰が困っているか: 1クライアント版の研究で、FedHASPの効果のうち連合学習でだけ現れるものと、その理由を切り分けたい研究者。
- 今の状況: 連合学習では、柔らかい学習（構成`fedsda-soft-training`）が固い学習（構成`fedsda-training-mass-diagnostics`）より精度が低い（間隔50で0.63pp）。1クライアントでは差がほぼ0。学習量の構成で測った純度の低下は設定によらず同じ程度で、集約で混ざり方が広がる様子も見えない。残る説明は、(a) 集約を通じて、そのモデルに固く割り当てられていないclientの柔らかい学習の更新がモデルへ混ざる道と、手元の柔らかい学習が、配られた多くのモデルへ分散する道である。今の柔らかい学習は、サーバの集約の重みと参加の判定に、clientの学習量（責任度qの合計）を使うので、2つの道を分けられない。
- 何を変えるか: 局所の学習は`fedsda-soft-training`と同じ柔らかい学習のまま、集約の重みと参加の判定だけを、固い割り当ての標本の件数（固い学習が使う値）に替えた構成を、製品の構成として足す。既定と既存の構成の挙動・runの識別子は変えない。結果を既存の固い学習・柔らかい学習の200 run（連合学習、10データ×間隔50・500×seed 0〜9）と対にして、精度が固い学習の水準へ戻るか（集約を通る道）、戻らないか（手元を通る道）を判断する。作業場所は`.worktrees/single-client`（ブランチ`research/single-client`）。

## Introduction
柔らかい学習の構成`fedsda-soft-training`から、サーバの集約の重みと参加の判定だけを固い割り当ての標本の件数に替えた構成`fedsda-soft-training-assignment-count-aggregation-weights`を足し、実行・保存・掃引できるようにする。これにより、固い学習・柔らかい学習・この構成の3つを同じ条件で比べ、柔らかい学習の精度の低下が集約を通って起きるかを直接確かめられる。

## Boundary Context
- **In scope**: 新しい構成の設定、集約の重みと参加の判定の切り替え、成果物の保存（既存の柔らかい学習と同じ記録）、既存の掃引コマンドへの接続、構成の一覧の文書。
- **Out of scope**: 既存の構成（固い学習・柔らかい学習を含む）の設定・値・runの識別子・保存の形式・goldenの変更。局所の学習・予測・検出・責任度の計算の変更。uraniumでの本規模の実行と、結果の集計の研究ツール（別に主担当が行う）。
- **Adjacent expectations**: 柔らかい学習の局所の処理（責任度、重み付きの抽出による学習、学習量の記録）と、固い割り当ての標本の保持は、既存の製品のものをそのまま使う。柔らかい学習でも、固い割り当ての標本は、固い学習と同じ規則で保持されている。

## Requirements

### Requirement 1: 局所の処理は柔らかい学習と同じ
**Objective:** As a 研究者, I want 新しい構成の局所の処理を`fedsda-soft-training`と同じにする, so that 精度の差を集約の重みの違いだけに帰せる

#### Acceptance Criteria
1. The system shall 新しい構成で、予測、柔らかい学習（責任度と重み付きの抽出による学習）、変化の検出、モデルの採用、学習量のコンセプト別の記録を、`fedsda-soft-training`と同じ設定で行う。
2. When 同じseed・データ・間隔で、新しい構成と`fedsda-soft-training`を最初の集約の前まで動かすとき, the system shall 両者の予測の列・パラメータ・乱数の状態を一致させる。
3. Where 各clientが保有するモデルが1つだけである（全標本の責任度がそのモデルに1で、学習量が固い割り当ての件数と等しい）とき, the system shall 新しい構成の集約の結果（グローバルモデルのパラメータ・損失の統計・上りの通信量）を`fedsda-soft-training`と厳密に（誤差の許容なしで）一致させる。

### Requirement 2: 集約の重みと参加の判定
**Objective:** As a 研究者, I want 集約で、各clientのモデルの重みを固い割り当ての標本の件数にする, so that そのモデルに固く割り当てられていないclientの柔らかい学習の更新を集約に混ぜない

#### Acceptance Criteria
1. When 新しい構成で集約するとき, the system shall 各clientの各モデルの重みを、そのclientがそのモデルへ固く割り当てた標本の件数（固い学習の構成が集約に使うのと同じ値）とし、学習量（責任度の合計）は使わない。
2. When clientがモデルを保有していて、そのモデルの固い割り当ての件数が0のとき, the system shall そのclientのそのモデルを集約に参加させず（概念固有部・損失の統計の平均に入れず、上りの通信にも数えない）、そのclientがそのモデルを柔らかい学習で更新していても同じに扱う。
3. When 新しい構成で集約するとき, the system shall 共有部の重みを、そのclientの参加するモデルの固い割り当ての件数の合計にする。
4. If どのclientもあるモデルの固い割り当ての件数を持たないとき, then the system shall そのモデルについて、既存の集約と同じ扱い（既存のグローバルモデルの概念固有部と統計を保つ）にする。
5. The system shall 集約の重みと参加の判定以外の集約の手順（対象のモデル、平均の順序、通信量・パラメータ・統計の反映の順）を変えない。

### Requirement 3: 設定と記録
**Objective:** As a 研究者, I want 集約の重みの元を設定として持ち、runの記録に残す, so that 結果から、どの重みで集約したかを確かめられる

#### Acceptance Criteria
1. The system shall 集約の重みの元を、学習の割当方式による今の扱い（既定）と、固い割り当ての件数の2つから選ぶ設定として持ち、既定以外の値をrunの設定の記録に残す。
2. If 固い割り当ての件数を選び、学習の割当方式が柔らかい学習でないとき, then the system shall runの準備の前に設定を拒否する。
3. The system shall 新しい構成のrunに、`fedsda-soft-training`と同じ記録（学習量のコンセプト別の記録を含む）と指標を保存する。

### Requirement 4: 掃引への接続と既存の構成の維持
**Objective:** As a 研究者, I want 既存の掃引コマンドで新しい構成を選べる, so that uraniumで既存の固い学習・柔らかい学習の結果と同じ手順で実行し、対にして集計できる

#### Acceptance Criteria
1. When 新しい構成を選ぶとき, the system shall 既存の掃引コマンドの計画の表示、並列の実行、保存、runの一覧のCSV、再開と競合の拒否まで、既存の構成と同じ手順で動く。
2. The system shall 既存の全構成の名前・順・設定・runの識別子・保存の形式・goldenを変えない。
3. The system shall 構成の一覧の文書に、新しい構成の目的と、`fedsda-soft-training`との違い（集約の重みと参加の判定だけ）を書く。
