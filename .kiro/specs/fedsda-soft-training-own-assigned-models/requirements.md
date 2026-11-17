# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedHASPの論文で、固い学習が連合学習でだけ効く理由を示したい研究者。
- 今の状況: 連合学習では、柔らかい学習（構成`fedsda-soft-training`）が固い学習より精度が低い（全体0.46pp、10/10 seed）。1クライアントでは差がほぼ0。集約の重みを固い割り当ての件数にした介入（構成`fedsda-soft-training-assignment-count-aggregation-weights`）では精度が戻らず、責任度が分かれる量も1クライアントと同じ程度で、どちらも原因ではなかった。残る有力な候補は、柔らかい学習が、他のclientが作って配られてきたモデル（そのclientが自分で固く割り当てたことのないモデル）まで、手元の標本で毎回フルサイズのbatchで学習してしまい、そのモデルを使う前に崩すこと。最初の区間（配られたモデルだけが余分にある）でも連合学習だけ差が出ること、差が最大なのが「自分は初めて・他clientが経験済み」の区間であることと合う。
- 何を変えるか: 柔らかい学習の責任度の対象を、そのclientが自分で固く割り当てたことのあるモデル（と、いま固く割り当てる先のモデル）だけに制限した構成を、製品の構成として足す。配られただけのモデルは、予測とモデルの選択には今どおり使い、柔らかい学習でだけ更新しない。既定と既存の構成の挙動・runの識別子は変えない。結果を既存の固い学習・柔らかい学習の200 run（連合学習、10データ×間隔50・500×seed 0〜9）と対にして、差が消えるかを見る。作業場所は`.worktrees/single-client`（ブランチ`research/single-client`）。

## Introduction
柔らかい学習の構成`fedsda-soft-training`から、責任度を求める対象のモデルだけを「そのclientが自分で固く割り当てた標本を持つモデル」に制限した構成`fedsda-soft-training-own-assigned-models`を足し、実行・保存・掃引できるようにする。これにより、柔らかい学習の精度の低下が、配られたモデルを手元で学習することから来るかを直接確かめられる。

## Boundary Context
- **In scope**: 新しい構成の設定、責任度の対象の制限、成果物の保存（既存の柔らかい学習と同じ記録）、既存の掃引コマンドへの接続、構成の一覧の文書。
- **Out of scope**: 既存の構成（固い学習・柔らかい学習・集約の重みを替えた構成を含む）の設定・値・runの識別子・保存の形式・goldenの変更。予測・変化の検出・モデルの選択と採用・集約の変更。uraniumでの本規模の実行と集計（主担当が別に行う）。
- **Adjacent expectations**: 柔らかい学習の局所の処理（責任度の式、重み付きの抽出による学習、学習量の記録、集約の重み）と、固い割り当ての標本の保持は、既存の製品のものを使う。固い割り当ての標本は、柔らかい学習でも固い学習と同じ規則で保持されている（spec `fedsda-soft-training-assignment-count-aggregation-weights`で確認済み）。

## Requirements

### Requirement 1: 責任度の対象の制限
**Objective:** As a 研究者, I want 柔らかい学習の責任度を、自分で固く割り当てたことのあるモデルだけに分ける, so that 配られただけのモデルを手元の標本で学習しない

#### Acceptance Criteria
1. When 新しい構成でFIFOから放す標本の責任度を求めるとき, the system shall 対象を、その時点でそのclientの固い割り当ての標本を1件以上持つ保有モデルと、その標本をいま固く割り当てる先のモデルに限り、それ以外の保有モデルの責任度を0にする（責任度の辞書に含めない）。
2. The system shall 対象のモデルの責任度を、`fedsda-soft-training`と同じ式（予測用のFixed-Shareの重み×真のラベルの確率を、対象のモデルの中で正規化）で求める。
3. The system shall 責任度を求めるときのFixed-Shareの重みの同期（保有モデルの集合の更新）を、`fedsda-soft-training`と同じく全保有モデルで行い、予測の重みの状態を変えない。
4. Where すべての保有モデルが対象である（全保有モデルが固い割り当ての標本を持つか、いま割り当てる先である）とき, the system shall `fedsda-soft-training`と同じ責任度を返す。
5. When 配られただけのモデルが、モデルの選択で固い割り当ての先になったとき, the system shall そのモデルを、その標本から責任度の対象に加える。

### Requirement 2: 局所のその他の処理は柔らかい学習と同じ
**Objective:** As a 研究者, I want 責任度の対象以外を`fedsda-soft-training`と同じにする, so that 精度の差を対象の制限だけに帰せる

#### Acceptance Criteria
1. The system shall 新しい構成で、予測、変化の検出、モデルの選択と採用、重み付きの抽出による学習、学習量の記録、集約（重みは責任度の合計）を、`fedsda-soft-training`と同じ設定で行う。
2. When 同じseed・データ・間隔で、新しい構成と`fedsda-soft-training`を動かし、どのclientも対象外の保有モデルを持たない間は, the system shall 両者の責任度・予測・パラメータを一致させる。

### Requirement 3: 設定と記録
**Objective:** As a 研究者, I want 責任度の対象を設定として持ち、runの記録に残す, so that 結果から、どの対象で学習したかを確かめられる

#### Acceptance Criteria
1. The system shall 責任度の対象を、全保有モデル（既定）と、固い割り当ての標本を持つモデルの2つから選ぶ設定として持ち、既定以外の値をrunの設定の記録に残す。
2. If 固い割り当ての標本を持つモデルを選び、学習の割当方式が柔らかい学習でないとき、または責任度の重みの更新方式が予測と共有する方式（既定）でないとき, then the system shall runの準備の前に設定を拒否する。
3. The system shall 新しい構成のrunに、`fedsda-soft-training`と同じ記録（学習量のコンセプト別の記録、責任度の実効モデル数を含む）と指標を保存する。

### Requirement 4: 掃引への接続と既存の構成の維持
**Objective:** As a 研究者, I want 既存の掃引コマンドで新しい構成を選べる, so that uraniumで既存の結果と同じ手順で実行し、対にして集計できる

#### Acceptance Criteria
1. When 新しい構成を選ぶとき, the system shall 既存の掃引コマンドの計画の表示、並列の実行、保存、runの一覧のCSV、再開と競合の拒否まで、既存の構成と同じ手順で動く。
2. The system shall 既存の全構成の名前・順・設定・runの識別子・保存の形式・goldenを変えない。
3. The system shall 構成の一覧の文書に、新しい構成の目的と、`fedsda-soft-training`との違い（責任度の対象だけ）を書く。
