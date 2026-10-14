# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの精度を、FedDrift・FedAvgと比べている研究者（ユーザー）。
- 今の状況: 新実装のFedAvg（構成`fedavg`）は、旧Oblivious（旧の局所更新の間隔τ=1）を引き継ぎ、標本を1つ予測するたびに局所更新を1回行う。round内の後の標本は、局所更新したモデルで予測する。元論文のFedAvgは、roundのデータで学習してから集約し、集約・配布されたモデルで予測する手順で、FedDriftの実装もこの形である。手元の診断（研究バックログの8）で、この更新の時機の違いが、変化の無い時期の精度に1〜3ポイントの差を生むことが分かった。
- 何を変えるか: 局所更新の間隔τを集約の間隔と同じにしたFedAvgを、構成`fedavg-local-updates-per-aggregation`として掃引のコマンドで選べるようにする。今の`fedavg`は変えない。

## Introduction
2026-10-15の会話で、ユーザーは、元論文に沿った形のFedAvgの基準値を研究室サーバで取ると決めた。掃引の軸（τの値を並べる形）でなく、構成名1つで指定する形と、名前`fedavg-local-updates-per-aggregation`を選んだ。

## Boundary Context
- 対象: 構成の一覧と、構成からrun設定を作る処理、goldenの事例の一覧の扱い（構成の一覧から作る今の形を、基準版1〜5の事例を固定する形へ変える）、新しい構成のtest、文書。
- 対象外: FedAvgのclient・サーバの処理（τ>1の処理は既にある）、FedSDA・FedDrift、掃引の軸、一覧の表の列（構成名の列で区別できる）、動作の基準版（5のまま）。
- 仮定: 構成の局所更新の間隔は、そのrunの集約間隔（`server_aggregation_interval_per_client_samples`）と同じ値にする。集約間隔の掃引で、runごとに値が変わる。

## Requirements
### Requirement 1: 局所更新を集約ごとにまとめるFedAvgを、構成名で選べる
**Objective:** As a 研究者, I want 局所更新を集約の間隔ごとにまとめるFedAvgを、構成名1つで実行できること, so that 元論文に沿った形のFedAvgの基準値を、ほかの手法と同じ条件で取れる。
#### Acceptance Criteria
1. The sweep command shall `--configurations`に`fedavg-local-updates-per-aggregation`を受ける。
2. The configuration shall `fedavg`の既定のrun設定と、局所更新の間隔（`local_update_interval_per_client_samples`）だけが違い、その値はrunの集約間隔と同じ。
3. When この構成で実行したとき, the FedAvg client shall roundの中の全部の標本を、そのroundの始めに配布された（集約された）モデルで予測し、局所更新は、roundの終わりに、集約間隔×1標本あたりの更新回数だけまとめて行う（保持標本がbatchの件数以上のとき）。
4. The run artifacts shall 構成名のディレクトリ（`<出力先>/fedavg-local-updates-per-aggregation/`）に、`fedavg`と同じ形で保存される。

### Requirement 2: これまでの照合を保つ
**Objective:** As a 開発者, I want 既存の構成と照合が値を変えずに通ること, so that 構成を足しても既存の動作が変わらないと確かめられる。
#### Acceptance Criteria
1. The validation shall 基準版1〜5のgoldenを、値も事例の一覧も変えずに通す。
2. The validation shall 全部の構成が、goldenの事例か、golden以外のtestで守る構成の一覧のどちらかに入っていることを確かめる（構成を足すとき、守り方を決め忘れない）。
3. The `fedavg` configuration shall run設定とrunの識別子を変えない。

### Requirement 3: 文書
**Objective:** As a 研究者, I want 文書が新しい構成と今の`fedavg`の違いを説明すること, so that 比べる対象を取り違えない。
#### Acceptance Criteria
1. The documentation shall 実験の入口の文書（`docs/experiments/experiment-sweep.md`）に、新しい構成、今の`fedavg`との違い（局所更新の時機と、予測に使うモデル）、実行の例を書く。集約間隔がbatchの件数（32）より小さいと、累積の保持標本が足りない最初の数roundの局所更新が、学習されずに捨てられること（既存の挙動。集約間隔は利用者が決める）も書く。
