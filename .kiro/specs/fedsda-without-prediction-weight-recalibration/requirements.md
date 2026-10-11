# Requirements Document

## Project Description (Input)

ablationの構成の1つ目。提案構成から、「集約後の、予測重みの再較正」を外した構成を、新実装で実行できるようにする。

旧実装での指定（`tools/run_main_ablation_suite.sh`の`no-recalibration`）: 提案構成と同じ構成名で、`--shared-backbone-routing-recalibration none`。旧のサーバ（`servers/shared_backbone.py`の`run_round`）は、この値のとき、ラウンドの同期の後の、clientの再較正（`recalibrate_routing_after_aggregation`）を、呼ばない。予測重み（Fixed-Share）も、診断重み（global、真の概念別）も、集約の後、そのまま使い続ける。

提案構成では、配布を受け取った後、保留中の標本の損失を、新しいモデルで計算し直して、予測重みとglobalの診断重みを作り直し、真の概念別の診断重みを始め直す。

## Requirements

### Requirement 1: 再較正をしない構成

#### Acceptance Criteria

1. Where 予測重みの、集約後の再較正の方針が「再較正しない」のとき, the FedSDA Run Client shall サーバの同期の後、予測重みと診断重みを、変えない。
2. The 構成 shall 掃引のコマンドで、`fedsda-without-prediction-weight-recalibration`という名前で選べる。この構成のrun設定は、提案構成と、再較正の方針だけが違う。
3. The 構成の全体run shall 小さい条件（sine2）で、実旧の、同じ指定の全体runと、指標と離散列が一致する。
4. The 照合 shall この構成に固有の経路を通る: 再較正の回数と、再生した標本数が、新旧とも0で、同じ条件の提案構成では0でない。結果が、提案構成と違う条件がある。
5. The 提案構成 shall 結果が変わらない。
