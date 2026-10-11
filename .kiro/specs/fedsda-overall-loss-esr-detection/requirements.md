# Requirements Document

## Project Description (Input)

ablationの構成の、移植の4つ目（順番を入れ替えた。予定の4つ目——AdaHedgeの重みで予測する構成——は、名前の判断をユーザーへ確認中）。提案構成の、損失の監視の対象を、全体の損失の系列だけにした構成を、新実装で実行できるようにする。

旧実装での指定（`tools/run_main_ablation_suite.sh`の`overall-esr`）: 構成名`FedSDA_NoCached_ResidualAdapter_ESR_RestartingSoftRouting`（ほかの引数は、提案構成と同じ）。旧のclient（`clients/fedsda.py`の`ESRFedSDAClient`）は、全体の損失の系列を、e-SRの検出器1つで監視する。検出器の表示名は`e-SR`。

提案構成（`ClassConditionalESRFedSDAClient`）は、全体の系列と、正解クラス別の系列の検出器を、固定の重みで混合する。

## Requirements

### Requirement 1: 全体の損失だけを監視する構成

#### Acceptance Criteria

1. Where 損失の監視の対象が「全体の損失」のとき, the 損失の監視 shall 全体の系列の検出器だけを更新し、e値・警報・変化の候補の位置を、その検出器のものにする。クラス別の検出器は、作らない。検出器の更新の件数は、1標本につき1。
2. The run設定 shall この監視の対象に対応する、検出器の表示名（`e-SR`）を要求する。
3. The 構成 shall 掃引のコマンドで、`fedsda-overall-loss-esr-detection`という名前で選べる。この構成のrun設定は、提案構成と、監視の対象と、検出器の表示名だけが違う。
4. The 構成の全体run shall 小さい条件（sine2）で、実旧の、同じ指定の全体runと、指標と離散列が一致する。
5. The 照合 shall この構成に固有の経路を通る: クラス別の検出器が無く、検出器の更新が、標本1件につき1回（提案構成は2回）。警報がある。結果が、提案構成と違う条件がある。
6. The 提案構成 shall 結果が変わらない。
