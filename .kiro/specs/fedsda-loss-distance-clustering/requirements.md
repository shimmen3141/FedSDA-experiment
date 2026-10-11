# Requirements Document

## Project Description (Input)

ablationの構成の3つ目。提案構成の、サーバのクラスタリングの「モデル対の比べ方」を、損失の増加量へ替えた構成を、新実装で実行できるようにする。

旧実装での指定（`tools/run_main_ablation_suite.sh`の`distance-average`）: 提案構成と同じ構成名で、`--clustering-decision distance --cluster-linkage average`。旧のサーバ（`servers/clustering.py`の`perform_hierarchical_clustering`）は、この値のとき、モデル対の判定の値として、損失の距離（互いの標本での平均損失が、自分の標本での平均損失より増える量の、大きいほう）を、そのまま使う。閾値（`distance_threshold`）と、linkage（average）は、提案構成と同じ。

提案構成では、判定の値は、クラス別の「片方だけが正解した割合」の、同時信頼下限の最大。

## Requirements

### Requirement 1: 損失の増加量でクラスタリングする構成

#### Acceptance Criteria

1. Where クラスタリングの判定の基準の、モデル対の比べ方が「損失の増加量」のとき, the サーバ shall 損失の距離を持つ対（4つの損失の統計の件数が、すべて下限以上）の判定の値として、損失の距離を使う。正誤の集計は、使わない。
2. The run設定 shall サーバが読む判定の基準と、統合の方式の宣言とで、モデル対の比べ方が違う組合せを、拒否する。
3. The 構成 shall 掃引のコマンドで、`fedsda-loss-distance-clustering`という名前で選べる。この構成のrun設定は、提案構成と、モデル対の比べ方（判定の基準と、宣言の2か所）だけが違う。
4. The 構成の全体run shall 小さい条件（sine2）で、実旧の、同じ指定の全体runと、指標と離散列が一致する。
5. The 照合 shall この構成に固有の経路を通る: 対の判定の値が、全部、損失の距離と同じで、同じクラスタになった対と、ならなかった対の、両方がある。同じ条件の提案構成では、判定の値が損失の距離と違う対があり、結果が違う条件がある。
6. The 提案構成 shall 結果が変わらない。
