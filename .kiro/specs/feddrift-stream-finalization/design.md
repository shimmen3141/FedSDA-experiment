# Design Document

## Overview
`FeddriftRunServer.finalize_observed_stream`を唯一の終端入口とする。既にserverが借りるclient tupleを使い、factoryの出力からそのまま呼ぶ。通常同期のR回進行を呼ばない。既存のclient flush・登録確認・配布ownerを再利用する。

## Requirements Traceability
| 要求 | 設計・検証 |
|---|---|
| 1.1, 1.2 | 全client buffer preflight→client順flush。空経路の実旧・無変化対照 |
| 2.1 | ready payload事前検査→ID発行/登録確認→ID順FedAvg→全配布。実旧run_round対照 |
| 2.2 | 通常/終端で共通の集約候補生成を使い、終端ではrepository IDsと重み更新無効を指定。学習/診断/隔離/通信を対照 |
| 2.3 | server処理を行った後だけ残るpendingをreadyへ設定。待機有無の対照 |
| 3.1 | 非更新検査と拒否時snapshot/RNG対照。生成結果は部品内で検査 |
| 3.2 | grad無効の終端とfactoryからの連続対照・共用fresh・依存境界 |

## Architecture & Components
- server入力検査は既存のモデル/optimizer/統計/sample/pending検査を再利用し、終端では学習用のgrad-enabled条件だけ外す。label等のsample契約は維持する。
- buffer preflightは全clientのprocessed count、buffer内容、検出器の損失/閾値と次の一時IDを非更新で検査する。空bufferにはbatch実行を行わない。
- 最終回収ではready snapshotのkey/shape・統計と、割当予定の正式IDが全clientの既存IDに衝突しないことを検査してから、順にparameter upload/登録通知、ID発行、値・統計・lineage登録、owner確認、確認通知を行う。新登録は初期集約重みを持ち、隔離を付けない。
- 集約は既存のclient順のmultiply/add/divideと統計の加重平均/M2=0を保つ。通常同期と共有する非公開helperの引数で対象IDとcached重み更新を指定する。値と統計を検査した後に集約通信とrepository更新・全配布を行う。終端ではcached重みを更新しない。
- 最終回収後に残るdelay付きpendingは既存queue-ready操作で同じpayloadをreadyへ昇格する。新しい学習・観測は行わない。

## Behavioral Contract
時刻は非負builtin int。exact client/server設定、共有run Random、owner入力契約を再検査する。flushが生成する数値は各client既存の候補計画内で検査する。全体トランザクション/rollbackはない。再呼出しは空buffer・readyなしなら何も増やさない。

## Differences from Legacy
- 公開APIは旧run_round/finalize_protocolの互換aliasを設けず、観測済みbufferから完結する終端操作とする。
- 契約外入力は可能な入力検査を全flushより前に集める。成功時の計算順/乱数は維持する。
- 旧の通常同期用学習条件を終端の入力検査へ混ぜず、no_gradで動く旧終端の能力を維持する。

## Test Strategy
実旧client.flushとBaseServer.run_roundを直接実行し、全client/server owner、parameter/grad/optimizer、統計・重み・隔離・lineage・通信8項目・診断・3乱数を既存helperで照合する。空、非検出buffer、ready、複数ready、待機pending、既存モデル/ゼロtraining、SGD/Adam、多クラス、通常同期後の終端、再呼出し、破損入力を覆う。factoryの実旧setupからも接続する。独立レビュー後に固定環境の全pytestとidentityを実施する。
