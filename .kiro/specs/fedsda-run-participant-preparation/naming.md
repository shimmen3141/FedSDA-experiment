# 命名表: fedsda-run-participant-preparation

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## ファイル・型

| 提案名 | 種類 | 役割・所有する状態 | 関連する名前との違い |
|---|---|---|---|
| `fedsda_run_server` | module（runtime） | 最終構成のFedSDAのサーバ1つの組立てと、実行の枠が求めるサーバの操作 | `fedsda_run_client`は、client 1つの組立てと操作。`server_round_synchronization`は、1ラウンドの同期の処理（サーバの操作が呼ぶ） |
| `FedsdaRunServerOwners` | frozen dataclass | サーバが持つ4つのowner（グローバルモデル、通信量、クロス評価の記録、クラスタリングの記録）への参照 | `FedsdaRunClientOwners`は、clientのowner |
| `FedsdaRunServer` | class | ownerと設定を持ち、3つの操作を、同期の関数と通信量のownerへ渡す。同期の結果を保持する | `FedsdaRunClient` |
| `fedsda_run_participant_factory` | module（runtime） | runごとの参加者（全clientとサーバ）の初期準備と、その設定の束 | `single_run_execution`は、factoryを受け取って全体runを進める枠 |
| `FedsdaRunParticipantSettings` | frozen dataclass | 全体runの参加者の準備に要る設定の束（clientの束、事前学習、モデルの構造、判定の基準、clientの上限） | `FedsdaRunClientSettings`は、client 1つの束（この束の一部） |
| `FedsdaRunParticipantFactory` | class | 実行の枠の`RunParticipantFactory`の実体。事前学習→全client→サーバを準備する | — |

## 関数・メソッド

| 提案名 | 入力 | 出力 | 役割 | 状態更新・副作用 |
|---|---|---|---|---|
| `assemble_fedsda_run_server` | clientの列、初期モデルID・パラメータ・統計、判定の基準、clientの上限、乱数生成器 | `FedsdaRunServer` | 4つのownerを作り、サーバを組み立てる | なし（新しいownerを作る） |
| `FedsdaRunServer.record_client_states_before_synchronization` | ラウンド | なし | 状態の報告の軽量メッセージを数える | 通信量 |
| `FedsdaRunServer.synchronize_models` | ラウンド、新規モデルの登録が可能か | なし | サーバの1ラウンドの同期を行い、結果を保持する | 各owner、client、乱数 |
| `FedsdaRunServer.finalize_started_communications` | 完了したラウンド数 | なし | 何もしない（終端まで持ち越す処理がない） | なし |
| `FedsdaRunServer.snapshot_server_round_synchronizations` | なし | 同期の結果のtuple | ラウンドの順の、同期の結果の写し | なし |
| `FedsdaRunParticipantFactory.validate_configuration` | なし | なし | 設定の束を再検査する | なし |
| `FedsdaRunParticipantFactory.prepare_run` | 固定条件、乱数源、標本生成器 | `RunParticipants` | 事前学習→全clientの組立て→サーバの組立て | 乱数、新しい参加者 |
| `FedsdaRunParticipantFactory.prepared_run_participants`（property） | なし | `RunParticipants`またはNone | 直前に準備した参加者 | なし |

## 判断が必要な点

- 操作名は、実行の枠の契約（`RunServerOperations`・`RunParticipantFactory`）の名前そのまま。
- 「participant」は、実行の枠の語（clientとサーバを合わせたもの）。束とfactoryは、clientだけでなく、サーバも準備するので、`run_client`ではなく`run_participant`にする。
