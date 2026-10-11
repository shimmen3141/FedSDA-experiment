# Design Document

## Overview

旧FedDriftServer._merge_cached_clustersの計算部分とBaseClient.apply_cached_mergeのパラメータ計算を、新methods/feddriftの計算へ移す。両者が同じ重みを異なる保存パラメータへ適用することをAPIで明示する。FedSDAの学習件数加重統合を流用しない。

## Boundary Commitments

所有するのは、分割とsnapshotからの代表ID・群内重み・加重parameter・統合統計・次の保存重み/隔離期限の生成。owner、モデル生成、通信、RNGは受け取らない。依存はtorchとlearning.loss_statisticsのBoundedLossMoments/検査だけ。数値部を先に分離する理由はresearch.md。後続runtimeは、全clientの計算/検査後にownerへ適用し、旧prepare_timestepの統合→登録とroundのFedAvg→配布を接続する。

## Components and Interfaces

`methods/feddrift/cached_model_cluster_consolidation.py`を追加する。

- `consolidate_feddrift_cached_model_clusters`: 非負時刻、モデルの完全分割、モデル別snapshot、過去重み、全体損失moment、隔離期限を受ける。`FeddriftCachedClusterConsolidation`を返す。
- `average_feddrift_cached_cluster_parameters`: 分割、clientまたはserver自身のsnapshot、代表ごとの群内重みを受け、代表ごとの新snapshotを返す。server計算もこの関数へ接続する。
- `FeddriftCachedClusterConsolidation`: 代表別parameters・loss moments・保存重み・残る隔離期限、全元ID→代表ID、代表別群内重みを持つ。frozen recordで属性再代入は不可、内部dictをimmutableとは約束しない。返すdictは入力と独立。

入力はbuiltin tuple/dict、ID/時刻/重み/期限はbuiltin int（boolを拒否）。分割は空でない各群、ID重複なし、snapshot ID集合の完全分割。空集合同士は許容する。snapshotは空でないstr→exact Tensor dict、非空CPU float32 strided/非nested/有限値。同じ群内ではキー集合と各キーのshapeが同じ（キー挿入順は一致不要）。異なる群のshapeは比較しない。重み/統計/期限dictのIDはsnapshotに存在し、欠落は許容する。server重みは負値も許容して0へ切り上げ、client重みは非負で全群・全memberに対応する。期限は非負。統計はexact BoundedLossMomentsを再検証する（fieldを手で破損した値も拒否）。クラス別統計はここへ渡さず、全体momentのみ扱う。

## Numerical Flow

serverは指定群順に最小IDを代表とする。各群のweightsはmember順にmax(過去重みまたは0,0)、合計0なら整数1へ置換。共通平均関数は全入力を検査後、no_grad下で合計と比率をPythonで計算し、最初のモデルのdict順で`value * ratio`、以後各モデルのdict順で`average[name] + value * ratio`。加算後に割る、stack/reduce、順のsortへ変更しない。均等fallbackでもclientは1.0、serverは1に由来する同じ比率で計算する。単独でも乗算し、新tensorを生成する。

統計はmember順に存在するmomentのみ集め、旧と同じbuiltin sumで件数・mean*nを計算し、合計nで割る。M2は合併せず0。保存重みは使用weightsの合計。期限は欠落0を含む最大値で、現在時刻と同じなら消去。全元IDのmappingは群/member順。出力の有限性も返却前に検査する。計算中の例外があっても入力・owner/RNGへの更新はない。

## Requirements Traceability

| 要求 | 部品/証拠 |
| --- | --- |
| 1.1, 1.2, 2.1, 2.2 | server計算・実旧_merge_cached_clustersとの完全一致 |
| 1.3, 2.3 | 共通平均・実旧_weighted_modelと実client.apply_cached_mergeの一致 |
| 3.1, 3.2 | 入力snapshot/勾配/3RNG・異常入力/空集合test |
| 3.3 | AST依存契約・fresh CPU・固定旧差分・全回帰 |

## 旧と違う点

旧が制約せず部分更新し得る契約外入力を新計算へは通さない。旧の正常経路のsnapshot/件数を受け、明示の型/有限値/完全分割を検証してから計算する。アルゴリズムと数値順は変更しない。旧clientのstore/optimizer再構築は本specの結果では再現しない。

## Validation

実旧serverを実__init__で生成し、実_merge_cached_clustersを呼ぶ（代役なし）。member非昇順、群非昇順、部分欠落/負/0/大重み、単独/空群集合、統計欠落/0件/不均等件数、期限過去/同時刻/未来を照合する。実旧clientも実__init__から生成し、client自身の値をserverと変えて実apply_cached_merge後のparamsを照合する。手計算の期待値だけで旧oracleを代替しない。対象test・依存・fresh・全pytest/旧11＋提案3golden、Ruff/Pyright/pip、独立CLIを完了条件とする。
