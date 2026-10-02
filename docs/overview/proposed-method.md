# FedSDAの最終提案構成 — Switching SoftRouting

本書は、論文で最終提案として評価するFedSDAの採用構成・処理順・説明範囲の正本である。
個別機能の全選択肢はcomponents、実験条件と比較対象はexperiments、実装の設定値はreferenceに分ける。
旧ADWIN構成の[fedsda-algorithm.md](fedsda-algorithm.md)は、過去の設計を説明する資料として保持する。

## 1. 解決する問題と各要素の役割

クライアントごとに異なる概念が時間とともに変化・再来するデータストリームを対象とする。
複数の概念モデルを保持し、予測、学習データの帰属、モデル再利用・作成、サーバでの統合を組み合わせる。

| 役割 | 採用する構成 | 説明する狙い | 詳細 |
|---|---|---|---|
| 損失の変化を監視 | ClassESR | 全体と正解クラス別の変化を監視する | [ドリフト検出](../components/drift-detection.md) |
| 概念間で表現を共有 | 共有バックボーン＋rank 8 Residual Adapter | 共通表現と概念固有の補正を分担する | [モデル構造](../components/shared-backbone.md) |
| 保持モデルを学習 | joint学習・平均勾配 | 概念別ストアの損失から共有部と概念固有部を更新する | [ローカル学習](../components/shared-backbone.md#ローカル学習方式) |
| ラベル観測前に予測 | Switching SoftRouting | Fixed-Shareで優勢モデルの変化を追跡し、予測を混合する | [SoftRouting](../components/soft-routing.md) |
| 集約で変わったモデルへroutingを追従 | FIFO replay | 集約後のモデルでFIFO標本を再評価し、routingの損失情報を再較正する | [再較正](../components/shared-backbone.md#集約後のルーティング再較正) |
| 再来した概念へ対応 | 最良適合の既存モデルを再利用 | 適合する保持モデルがあれば、新規モデル作成を回避する | [モデル再利用](../components/model-reuse.md) |
| 新しい概念へ対応 | forward persistent | 既存モデルの再適合と候補の前半・後半での優位を確認して新規作成する | [新規モデル作成](../components/new-model-creation.md#forward_persistent) |
| モデル群を整理 | class-functional confidence、average linkage、通常merge | クラス別の機能的評価で統合候補を作り、モデルを統合する | [統合処理](../components/model-consolidation.md) |

採用要素の役割と、研究として新規性を主張する範囲は別途対応付ける。
各要素の有効性は、精度・回復・通信・計算・モデル数のどこに寄与するかを比較実験で示す。

## 2. 最終構成の固定設定

実装上のmode名は`FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting`である。
mode名の`RestartingSoftRouting`だけでは予測方式を一意に定められないため、文脈`switching`まで含めて指定する。

| 設定 | 採用値 |
|---|---|
| SoftRouting文脈 | `switching` |
| SoftRouting有効化期間 | `always`（常時混合） |
| routingのモデル集合 | `all` |
| 共有表現の学習・勾配統合 | `joint`・`mean` |
| Residual Adapter rank | 8 |
| 集約後再較正 | `fifo_replay` |
| 新規作成方針 | `forward_persistent` |
| FIFO長・前向き検証件数 | N_FIFO=30、N_forward=10 |
| 適合・統合判定の設定値 | γ=0.1 |
| クラスタリング判定・linkage | `class_functional_confidence`・`average` |
| クラスタリング実行方針 | `on_new_model` |
| クラスタリング後の処理 | `merge` |
| 検出エピソード統合 | 無効 |

採用値は主要実験の構成を定義する。実装の`config.py`にある全オプションの初期値とは一致しない。
例えば`SOFT_ROUTING_CONTEXT`の初期値は`global`であり、主要実験では`switching`を明示指定する。
完全なrun設定は各実験のmanifest、設定の能力・依存関係は[自動生成リファレンス](../reference/options.md)を参照する。

## 3. 予測・割当・学習の区別

- **予測**: 保持モデルの出力をSwitching重みで混合する。重みは現在のラベルを観測する前のものを使う。
- **割当**: `current_model_id`はドリフト検出、履歴損失、データストアへの帰属に使う。
- **学習**: 概念別データストアを使い、共有バックボーンと概念固有部をjointで更新する。

予測の重み最大モデルが、そのままデータ帰属先になるとは限らない。
ClassESRが監視するのは現行割当モデルの損失であり、混合した最終予測の損失ではない。
FIFOのデータ帰属保留と、前向き検証sessionの候補比較も別の状態として説明する。

## 4. 1サンプルとサーバ同期の処理順

1. 新しい入力に対して各モデルの出力を得て、Switching重みで予測する。
2. 正解ラベルで予測を評価し、次回用のrouting損失情報を更新する。
3. 前向き検証sessionがあれば新着標本を観測し、必要な件数に達した場合は採否を確定する。
4. その時点の現行割当モデルの損失でClassESRを更新し、新着標本をFIFOへ追加する。
5. 警報時はFIFOの検知区間を評価し、既存モデル再利用または仮モデルの前向き検証へ進む。
   平時はFIFO上限を超える最古標本を現行ストアへ確定し、ローカル学習する。
6. 集約間隔Aの境界では、サーバで新規ID登録、モデルごとのFedAvg、必要時のクラスタリング・merge、配布を行う。
7. 配布後のモデルでFIFO replayを行い、次のサンプルへ進む。

前向き検証の確定は、その標本のClassESR更新より前に行われる。
新規ID登録はID管理であり、パラメータの二重アップロードを意味しない。
条件分岐の詳細は`clients/fedsda.py::process_one_step`と`servers/fedsda.py::run_round`を正本とする。
作成中のコマ送り図はローカルの`docs/overview/fedsda-processing-flow.html`で確認できる。

## 5. Switchingを採用する理由

Switchingはモデル出力を直接混合する構成である。
Meta-switchingは、クラス文脈を使うMeta mixtureとSwitching mixtureを上位で選択する構成で、説明する階層が増える。

既存random実験の6データセット・5 seed・4集約間隔を等重みで集計すると、
SwitchingとMeta-switchingの全期間精度は約91.290%と91.339%だった。
平均差は約0.049パーセントポイントで、Switchingがすべての条件で高精度だったわけではない。
単純な構成で近い精度を得られたことから、Switchingを主要構成とし、Meta-switchingは設計選択の補足比較に置く。
実装ハッシュが異なる結果を含むため、コード来歴の差は実験資料で付記する。

## 6. 実験と主張の対応

| 区分 | 答える問い | 配置 |
|---|---|---|
| 手法全体の比較 | FedDriftに対する精度・回復・資源消費はどうか | experiments |
| 主要ablation | 採用要素を除去・置換すると何が変わるか | [ablation計画](../experiments/ablation-plan.md) |
| 設計選択・感度分析 | Switching、rank、学習方法などをなぜ採用したか | componentsの設計説明とexperimentsの結果索引 |
| 過去の探索・今後の候補 | 何を試し、何を採用せず、何を残したか | [研究バックログ](../research/research-backlog.md) |

forward検証を除くと精度が上がる条件もあるため、その寄与は新規モデル数・通信・計算量とのトレードオフとして示す。
hard-routing比較は付随するFIFO再較正も無効になり、混合だけを変更した比較とは解釈しない。
FedDriftの復元byte値は既定MLP・float32を仮定した参考値で、実行時前提を確認するまで正式baselineへ採用しない。

## 7. 採用範囲と検討資料

Meta-switching、PCGrad、Shadow tournament、非劣性merge、`drift_recovery` activationなどは
選択可能な方式・比較候補として個別文書に残す。最終構成への変更は本書と比較条件を合わせて更新する。
真の概念IDを使うoracle診断は、評価の補助であり最終提案の予測・検出・学習には使わない。
論文草稿`main_jp.tex`と旧実装用仕様書のADWIN中心の説明は、最終構成に合わせる編集が必要である。
