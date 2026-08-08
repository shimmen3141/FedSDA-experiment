# 実験設定と掃引計画

実験コードでは、「一回のrunで何を実行するか」と「何通りのrunを生成するか」を分離する。

## ExperimentConfiguration

[`experiment_spec/configuration.py`](../federated_drift_experiment/experiment_spec/configuration.py)の
`ExperimentConfiguration`は、mode・dataset・seed・アルゴリズム選択肢・解決済みパラメータを持つ。
一つのインスタンスは一つのrunに対応し、掃引の空指定や無効化状態は持たない。

`AlgorithmOptions`はクラスタリング方式、検出エピソード、新規モデル作成方針、FIFO長、検証数など、
掃引値とは独立したアルゴリズム設定をまとめる。

`ExperimentConfiguration.activated()` は、一つのrunに必要な設定だけを実行中に有効化し、
正常終了時も例外発生時も元の既定値へ戻す。これにより、掃引実行側は`config`の保存・書換え・復元を
個別に記述せず、解決済みrunを順番に実行するだけでよい。

```text
CLI -> SweepPlan -> ExperimentConfiguration -> activated() -> experiment
                         一つのrun             設定境界
```

`config.py`は削除せず、既定値とデータセット定義の一覧として残す。現在の`activated()`は、既存の
クライアント・サーバが参照するモジュール設定との移行境界である。将来、各コンポーネントへ設定を
直接渡す場合も変更箇所はこの境界の内側に限定できる。

## 完全な依存注入が必要になる条件

現在の実行方式は、1プロセス内でrunを直列実行する限り十分である。次のいずれかが必要になった場合は、
モジュール`config`を実行中に切り替える方式から、設定オブジェクトをコンポーネントへ直接渡す方式へ移行する。

- 同一Pythonプロセス内で複数runをスレッドまたは非同期タスクとして並列実行する
- 一つのrun内でクライアントごとに異なる設定を持たせる
- 同じプロセスで複数の実験環境を同時に保持する
- クライアント・サーバ単体テストで、グローバル設定の退避や`monkeypatch`を不要にしたい
- 実行途中の設定変更を禁止し、設定の不変性を型とコンストラクタで保証したい
- 実験を別プロセス・分散ワーカーへ送り、設定を直列化して再現したい

特に、`activated()`はプロセス全体のモジュール状態を一時変更するため、同一プロセス内の並列runには
対応しない。プロセスを分けた並列実験は、それぞれ独立した`config`を持つため現在の方式でも問題ない。

## 段階的な移行手順

全面的な書換えを避け、設定を使う場所から順に次の単位で移行する。

1. `ExperimentConfiguration`から、実験規模・データ・学習・検出・通信など用途別の不変設定を生成する。
2. `run_random_drift_experiment()`へ解決済み設定を渡し、データ生成とサーバ・クライアント構築で使用する。
3. クライアント、サーバ、検出器、モデルの各コンストラクタには、必要な部分設定だけを渡す。
4. 各クラス内の`config.*`参照を、コンストラクタで受け取った設定へ置き換える。
5. 移行済みコンポーネントについて、異なる設定のインスタンスを同時生成できるテストを追加する。
6. `experiment.py`以下から可変な`config.*`参照がなくなった後、`activated()`と`temporary_config()`を削除する。

```text
config.py（既定値カタログ）
        ↓ resolve
ExperimentConfiguration（1 runの完全な設定）
        ├─ DataSettings      → データ生成
        ├─ ClientSettings    → クライアント
        ├─ ServerSettings    → サーバ
        ├─ DetectorSettings  → 検出器
        └─ ModelSettings     → モデル
```

設定オブジェクトを一つ丸ごと全クラスへ渡すのではなく、各クラスが必要な小さい部分設定だけを受け取る。
これにより引数の増加と不要な依存を抑える。`config.py`は移行後も既定値カタログとして残し、削除しない。

## SweepPlan

[`experiment_spec/sweep.py`](../federated_drift_experiment/experiment_spec/sweep.py)の`SweepPlan`は、データセット、seed、modeと
複数の`SweepAxis`を保持し、`ExperimentConfiguration`列を生成する。

```text
SweepPlan
 ├─ datasets / seeds / modes
 ├─ AlgorithmOptions
 └─ SweepAxis[]
      ├─ 変化させるparameter_id
      ├─ sweep values
      ├─ 適用対象（FedSDA ADWIN / FedSDA / FedDrift）
      └─ fixed values
             ↓
      ExperimentConfiguration[]
```

## 掃引時の固定値

固定値はSweepPlan全体へ曖昧に置かず、対応する`SweepAxis.fixed_values`へ所属させる。

| 掃引軸 | 軸の値 | その軸に属する固定値 |
|---|---|---|
| δ_ADWIN sweep | δ_ADWIN | A、γ |
| A sweep | A | δ_ADWIN、γ |
| B_detect sweep | B_detect | δ_FedDrift |
| δ_FedDrift sweep | δ_FedDrift | B_detect |

たとえば`--fixed-aggregation-interval`は「実験全体の固定A」ではなく、δ_ADWIN掃引から生成される
runにだけ反映される。A掃引ではA自身が掃引値となり、同じ固定値は使わない。

CLIの未指定・無効化・既定値は`SweepPlan`を構築する前に解決する。これにより実行層は、空リストや
「親オプションがオフなので無視」といったCLI固有状態を扱わない。

実験を開始せずに解決結果を確認するには、通常のコマンドへ`--print-plan`を追加する。対象mode、
各掃引軸、固定値、総run数が表示され、出力ディレクトリは作成されない。

```bash
python run_pareto_sweep.py --no-feddrift --no-baselines --no-adwin-sweep --aggregation-intervals 50 100 --print-plan
```
