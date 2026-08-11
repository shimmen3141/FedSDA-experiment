# 実験設定と掃引計画

実験コードでは、「一回のrunで何を実行するか」と「何通りのrunを生成するか」を分離する。

## 実行manifestと重複検出

`run_pareto_sweep.py`は実験開始時、Pareto・raw出力先の共通親へ`manifest.json`を作成する。
manifestには解決済みrun設定、CLI引数、コード・実行環境・回帰goldenのSHA-256、開始時刻を保存し、
正常終了時にCSVハッシュ・raw件数・完了時刻を追記する。例外や中断を捕捉できた場合は`failed`、
プロセスが強制終了した場合は`running`のまま残るため、未完了結果も区別できる。

開始前には`results/`以下の完了manifestをrun単位で照合する。計画全体ではなく、一つでも
同一設定・同一実装・同一数値環境・同一goldenのrunがあれば、該当manifestの場所と件数を表示する。
既定の`error`は最初のrunを開始する前に計画全体を停止する。意図的な再実験では`warn`、照合自体が
不要な場合だけ`ignore`を指定する。コードまたはgoldenが異なる場合は「設定一致・由来相違」として
表示するだけで、既定では実験済みと判定しない。

過去のCSVにはコード由来が保存されていないため、次で事後manifestを作成できる。

```bash
python -m tools.experiments.manifests backfill results/results_YYYYMMDD_HHMMSS
python -m tools.experiments.manifests backfill results --recursive
# schedule記録導入前の結果で、実験条件からrandomと確認できる場合だけ明示する
python -m tools.experiments.manifests backfill results --recursive --concept-schedule random
```

`--recursive`は各`pareto/`の親を独立した実験variantとして検出し、回復分析などのCSVを除外して
一括補完する。このmanifestは`provenance_status=unknown_backfill`となり、将来の実験と設定が一致しても
警告だけで自動停止の根拠にはしない。人間向けの巨大な一覧は生成せず、実験開始前の照合が必要箇所だけを
機械的に表示する。

CSVを失ったがNPZが残る結果は次で復元する。

```bash
python -m tools.experiments.artifacts results/results_YYYYMMDD_HHMMSS --tag recovered
```

新しいNPZには全CSV指標を埋め込むため完全復元となる。旧NPZでは、accuracy・stable_accuracyを履歴から、
通信モデル数・最終モデル数を進捗ログから復元し、それ以外の復元不能な値は`NaN`とする。この場合は
`.reconstruction.json`の`quality=partial`を記録し、論文用baselineの根拠にはしない。復元後に
`backfill --recursive`を実行すれば、旧結果も設定単位の事前照合対象になる。

成果物のstemは72文字を上限とし、超過分を内容由来のSHA-256短縮値へ置き換える。NPZ名も
`run_<dataset>_s<seed>_<hash>.npz`とし、完全な条件はNPZ・CSV・manifest内部に保持する。

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

`run_pareto_sweep.py --workers N`は、この性質を利用して解決済みの
`ExperimentConfiguration`をspawn型の独立プロセスへ一つずつ渡す。各プロセス内ではrunを逐次実行し、
親プロセスだけが最終CSVと図を生成する。完了順が変わっても、CSV行は`SweepPlan`の計画順へ戻す。
`TOTAL_DATA_POINTS`、クライアント数、事前学習規模など、run設定の外側にある実験規模もworker初期化時に
明示的に複製する。このため、並列数によって精度・通信量・検出・モデル数・計算回数は変化しない。
一方、`runtime_seconds`と`client_compute_seconds_*`は資源競合を含む実測値なので一致対象ではない。

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
各掃引軸、固定値、総run数に加え、既存manifestと一部でも重複するrun数と該当manifestが表示される。
出力ディレクトリやmanifestは作成されない。

```bash
python run_pareto_sweep.py --no-feddrift --no-baselines --no-adwin-sweep --aggregation-intervals 50 100 --print-plan
```
