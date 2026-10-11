# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験を、研究室サーバで実行する研究者と、公開されたコードで実験を再現する人。

現在の状況: 新実装は、完全なrun設定から、1 runを実行して、要約（JSON）と記録の表（NPZ）を保存できる（関数として）。複数の条件（dataset、seed、集約間隔）をまとめて、並列に実行するコマンドは、ない。旧実装は、`run_pareto_sweep.py`と`tools/run_server_sweep.sh`で、これを行っている。

ユーザーの決定（2026-10-11）: ablationを、新実装でも実行する（計算量など、集計方法を変えた指標を確かめる。公開するコードで、他人が実験を再現できるようにする）。まず、新実装の実験の入口を、提案構成だけで作る。旧実装の構成名は分かりにくいので、新実装の語に合わせて、分かりやすくする。新実装の実験本体は、MNISTのファイルを取得しない（置いてあるファイルだけを読む）。

## Introduction

研究者が、1つのコマンドで、構成・dataset・seed・集約間隔の組合せを、並列に実行し、runごとの結果と、一覧の表を、決まった場所に得られるようにする。途中で止まっても、同じコマンドで、済んだrunを飛ばして続けられる。

## Boundary Context

- **In scope**: 掃引の計画（構成・dataset・seed・規模・集約間隔の組合せ）、runの識別子、逐次と並列の実行、済んだrunを飛ばす再開、runごとの保存、構成ごとの一覧の表（CSV）、実行の記録（計画、コマンド、コードの版、環境）、コマンド。構成は、提案構成だけ。
- **Out of scope**: 提案構成以外の構成（ablationの構成、FedDrift。後のspec）。図の作成（Paretoの図ほか）。MNISTのファイルの取得（旧実装を外す段階で、小さいコマンドを別に作る）。結果のディレクトリをまたぐ、重複の検査。旧実装の入口（`run_pareto_sweep.py`、`tools/`）は、変えない。
- **Adjacent expectations**: 1 runの実行と保存（`execute_fedsda_experiment_run`、`save_fedsda_experiment_run`）の結果は、変えない。

## Requirements

### Requirement 1: 掃引の計画

#### Acceptance Criteria

1. When 構成の名前・dataset・seed・client数・clientあたりの標本数・集約間隔の一覧を与えたとき, the Experiment Sweep Plan shall その全部の組合せのrunを、決まった順（構成、dataset、集約間隔、seed）で返す。各runは、完全なrun設定と、識別子を持つ。
2. The 識別子 shall 人が読める部分（dataset、集約間隔、seed）と、構成の名前と完全なrun設定から計算した短いhashを持ち、同じ条件なら、いつも同じになる。条件のどれかが違えば、違う識別子になる。
3. If 一覧が空、値が重複、知らない構成やdataset、不正な値のとき, the 計画 shall 実行の前に、拒否する。

### Requirement 2: 実行と保存

#### Acceptance Criteria

1. When 計画のrunを実行したとき, the Experiment Sweep Execution shall runごとに、要約と記録の表を、`<出力先>/<構成の名前>/runs/<識別子>.json`・`.npz`へ保存する。保存した内容は、同じ設定で、1 runを直接実行して保存した内容と、同じである。
2. While worker数が2以上のとき, the 実行 shall runを、別々のprocessで並列に実行する。各processの計算のthreadは、1つにする。結果は、逐次の実行と、同じである。
3. When 出力先に、同じ識別子の、済んだrun（要約と表の両方があり、要約のrun設定が、計画と同じ）があるとき, the 実行 shall そのrunを実行せず、済んだものとして数える。
4. If 同じ識別子のファイルが、片方しかない、または、要約のrun設定が計画と違うとき, the 実行 shall そのrunを上書きせず、失敗として報告する。
5. If あるrunが例外で失敗したとき, the 実行 shall ほかのrunを続け、失敗したrunと理由を報告する。

### Requirement 3: 一覧の表と、実行の記録

#### Acceptance Criteria

1. When 実行が終わったとき, the 実行 shall 構成ごとに、その構成の`runs/`にある全部の要約から、1 runを1行とする表（CSV: 識別子、構成の名前、実行条件、指標の全部——入れ子の指標は、`.`でつないだ名前——）を作る。
2. When 実行が終わったとき, the 実行 shall 実行の記録（計画、コマンドの引数、コードの版——gitのcommitと、未コミットの変更の有無。取れなければ、取れなかったこと——、実行環境、開始と終了の時刻、runごとの結果）を、出力先へ、実行ごとの別のファイルとして保存する。

### Requirement 4: コマンド

#### Acceptance Criteria

1. The コマンド shall リポジトリをcloneした状態で、packageのinstallや、環境変数の設定なしに、リポジトリの直下から実行できる（Pythonの環境と、MNISTを使うときのファイルの配置を除く）。
2. When `--print-plan`を指定したとき, the コマンド shall 計画のrunと、出力先にある済んだrunを表示し、何も実行せず、何も作らない。
3. If MNISTのdatasetを計画に含み、MNISTのファイルが置かれていないとき, the コマンド shall 実行の前に、置き場所を示して、止まる。
4. The コマンド shall 全部のrunが、済んだか、飛ばされたとき、終了コード0を返し、失敗したrunがあるとき、0以外を返す。
