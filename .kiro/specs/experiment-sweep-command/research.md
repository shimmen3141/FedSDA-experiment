# Research & Design Decisions

## Summary

- **Feature**: `experiment-sweep-command`
- **Discovery Scope**: New Feature（1 runの実行と保存の上に、掃引と、コマンドを足す）
- **Key Findings**:
  - 旧の入口（`run_pareto_sweep.py`、1520行）は、掃引の軸の解決、並列（`spawn`のprocess、重いrunから投入）、CSV、図、manifest、重複の検査を、1つのscriptで行う。`tools/run_server_sweep.sh`が、出力先とログを組み立てる。
  - 新の1 runは、完全なrun設定（`FedsdaRunSettings`）だけで決まり、要約に、run設定の全部と、実行条件が入っている。出力のディレクトリの中身だけから、一覧の表を作り直せる。
  - 新のpackageは、installしていない（testは、pytestの設定で`src`を経路へ足す）。

## Design Decisions

### Decision: リポジトリ直下の、薄いscriptを入口にする

- **Alternatives**: (1)`pip install -e .`してから`python -m …`。(2)`PYTHONPATH=src python -m …`。(3)直下のscriptが、`src`を経路へ足す。
- **Selected Approach**: (3)。
- **Rationale**: cloneした人が、installも、環境変数も要らない（ユーザーの要望: 分かりにくい設定・操作を必要としない）。

### Decision: 済んだrunを飛ばす再開を、出力先の中だけで行う

- **Context**: 旧は、結果のルートの全部のmanifestと照合して、重複を止める。
- **Selected Approach**: 出力先の、同じ識別子のファイルだけを見る。run設定が同じなら飛ばし、違えば上書きせずに失敗にする。ディレクトリをまたぐ重複の検査は、しない。
- **Rationale**: 識別子が、run設定の全部のhashを持つので、同じ出力先では、取り違えが起きない。長い実験が途中で止まったとき、同じコマンドで続けられる。別の出力先との重複は、実験の計画の問題で、必要になったら足す。

### Decision: 一覧の表は、`runs/`の要約から、毎回作り直す

- **Rationale**: 実行の途中の状態を持たないので、再開や、並列の実行と、食い違わない。

### Decision: 実験の構成を、名前で選ぶ

- **Selected Approach**: 構成の名前から、完全なrun設定を作る関数を、1か所に置く。構成を足すときは、ここへ足す。
- **Rationale**: ablationの構成は、提案構成から、1要素だけ違う。名前と、変える点の対応を、1か所で読める。

## Risks & Mitigations

- 並列のprocessで、結果が変わる（乱数、thread） — 各runは、自分のseedから乱数を作る。threadは1つにする。並列と逐次の結果が同じことを、testで確かめる。
- Windowsで、`spawn`のprocessが、packageを読めない — 親のprocessの読込みの経路が、子へ渡ることを、並列のtestで確かめる（Windowsの基準環境）。Linuxは、WSLで確かめる。
