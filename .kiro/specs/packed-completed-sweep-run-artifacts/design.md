# Design Document

## Overview
全runが済んだ掃引の最後に、構成ごとの`runs/`（runごとの要約のJSONと記録の表のNPZ）を、その構成のディレクトリの`runs.zip`へまとめる。まとめた出力先では、runの状態の判定をzipから読み、計画にzipに無いrunがあるときだけ、計算の前にzipを`runs/`へ戻す。runの値・乱数の消費・記録の形式は変えない。開始の記録のまとめ（spec `lzma-record-tables-and-packed-run-start-records`）は今のまま。

## Boundary Commitments
- 持つもの: 新しい`sweep/packed_run_artifact_archive.py`（まとめ・zipからの読み出し・戻し）、`sweep/experiment_sweep_execution.py`の`find_existing_run_state`（zipから判定）と`execute_experiment_sweep`（計算の前の戻し）、`sweep/experiment_sweep_command.py`と`run_behavior_fix_comparison.py`の終わりのまとめの呼び出し、依存境界の許可リスト、共用のfresh smoke、文書（`docs/experiments/experiment-sweep.md`、研究ツール`tools/research_results_archive/archive.py`の冒頭の説明）。
- 持たないもの: runの計算と成果物の書き出し（`*_run_artifact_files.py`）、読込み関数、開始の記録、掃引の計画、`run_summary.csv`の書き方、研究ツールの処理（説明の文だけ足す）、製品の外の分析のツール。
- 許す依存: 新しいmoduleは標準の`hashlib`・`json`・`os`・`time`・`uuid`・`zipfile`・`pathlib`・`dataclasses`だけ（torch・numpy・ほかの製品のmoduleを読まない）。`experiment_sweep_execution.py`は新しいmoduleの公開の名前と`tempfile`を足す。
- 再検証が要る変更: まとめた後の出力先は`runs/`を持たない。`runs/`を直接読む製品の外のツール（`tools/`の多く）は、研究ツールの`unpack`の後に使う（文書に書く）。製品の中で`runs/`を読むのは次の4つ。
  - 状態の判定: この設計で扱う。
  - `run_summary.csv`の書き出し: `runs/`が無ければ書き直さない（今と同じ条件）。
  - 再開の候補の探索: 状態の判定を通る。
  - 中断したrunの直し: `runs/`にNPZだけが残るrunだけが対象で、まとめた構成には無い。

## zipの形式（研究ツールの`unpack`と互換）
- 置き場所: `<出力先>/<構成>/runs.zip`。zipの中のパスは`runs/<元のファイル名>`。
- 要約（`.json`）は`ZIP_LZMA`で入れる。記録の表（`.npz`。製品がLZMAで書いたNPZ）は、`ZIP_STORED`でバイト列をそのまま入れる。
- `archive_manifest.json`（`ZIP_LZMA`）: `{"files": [{"path": "runs/<名前>", "bytes": <バイト数>, "sha256": "<hex>"}, ...]}`（パスの昇順）。
  - 研究ツールの`unpack_configuration`は、`arrays`の無い項目をsha256で照合するので、そのまま戻せる。
  - 研究ツールのmanifestにある`arrays`（NPZの中を書き直したときの記録）は書かない。バイト列を変えないので要らない。
- 研究ツールの`pack`は、既に`runs.zip`を持つ構成をまとめない（今の`pack_configuration`の「まとめ済みを飛ばす」）。

## 新しいmodule `packed_run_artifact_archive`
- `RUN_ARTIFACT_ARCHIVE_NAME = "runs.zip"`（公開の定数）。

### `pack_configuration_run_artifacts(configuration_directory: Path) -> str | None`
1つの構成をまとめる。まとめたら`None`、まとめなかったか削除が途中で失敗したら警告の文字列を返す。例外は出さない（`Exception`を捕まえて警告にする）。呼出し側は、`runs/`がある構成だけを渡す。
1. `runs/`がディレクトリでない・リンク、または`runs.zip`が既にある（リンクを含む）なら、警告を返す。
2. `runs/`の直下を列挙し、全てが通常のファイルで、名前が`<run>.json`と`<run>.npz`の組（同じ`<run>`の両方）であることを確かめる。組でない物（片方だけのrun、`.partial`などほかの名前、ディレクトリ）があれば警告を返す。
3. 残っている`runs.zip.partial`を消し、`runs.zip.partial`へ書く。各ファイルは1MBずつ読みながらsha256とバイト数を数え、`ZipFile.open(パス, "w", force_zip64=True)`へ書く。最後にmanifestを書き、fsyncする。
4. 読み戻して照合する。memberの集合がmanifestの項目と`archive_manifest.json`を合わせたものと同じこと、全memberを1MBずつ読んだsha256とバイト数が手順3の値と一致することを確かめる。合わなければ例外にする（手順6）。
5. `os.replace`で`runs.zip`へ置く。`PermissionError`のときは、開始の記録のまとめと同じく0.5秒ずつ最大8回やり直す（Windowsで作ったばかりのzipの置き換えが拒否されることがあるため）。その後、元のファイルを消し、`runs/`を消す。削除の途中の失敗は「まとめたが、一部の個別のファイルが残った」という警告にする（各ファイルは`runs/`かzipのどちらかにある）。
6. zipを置く前の`Exception`では、`.partial`を消して元を残し、「まとめなかった（理由）」を返す。

### `pack_completed_sweep_run_artifacts(output_directory: Path, configuration_names: tuple[str, ...]) -> RunArtifactPackingResult`
- 各構成について、`runs/`があれば`pack_configuration_run_artifacts`を呼ぶ（`runs/`が無い構成は飛ばす）。
- 結果は`RunArtifactPackingResult(packed_configuration_names: tuple[str, ...], warning_descriptions: tuple[str, ...])`（frozen dataclass）。
- 呼出し側が出力先の排他を持つ。

### 読み出しと戻し
- `read_packed_run_artifact_files(configuration_directory: Path, run_identifier: str) -> tuple[bytes, bytes] | None`
  - `runs.zip`のmanifestと、`runs/<run>.json`・`runs/<run>.npz`のmemberを読み、sha256がmanifestと合えば（要約, 記録の表）のバイト列を返す。zipに両方とも無ければ`None`。
  - 片方だけ・manifestと合わない・zipが壊れている（`zipfile.BadZipFile`、`lzma`の例外、`json`の失敗、`KeyError`など`OSError`以外の`Exception`）ときは`ValueError`にする。`OSError`はそのまま出す。
- `list_packed_run_identifiers(configuration_directory: Path) -> frozenset[str]`: manifestに要約と記録の表の両方があるrunの識別子。zipが読めなければ`ValueError`。
- `restore_packed_run_artifacts(configuration_directory: Path) -> None`
  - manifestの各項目について、パスが`runs/<名前>`で、名前に区切り文字・`..`が無いことを確かめる。
  - 戻す先が無ければ`<名前>.<uuid>.partial`へ書いてsha256を確かめてから`os.replace`する。あって中身のsha256がmanifestと同じなら何もしない（前の戻しが途中で止まった場合）。違えば`ValueError`。
  - 全項目が済んだら`runs.zip`を消す。読めないzip・照合の不一致は`ValueError`。
  - zipを消す前に失敗しても、戻し終えたファイルはmanifestと同じなので残す。

## 状態の判定（`find_existing_run_state`）
- 今の判定（`runs/<run>.json`・`.npz`のどちらかがあれば、今のまま）の後に、どちらも無く、構成のディレクトリに`runs.zip`があるときだけ、`read_packed_run_artifact_files`を呼ぶ。
  - `None`なら「無い」（`missing`）。
  - `ValueError`・`OSError`なら「食い違う」（`conflicting`）。説明は`existing packed run files cannot be read: <zipのパス> (<型>: <理由>)`。
  - 読めたら、`tempfile.TemporaryDirectory()`に`<run>.json`と`<run>.npz`を書き、今と同じ読込み関数（`load_*_run_artifacts(summary_file_path=...)`）と、今と同じ基準版・run設定の比べ方で判定する。説明のパスは`<zipのパス>!runs/<run>.json`。
  - 一時のディレクトリは出力先の外（`TMP`）なので、`--print-plan`の「出力先に何も作らない」は保たれる。
- これで、次の4つが、まとめた出力先でも今と同じに働く: `collect_behavior_baseline_mismatch_descriptions`、`execute_experiment_sweep`の判定、`find_sweep_resumption_candidates`（`find_state`として渡される）、比較scriptの`--print-plan`。
- zipはrunごとに開く（数百runで問題ない。実測は実装で記録する）。

## 計算の前の戻し（`execute_experiment_sweep`）
- 出力先の排他の中（内側の呼び出し）で、全runの状態を調べる前に、計画にある構成ごとに次を調べる。`runs.zip`があり、その構成の計画のrunのうち1つでも`list_packed_run_identifiers`に無く`runs/`にも無ければ、`restore_packed_run_artifacts`を呼ぶ。
- 戻しが`ValueError`・`OSError`で失敗したら、その構成の計画の全runを、計算せずに`failed`（説明は理由）とする。これらのrunは待ちに入れない。ほかの構成のrunは今と同じに扱う。
- 外側（排他の前）の基準版の確認は、読むだけなので、zipから判定する（戻さない）。
- 計画の全runがzipにあるときは戻さない（要求2.2。zip・`run_summary.csv`・`.resume/`を変えない）。

## 終わりのまとめ
- 掃引のコマンド（`_execute_command_in_output`）:
  - `run_summary.csv`を書いた後、掃引の記録を書く前に、`failed`が無く、表から外した読めない要約も無いときだけ、`pack_completed_sweep_run_artifacts(output_directory, sweep_plan.configuration_names)`を呼ぶ。コマンドは既に出力先の排他を持っている。
  - まとめた構成ごとに`packed run artifacts: <構成>/runs.zip`、警告ごとに`warning: <理由>`を表示する。終了コードは変えない。
- 比較script（`run_behavior_fix_comparison.py`）: 表を書いた後、同じ条件のとき、`hold_sweep_output_lock(output_directory)`の中で呼ぶ（掃引の排他は掃引の中で外れているため）。表示も同じ。
- 例外（中断を含む）で掃引が抜けた経路では、表もまとめも今と同じく行われない。

## 研究ツール
- `tools/research_results_archive/archive.py`の冒頭の説明に、次の2点を足す。処理は変えない。
  - 製品がまとめた`runs.zip`も`unpack`で戻せる。
  - 製品のmanifestは`arrays`を持たず、sha256で照合される。

## 文書
`docs/experiments/experiment-sweep.md`の出力の説明に、次を足す。
- `runs.zip`と、まとめる条件（失敗・読めない要約・中断が無い）。
- まとめた後の流し直し（全部飛ばす）と、runの追加（計算の前に戻し、終わりにまとめ直す）。
- 壊れたzipのときの扱い: その構成のrunが失敗になる。研究ツールの`unpack`も失敗するので、手で調べる。
- runのファイルを読む分析の前に、`python -m tools.research_results_archive.archive unpack <フォルダ>`で戻すこと。
- まとめの間は、構成のディレクトリの容量が一時的に約2倍要ること。

## Testing Strategy
- module（`test_packed_run_artifact_archive.py`）:
  - まとめた後、`runs/`が消え、zipの全memberのバイト列が元と同じで、manifestが元のパス・バイト数・sha256を持ち、要約が`ZIP_LZMA`・記録の表が`ZIP_STORED`であること。
  - 組でないファイル・片方だけのrun・ディレクトリ・既存の`runs.zip`・リンクでは、まとめず警告を返すこと。
  - 照合の失敗（読み戻しをmonkeypatch）・書き込みのOSError・置き換えの失敗で、元が残り`.partial`が残らないこと。残った`.partial`を消すこと。
  - `read_packed_run_artifact_files`が、zipに無いrunで`None`、片方だけ・sha256の不一致・壊れたzipで`ValueError`を返すこと。
  - `restore_packed_run_artifacts`で元に戻ること。既にある同じ中身のファイルは受け入れ、違う中身・不正なパスでは`ValueError`で、zipが残ること。
  - 研究ツールの`unpack_configuration`（`tools.research_results_archive.archive`をtestからimport）で、製品のzipが元のバイト列に戻ること。
- 状態の判定・掃引（`test_experiment_sweep.py`に足す）:
  - 2 runの小さい掃引をコマンドで流すと、構成ごとに`runs.zip`ができ、`run_summary.csv`が残ること。
  - 同じコマンドを流し直すと全runが`skipped_existing`で、出力先の全ファイル（パスとバイト列）が変わらないこと。
  - `--print-plan`が`completed`を表示し、出力先に何も作らないこと。
  - 再開の候補として見つかり、済んだ数が全runであること。
  - seedを足したコマンドで、戻してから足したrunだけを計算し、終わりに全runがzipにまとまり、表が全runの行を持つこと。
  - 1 runを失敗させた掃引、読めない要約がある掃引では、まとめないこと。
  - まとめの失敗（monkeypatch）で警告が出て、終了コードが変わらないこと。
  - 壊れたzip（バイト列を壊す）で、その構成のrunが計算されずに失敗になり、zipが変わらないこと。
  - zipの中の要約のrun設定が違うrunが「食い違う」になること。
  - 基準版の違う要約がzipにあるとき、今と同じく掃引を始めずに止まること。
- 比較script: 小さい比較の後に`runs.zip`ができること（既存の比較scriptのtestに足す）。
- 依存境界: 新しいmoduleの許可集合と、`experiment_sweep_execution.py`・`experiment_sweep_command.py`の追加を、実際にimportする名前で登録する。
- 共用のfresh smoke: 既存の掃引の2回実行（再開）の流れで、1回目の後に`runs.zip`ができ、2回目が全部飛ばされることを足す。
- 回帰: 全版のgolden、既存の成果物・掃引・再開のtest。全pytestは`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`で全部を流す。
