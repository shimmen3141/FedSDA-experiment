# Design Document

## Overview
全runが済んだ掃引の最後に、構成ごとの`runs/`（runごとの要約のJSONと記録の表のNPZ）を、その構成のディレクトリの`runs.zip`へまとめる。まとめた出力先では、runの状態の判定をzipから読み、計画にzipに無いrunがあるときだけ、計算の前にzipを`runs/`へ戻す。runの値・乱数の消費・記録の形式は変えない。開始の記録のまとめ（spec `lzma-record-tables-and-packed-run-start-records`）は今のまま。

## Boundary Commitments
- 持つもの: 新しい`sweep/packed_run_artifact_archive.py`（まとめ・zipからの読み出し・戻し）、`sweep/experiment_sweep_execution.py`の`find_existing_run_state`（zipから判定）と`execute_experiment_sweep`（計算の前の戻し）、`sweep/experiment_sweep_command.py`と`run_behavior_fix_comparison.py`の終わりのまとめの呼び出しと`run_summary.csv`を書き直す条件、`analyze_behavior_fix_comparison.py`のまとめた構成での停止、依存境界の許可リスト、共用のfresh smoke、文書（`docs/experiments/experiment-sweep.md`、研究ツール`tools/research_results_archive/archive.py`の冒頭の説明）。
- 持たないもの: runの計算と成果物の書き出し（`*_run_artifact_files.py`）、読込み関数、開始の記録、掃引の計画、`run_summary.csv`の書き方、研究ツールの処理（説明の文だけ足す）、製品の外の分析のツール。
- 許す依存: 新しいmoduleは標準の`hashlib`・`json`・`os`・`time`・`uuid`・`zipfile`・`pathlib`・`dataclasses`だけ（torch・numpy・ほかの製品のmoduleを読まない）。`experiment_sweep_execution.py`は新しいmoduleの公開の名前と`tempfile`を足す。
- 重複: `experiment_sweep_resumption`の非公開の`_is_link`・`_replace_with_retries`（`PermissionError`のやり直し）と同じ処理を、新しいmoduleに非公開で持つ（非公開の名前は他のmoduleから読めない規約のため。共通化は今回しない）。
- 掃引のコマンドでの計画の違う掃引: 今のコマンドは、排他の中で`write_sweep_resumption_plan`を呼び、出力先の`.resume/sweep.json`と計画が違えば止める（`ValueError`、終了コード2）。この規則は変えない。そのため、zipに無いrunを足す戻しが働くのは、`execute_experiment_sweep`を直接呼ぶ経路（比較scriptは`sweep.json`を書かない）と、`runs/`とzipが両方ある途中の状態から同じ計画で続ける場合である。
- 研究ツールでまとめたzip: NPZの中をLZMAへ書き直しており、manifestのsha256は元のNPZのバイト列なので、製品のsha256の照合では合わない。製品は、manifestの項目に`arrays`がある（研究ツールの形）ことを見て、「研究ツールでまとめたzipなので、`unpack`してから流し直す」という理由で読めない（食い違う）とする。研究ツールの形を製品で読む処理は持たない。
- 再検証が要る変更: まとめた後の出力先は`runs/`を持たない。`runs/`を直接読む製品の外のツール（`tools/`の多く）は、研究ツールの`unpack`の後に使う（文書に書く）。製品の中（`src/`とリポジトリのルートのscript）で`runs/`を読むのは次の6つ。
  - 開始の記録を書く前の確認（`write_run_start_record`。そのrunのJSON・NPZが`runs/`に無いこと）: 計算するrunは、戻しの後（`runs/`がある状態）か、zipに無いrunなので、今のままでよい。
  - 状態の判定: この設計で扱う。
  - `run_summary.csv`の書き出し: `runs/`が無ければ書き直さない（今と同じ条件）。
  - 再開の候補の探索: 状態の判定を通る。
  - 中断したrunの直し: `runs/`にNPZだけが残るrunだけが対象で、まとめた構成には無い。
  - 比較の集計script（`analyze_behavior_fix_comparison.py`）: 組合せごとの表は`run_summary.csv`から読むので、まとめても残る。短い警報の連鎖の節（`--chains`）だけが`runs/<run>.npz`を読み、無いと黙って飛ばす（0件になる）。そこで、`--chains`のときだけ、集計の入口で、読む全ての組合せのディレクトリを調べ、1つでも`runs/`が無く`runs.zip`がある組合せがあれば、何も読む前に、その組合せと`unpack`のコマンドを標準エラーへ出して終了コード2で止める。`--chains`の無い集計は今のまま。

## zipの形式（研究ツールの`unpack`と互換）
- 置き場所: `<出力先>/<構成>/runs.zip`。zipの中のパスは`runs/<元のファイル名>`。
- 要約（`.json`）は`ZIP_LZMA`で入れる。記録の表（`.npz`。製品がLZMAで書いたNPZ）は、`ZIP_STORED`でバイト列をそのまま入れる。
- `archive_manifest.json`（`ZIP_LZMA`）: `{"format_version": 1, "files": [{"path": "runs/<名前>", "byte_count": <バイト数>, "sha256": "<hex>"}, ...]}`（パスの昇順。キーは研究ツールと同じ）。
  - 研究ツールの`unpack_configuration`は、`arrays`の無い項目をsha256で照合するので、そのまま戻せる。
  - 研究ツールのmanifestにある`arrays`（NPZの中を書き直したときの記録）は書かない。バイト列を変えないので要らない。
- 研究ツールの`pack`は、既に`runs.zip`を持つ構成をまとめない（`_process`の「まとめ済み」の飛ばし。`pack_configuration`もzipがあれば`FileExistsError`）。

## 新しいmodule `packed_run_artifact_archive`
- `RUN_ARTIFACT_ARCHIVE_NAME = "runs.zip"`（公開の定数）。

### `pack_configuration_run_artifacts(configuration_directory: Path) -> tuple[bool, str | None]`
1つの構成をまとめる。(zipを置いたか, 警告の文字列か`None`)を返す。zipを置いた後に元の削除が途中で失敗したときは(`True`, 警告)、まとめなかったときは(`False`, 理由)。例外は出さない（`Exception`を捕まえて警告にする）。呼出し側は、`runs/`がある構成だけを渡す。
1. `runs/`がディレクトリでない・リンク、または`runs.zip`が既にある（リンクを含む）なら、警告を返す。
2. `runs/`の直下を列挙し、全てが通常のファイルで、名前が`<run>.json`と`<run>.npz`の組（同じ`<run>`の両方）であることを確かめる。組でない物（片方だけのrun、`.partial`などほかの名前、ディレクトリ）があれば警告を返す。`runs/`が空なら、何もせずに(`False`, `None`)を返す。
3. 残っている`runs.zip.partial`を消し、`runs.zip.partial`へ書く。各ファイルは、`ZipInfo(パス)`の`compress_type`に要約なら`ZIP_LZMA`、記録の表なら`ZIP_STORED`を指定し、`ZipFile.open(zipinfo, "w", force_zip64=True)`へ、1MBずつ読みながら書く。書きながらsha256とバイト数を数える。最後にmanifestを書き、fsyncする。
4. 読み戻して照合する。memberの集合がmanifestの項目と`archive_manifest.json`を合わせたものと同じこと、全memberを1MBずつ読んだsha256とバイト数が手順3の値と一致することを確かめる。手順3の値は、元のファイルを読んだバイト列から数えたものなので、これで要求1.3の「元のファイルのバイト列との一致」を確かめたことになる（まとめの間は出力先の排他を持つので、元が書き換わらない）。合わなければ例外にする（手順6）。
5. `os.replace`で`runs.zip`へ置く。`PermissionError`のときは、開始の記録のまとめと同じく0.5秒ずつ最大8回やり直す（Windowsで作ったばかりのzipの置き換えが拒否されることがあるため）。その後、元のファイルを消し、`runs/`を消す。削除の途中の失敗は「まとめたが、一部の個別のファイルが残った」という警告にする（各ファイルは`runs/`かzipのどちらかにある）。
6. zipを置く前の`Exception`では、`.partial`を消して元を残し、「まとめなかった（理由）」を返す。

### `pack_completed_sweep_run_artifacts(output_directory: Path, configuration_names: tuple[str, ...]) -> RunArtifactPackingResult`
- 各構成について、`runs/`があれば`pack_configuration_run_artifacts`を呼ぶ（`runs/`が無い構成は飛ばす）。
- 結果は`RunArtifactPackingResult(packed_configuration_names: tuple[str, ...], warning_descriptions: tuple[str, ...])`（frozen dataclass）。zipを置いた構成は、削除の警告があっても`packed_configuration_names`に入る。
- 呼出し側が出力先の排他を持つ。

### 読み出しと戻し
- `read_packed_run_summary_and_record_table(configuration_directory: Path, run_identifier: str) -> tuple[bytes, bytes] | None`
  - `runs.zip`のmanifestと、`runs/<run>.json`・`runs/<run>.npz`のmemberを読み、sha256がmanifestと合えば（要約, 記録の表）のバイト列を返す。zipに両方とも無ければ`None`。
  - 片方だけ・manifestと合わない・zipが壊れている（`zipfile.BadZipFile`、`lzma`の例外、`json`の失敗、`KeyError`など`OSError`以外の`Exception`）ときは`ValueError`にする。`OSError`はそのまま出す。
  - `lzma`が無い環境（zipfileがLZMAのmemberで`RuntimeError`を出す）は、`ValueError`の説明を「LZMAが使えないので読めない」と分ける（壊れたzipと区別するため）。掃引のコマンドは`_output_lock_held=True`で呼ぶので、`execute_experiment_sweep`の`ensure_lzma_compression_available`を通らない。そのため、この変換は、読み出し・一覧・戻しの3つ全てで同じにする（`OSError`以外の`Exception`を`ValueError`へ。`RuntimeError`も外へ出さない）。
  - manifestの項目に`arrays`があれば（研究ツールの形）、照合の前に「研究ツールでまとめたzipなので、`unpack`してから流し直す」という`ValueError`にする。
- `list_packed_run_identifiers(configuration_directory: Path) -> frozenset[str]`: manifestに要約と記録の表の両方があるrunの識別子。zipが読めなければ`ValueError`。
- `restore_packed_run_artifacts(configuration_directory: Path) -> None`
  - manifestの各項目について、パスが`runs/<名前>`で、名前に区切り文字・`..`が無いことを確かめる。
  - 構成のディレクトリ・`runs/`・`runs.zip`がリンクなら`ValueError`（`experiment_sweep_resumption`の`_safe_path`・`_is_link`と同じ確認を、非公開で持つ）。`runs/`が無ければ作る。
  - 始める前に、`runs/`の中の、manifestの名前に一致する`<名前>.*.partial`（前の戻しが中断で残したもの）を消す。
  - 戻す先が無ければ`runs/<名前>.<uuid>.partial`へ書いてsha256を確かめてから、まとめと同じやり直し付きの`os.replace`で置く。途中で失敗したら、その`.partial`を消してから例外を出す（`.partial`が残ると次のまとめの組の確認で警告になるため）。戻す先があって中身のsha256がmanifestと同じなら何もしない（前の戻しが途中で止まった場合）。違えば`ValueError`。
  - 全項目が済んだら`runs.zip`を消す。読めないzip・照合の不一致は`ValueError`。
  - zipを消す前に失敗しても、戻し終えたファイルはmanifestと同じなので残す。

## 状態の判定（`find_existing_run_state`）
- 今の判定（`runs/<run>.json`・`.npz`のどちらかがあれば、今のまま）の後に、どちらも無く、構成のディレクトリに`runs.zip`があるときだけ、`read_packed_run_summary_and_record_table`を呼ぶ。
  - `None`なら「無い」（`missing`）。
  - `ValueError`・`OSError`なら「食い違う」（`conflicting`）。説明は`existing packed run files cannot be read: <zipのパス> (<型>: <理由>)`。例外を外へ出さないので、再開の候補の探索（候補ごとの`OSError`・`ValueError`で、その候補を一覧から落とす）でも、壊れたzipの構成は候補に残り、そのrunは「食い違う」として数えられる。
  - 読めたら、`tempfile.TemporaryDirectory()`に`<run>.json`と`<run>.npz`を書き、今と同じ読込み関数（`load_*_run_artifacts(summary_file_path=...)`）と、今と同じ基準版・run設定の比べ方で判定する。説明のパスは`<zipのパス>!runs/<run>.json`。
  - 一時のディレクトリは出力先の外（`TMP`）なので、`--print-plan`の「出力先に何も作らない」は保たれる。
- 1つのrunの判定は、zipを開き、要約と記録の表を読んで一時のディレクトリへ書き、読込み関数で読む。判断の基準: 520 runのまとめた出力先で`--print-plan`が2分を超えたら、1回の呼び出しの中で構成ごとにzipを1回だけ開く形に直す（今回の実装で測って決める）。
- これで、次の4つが、まとめた出力先でも今と同じに働く: `collect_behavior_baseline_mismatch_descriptions`、`execute_experiment_sweep`の判定、`find_sweep_resumption_candidates`（`find_state`として渡される）、比較scriptの`--print-plan`。
- zipはrunごとに開く。1回の掃引で、同じrunのzipを、排他の前の基準版の確認・排他の中の判定・（コマンドの）再開の候補の探索で何度か開く。520 runのまとめた出力先で、`--print-plan`と流し直しの所要時間を実装で測り、tasks.mdに記録する。

## 計算の前の戻し（`execute_experiment_sweep`）
- 出力先の排他の中（内側の呼び出し）で、全runの状態を調べる前に、計画にある構成ごとに次を調べる。`runs.zip`があり、(a) `runs/`もある（前の戻し・まとめが途中で止まった）、または、(b) その構成の計画のrunのうち1つでも`list_packed_run_identifiers`に無く`runs/`にも無いなら、`restore_packed_run_artifacts`を呼ぶ。(a)では、同じ中身のファイルを受け入れるので、重複も消える。
- `list_packed_run_identifiers`と戻しの両方を1つの`try`に入れる。`ValueError`・`OSError`で失敗したら、その構成の計画の全runを`failed`（説明は理由）として記録し、その後の状態のループ（済んだrunを`skipped_existing`、無いrunを待ちにする処理と、中断したrunの直し）の対象から外す。これで、一部が戻った`runs/`にあるrunも飛ばされず、zipに無いrunも計算されない。ほかの構成のrunは今と同じに扱う。例外は`execute_experiment_sweep`の外へ出さない（掃引のコマンドの使い方の誤り（終了コード2）にしない）。
- 外側（排他の前）の基準版の確認は、読むだけなので、zipから判定する（戻さない）。
- 優先順位: `runs/`がある構成は、計画の全runがzipにあっても戻す（上の(a)。終わりにまとめ直すので、zipは作り直される）。`runs/`が無く、計画の全runがzipにあるときだけ戻さない（要求2.2。zip・`run_summary.csv`・`.resume/`を変えない）。

## 終わりのまとめ
- 掃引のコマンド（`_execute_command_in_output`）:
  - `run_summary.csv`を書き直すのは、今の条件（`runs/`がある）に加えて、`runs.zip`が無い構成だけにする（要求2.5。戻しが途中で失敗した構成の表を保つ）。
  - `run_summary.csv`と掃引の記録を書いた後、結果の件数を表示する前に、`failed`が無く、表から外した読めない要約も無いときだけ、`pack_completed_sweep_run_artifacts(output_directory, sweep_plan.configuration_names)`を呼ぶ。コマンドは既に出力先の排他を持っている。
  - まとめた構成ごとに`packed run artifacts: <構成>/runs.zip`、警告ごとに`warning: <理由>`を表示する。終了コードは変えない。
- 比較script（`run_behavior_fix_comparison.py`）: 表を書き直す条件はコマンドと同じ。表と掃引の記録を書いた後、同じ条件のとき、`hold_sweep_output_lock(output_directory)`の中で呼ぶ（掃引の排他は掃引の中で外れているため）。表示も同じ。排他を取れない（別のprocessが同じ出力先を使っている。`hold_sweep_output_lock`の`ValueError`、または`OSError`）ときは、まとめずに警告を表示し、終了コードは変えない。表の書き出しとまとめの間に別のprocessが入る隙間は、今の比較scriptの表の書き出しと同じ性質で、この変更では塞がない（同じ出力先を同時に使わない運用）。
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
- `runs/`とzipが両方ある構成（戻し・まとめが途中で止まった）は、研究ツールの`unpack`では戻せない（戻す先にファイルがあると止まる）。同じ掃引のコマンドを流し直すと、製品が戻してまとめ直す。
- 研究ツールでまとめたzipの構成で掃引を流し直す前には、研究ツールの`unpack`で戻す（製品は読めないので、そのrunは失敗になる）。
- 掃引のコマンドで、同じ出力先へ計画の違う掃引を流すことは、今と同じくできない。
- 掃引のまとめの途中に、排他を取らない`--print-plan`や再開の候補の探索を動かすと、一時的に「食い違う」と見えることがある。
- runのファイルを読む分析の前に、`python -m tools.research_results_archive.archive unpack <フォルダ>`で戻すこと。
- まとめの間は、構成のディレクトリの容量が一時的に約2倍要ること。

## Testing Strategy
- module（`test_packed_run_artifact_archive.py`）:
  - まとめた後、`runs/`が消え、zipの全memberのバイト列が元と同じで、manifestが`format_version`と元のパス・バイト数・sha256を持ち、要約が`ZIP_LZMA`・記録の表が`ZIP_STORED`であること。戻り値が(`True`, `None`)であること。元の削除の失敗（monkeypatch）で(`True`, 警告)になること。
  - 組でないファイル・片方だけのrun・ディレクトリ・既存の`runs.zip`・リンクでは、まとめず警告を返すこと。
  - 照合の失敗（読み戻しをmonkeypatch）・書き込みのOSError・置き換えの失敗で、元が残り`.partial`が残らないこと。残った`.partial`を消すこと。
  - `read_packed_run_summary_and_record_table`が、zipに無いrunで`None`、片方だけ・sha256の不一致・壊れたzipで`ValueError`を返すこと。
  - `restore_packed_run_artifacts`で元に戻ること。既にある同じ中身のファイルは受け入れ、違う中身・不正なパスでは`ValueError`で、zipが残ること。
  - 研究ツールの`unpack_configuration`（`tools.research_results_archive.archive`をtestからimport）で、製品のzipが元のバイト列に戻ること。研究ツールの`_process`の`pack`が、製品のまとめた構成を「まとめ済み」として飛ばすこと。研究ツールの`pack_configuration`でまとめたzipを、製品が「研究ツールでまとめたzip」という理由で読めないとすること。
  - 戻しの前に、前の中断で残った`.partial`が消えること。リンクの`runs/`・zipで`ValueError`になること。`lzma`が無い状態（monkeypatch）で、読み出し・一覧・戻しが`ValueError`（説明がLZMA）になること。
- 状態の判定・掃引（`test_experiment_sweep.py`に足す）:
  - 2 runの小さい掃引をコマンドで流すと、構成ごとに`runs.zip`ができ、`run_summary.csv`が残ること。
  - 同じコマンドを流し直すと全runが`skipped_existing`で、`runs.zip`・`run_summary.csv`・`.resume/`の全ファイルのバイト列が変わらないこと（掃引の記録は新しく増え、READMEは今と同じく書かれる）。
  - `--print-plan`が`completed`を表示し、出力先に何も作らないこと。
  - 再開の候補として見つかり、済んだ数が全runであること。
  - `execute_experiment_sweep`を直接呼び、seedを足した計画で、戻してから足したrunだけを計算し、終わりに（比較scriptと同じ呼び方の）まとめで全runがzipにまとまること。
  - 掃引のコマンドで、まとめた出力先へ計画の違う掃引を流すと、今と同じく拒否される（終了コード2、zipが変わらない）こと。
  - `runs/`とzipが両方ある出力先（戻しの途中で止まった状態を作る）で流し直すと、計画の全runがzipにあっても戻してから続け、終わりにまとまること。
  - 進み具合の報告に、まとめた構成と警告が出ること（要求1.7）。`--print-plan`がまとめず、`runs/`を消さないこと。
  - 1 runを失敗させた掃引、読めない要約がある掃引、例外（中断）を注入した掃引では、まとめず、`runs/`が残り`runs.zip`が無いこと。
  - まとめの失敗（monkeypatch）で警告が出て、終了コードが変わらないこと。
  - 壊れたzip（バイト列を壊す）で、その構成のrunが計算されずに失敗になり（終了コード1。使い方の誤りの2ではない）、zipと`run_summary.csv`が変わらないこと。
  - 戻しが途中で失敗した構成（`runs/`に同じ名前で違う中身のファイルを置く）で、`runs/`にあるrunが`skipped_existing`にならず、zipに無いrunが計算されず、全runが`failed`になること。
  - `lzma`が無い状態（monkeypatch）で、掃引のコマンドの経路でも、まとめた構成の判定の説明が壊れたzipと区別され、例外が外へ出ないこと。
  - 再開の候補の探索で、壊れたzipの構成が候補に残り、そのrunが「食い違う」として数えられること。
  - zipの中の要約のrun設定が違うrunが「食い違う」になること。
  - 基準版の違う要約がzipにあるとき、今と同じく掃引を始めずに止まること。
- 比較script: 小さい比較の後に`runs.zip`ができること。排他を取れないとき、まとめずに警告を出し終了コードが変わらないこと（既存の比較scriptのtestに足す）。
- 比較の集計script: `--chains`のとき、まとめた構成を見つけると、集計せずに`unpack`を示して終了コード2で止まること。`--chains`の無い集計は、まとめた構成でも今と同じ出力になること。
- 既存のtestの書き換え: 掃引のコマンド・比較scriptを流した後に`runs/`を読む既存のtestは、まとめの後は`runs/`が無いので、test用の補助（`restore_packed_run_artifacts`で戻してから読む）を通す形に直す。対象（`runs/`の読み取りと`main(`などの呼出しを突き合わせて洗い出した）: `tests/refactoring/`の`fresh_process_smoke.py`、`test_behavior_fix_comparison_scripts.py`、`test_dynamic_weighted_majority_experiment_sweep.py`、`test_experiment_sweep.py`、`test_experiment_sweep_resumption.py`、`test_fedavg_experiment_sweep.py`、`test_fedavg_local_updates_per_aggregation.py`、`test_feddrift_experiment_sweep.py`、`test_fedsda_fixed_share_time_scale_axis.py`、`test_fedsda_pending_assignment_fifo_capacity_axis.py`。実装で、各testの期待（`runs/`のファイル名・数）を変えずに、読み方だけを変える。単一ストリームのパッケージのtest（`tests/single_stream_hard_assignment_soft_prediction/`）は対象外（そのコマンドは別）。
- 依存境界: 新しいmoduleの許可集合と、`experiment_sweep_execution.py`・`experiment_sweep_command.py`の追加を、実際にimportする名前で登録する。
- 共用のfresh smoke: 既存の掃引の2回実行（再開）の流れで、1回目の後に`runs.zip`ができ、2回目が全部飛ばされることを足す。
- 回帰: 全版のgolden、既存の成果物・掃引・再開のtest。全pytestは`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`で全部を流す。
