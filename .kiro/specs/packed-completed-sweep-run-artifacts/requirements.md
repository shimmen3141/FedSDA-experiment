# Requirements Document

## Project Description (Input)
- 困っている人: 研究室サーバ（uranium）で製品の掃引（`run_experiment_sweep.py`と、それを呼ぶ研究ツール）を流し、結果を手元へ転送して分析する研究者（ユーザー）。
- 今の状況: 2026-11-20のspec `lzma-record-tables-and-packed-run-start-records`で、runの記録の表をLZMAのNPZで書き、全runが済んだ掃引の開始の記録（`.resume`）を1つのzipへまとめた。runの成果物（`<構成>/runs/`のrunごとのJSONとNPZ、1 runあたり2つ）は対象外だったため、例えば520 runの掃引で1,040ファイルが残り、転送に時間がかかる（2026-11-23に、uraniumの結果で確かめた）。済んだ結果を`runs.zip`へまとめる研究ツール`tools/research_results_archive`はあるが、手で実行する必要がある。
- 変えたいこと: 全runが済み失敗が無い掃引の最後に、構成ごとのrunの成果物を1つのzipへまとめ、照合してから元のファイルを消す（ユーザーの依頼、2026-11-23）。中断したrunの安全な再開には、完了するまでまとめないことで影響させない（ユーザーの提案）。まとめた後に同じ掃引を流し直したときは、まとめたrunを済んだものとして扱って飛ばし、`run_summary.csv`を保つ。まとめた構成へ新しいrunを足すとき・再開先の候補の探索・重複の確認での扱いも決める。既存の研究ツール`tools/research_results_archive`（`runs.zip`と`archive_manifest.json`、NPZの中をLZMAへ書き直す形）との関係（同じ形式を使うか、`unpack`で戻せるか）を決める。
- 作業場所: `.worktrees/single-client`（ブランチ`research/single-client`）。

## Boundary Context
- In scope: 掃引のコマンド（`run_experiment_sweep.py`。それを呼ぶ研究ツールを含む）と比較script（`run_behavior_fix_comparison.py`）の終わりの、構成ごとのrunの成果物のまとめ。比較の集計script（`analyze_behavior_fix_comparison.py`）が、まとめた構成を黙って飛ばさないこと。まとめた出力先での済んだrunの判定・再開の候補の探索・同じ出力先へのrunの追加。研究ツール`tools/research_results_archive`の`unpack`で戻せること。文書。
- Out of scope: 製品の外の分析のツール（`tools/`の研究ツールの多く）を、まとめたzipから読めるようにすること（`unpack`してから使う）。単一ストリームのパッケージ（`single_stream_hard_assignment_soft_prediction`）の出力。旧Pareto用の`run_pareto_sweep.py`とその重複の確認（`tools/experiments/manifests.py`）。既存の結果の書き直し（研究ツールが扱う）。runの値・記録の形式。
- Adjacent expectations: 開始の記録のまとめ（spec `lzma-record-tables-and-packed-run-start-records`の要求2）は今のまま。runの記録の表は、そのspecでLZMAのNPZになっている。

## Requirements

### Requirement 1: 済んだ掃引のrunの成果物のまとめ
**Objective:** 研究者として、全runが済んだ掃引の結果を、構成ごとに1つのファイルで受け取りたい。サーバからの転送のファイルの数を減らすため。

#### Acceptance Criteria
1. When 掃引のコマンドまたは比較scriptで、計画の全runの`run_status`が`completed`か`skipped_existing`（`failed`が無い）で、`run_summary.csv`と掃引の記録を書き終えたとき, the sweep shall 出力先の排他を持ったまま、計画にある各構成のディレクトリの`runs/`の全ファイルを、その構成のディレクトリの`runs.zip`へまとめる。
2. The sweep shall zipの中に、元のファイルを`runs/<ファイル名>`のパスで入れ、元の各ファイルのパス・バイト数・sha256を記録したmanifest（`archive_manifest.json`）を入れる。runの記録の表（NPZ）はバイト列を変えずに入れ、要約（JSON）は圧縮して入れる。
3. When zipを書き終えたとき, the sweep shall zipを読み戻して全ファイルのバイト列が元と一致することを確かめてから、zipを置き、その後で元のファイルと空になった`runs/`を消す。一致しなければ、zipを置かずに元を残す。
4. If `runs/`に、runの要約と記録の表の組になっていないファイル（片方だけのrun、ほかの名前のファイル、ディレクトリ）がある、または、`runs.zip`が既にあり`runs/`もある（下の要求3の追加の途中を除く）, then the sweep shall その構成をまとめずに残し、理由を警告として報告する。
5. If 失敗したrunがある、`run_summary.csv`から外した読めない要約がある、掃引が例外（中断を含む）で終わった、または、`--print-plan`のとき, the sweep shall runの成果物をまとめない（今と同じに残す）。
6. If まとめの途中で読み書き・照合の失敗（容量不足などのOSError、照合の不一致）が起きたら, the sweep shall 作りかけのzipを消し、元のファイルを残し、警告を報告して、掃引の終了コードは変えない。元のファイルの削除の途中で止まっても、各ファイルは個別のファイルかzipのどちらかに残る。
7. The sweep shall まとめた構成の数と、警告を、進み具合の報告に出す。`run_summary.csv`・README・掃引の記録・`.resume/`は、まとめずに残す。

### Requirement 2: まとめた出力先での流し直しと再開の候補
**Objective:** 研究者として、まとめた後も、同じ掃引のコマンドを流し直したときに計算し直さず、結果の表も壊れないでほしい。

#### Acceptance Criteria
1. When まとめた構成（`runs/`が無く`runs.zip`がある）の計画のrunを調べるとき, the sweep shall zipの中の要約と記録の表から、今の個別のファイルと同じ基準（基準版・run設定の一致、読めること）で、済んだrun・食い違うrun・基準版の違うrunを判定する。zipに無いrunは「無い」とする。
2. When まとめた出力先（`runs/`が無い）で、同じ掃引を流し直し、全runがzipの中で済んでいるとき, the sweep shall 全runを飛ばし（`skipped_existing`）、zip・`run_summary.csv`・`.resume/`を変えない（README・掃引の記録は今と同じく書く）。
5. While 構成のディレクトリに`runs.zip`が残っているとき（戻しが途中で失敗した場合を含む）, the sweep shall その構成の`run_summary.csv`を書き直さない（一部だけ戻った`runs/`から表を作り直して行を失わないため）。
3. The resumption shall まとめた出力先も、今と同じく、再開の候補として見つけ、済んだrunの数を数える（`--print-plan`の表示を含む）。
4. If zipが壊れていて読めない、またはmanifestと中身が合わない, then the sweep shall そのzipのrunを「食い違う」として扱い、計算せずに失敗として報告する（zipを書き換えない）。

### Requirement 3: まとめた構成へのrunの追加
**Objective:** 研究者として、まとめた出力先に、seedなどを足した掃引を流せるようにしたい。

#### Acceptance Criteria
1. When まとめた構成の計画に、zipにも`runs/`にも無いrunがあるとき、または、`runs/`と`runs.zip`が両方あるとき（前の戻し・まとめが途中で止まった場合）, the sweep shall runの計算を始める前に、出力先の排他を持ったまま、zipの全ファイルを`runs/`へ戻し、manifestのsha256と照合してからzipを消す。その後は、今と同じに、無いrunを計算し、済んだrunを飛ばす。
2. If zipが読めない、戻す先に同じ名前で中身の違うファイルがある、または、戻したファイルがmanifestと合わない, then the sweep shall zipを消さず、その構成の計画の全runを、計算も飛ばし（`skipped_existing`）もせずに、理由を付けて失敗として報告する（ほかの構成のrunは今と同じに扱う。戻し終えたファイルはmanifestと同じなので残してよい）。
3. When 追加した掃引の全runが済んだとき, the sweep shall 要求1と同じに、戻したrunと新しいrunを合わせて、もう一度まとめる。

### Requirement 4: 研究ツールとの互換
**Objective:** 研究者として、製品がまとめたzipを、既存の研究ツールで戻せるようにしたい。

#### Acceptance Criteria
1. The archive shall 研究ツール`tools/research_results_archive`の`unpack`で、元のパスとバイト列のまま戻せる形にする。
2. The research tool shall 製品がまとめた構成を、`pack`で二重にまとめない（今の「まとめ済みを飛ばす」のとおり）。

### Requirement 5: 既存の値の維持と検証
**Objective:** 研究者として、runの値と既存の結果の扱いが変わらないことを確かめたい。

#### Acceptance Criteria
1. The goldens shall 全版のgoldenの値を変えない。`BEHAVIOR_BASELINE_VERSION`は上げない。
2. The tests shall 次を確かめる: (i) 全runが済んだ掃引で、構成ごとに`runs.zip`ができ、`runs/`が消え、zipの全ファイルのバイト列が元と同じ。(ii) 失敗したrunがある掃引・読めない要約がある掃引・中断した掃引（例外を注入）ではまとめず、`runs/`が残る。(iii) 照合の失敗・書き込みのOSErrorで元が残り、終了コードが変わらない。(iv) まとめた出力先で同じ掃引を流し直すと全runが飛ばされ、zip・`run_summary.csv`・`.resume/`のバイト列が変わらない。(v) まとめた出力先が再開の候補として見つかる。(vi) zipに無いrunを足すと、戻してから計算し、終わりにまとめ直す。`runs/`とzipが両方ある出力先でも、戻してから続ける。(vii) 壊れたzipでは、その構成のrunが計算されずに失敗になり、zipと`run_summary.csv`が変わらない。(viii) 研究ツールの`unpack`で元に戻り、研究ツールの`pack`が製品のまとめた構成を飛ばす。(ix) 比較scriptでもまとまる。(x) 比較の集計scriptが、まとめた構成で止まる。

### Requirement 6: 文書
**Objective:** 研究者として、まとめた結果の扱い方を知りたい。

#### Acceptance Criteria
1. The docs shall `docs/experiments/experiment-sweep.md`の出力の説明に、`runs.zip`、まとめる条件、流し直しと追加の扱い、runのファイルを読む分析の前に研究ツールの`unpack`で戻すことを足す。
2. The docs shall 研究ツール`tools/research_results_archive`の冒頭の説明に、製品がまとめたzipも`unpack`で戻せることを足す。
3. If 比較の集計script（`analyze_behavior_fix_comparison.py`）が、`runs/`が無く`runs.zip`がある組合せを見つけたら, then the script shall 集計を始めずに、研究ツールの`unpack`で戻すよう示して止める（黙って0件として集計しない）。
