# Research

## 今の実装（2026-10-14、bbb3c67）
- コードの版は、`describe_code_revision`（`sweep/experiment_sweep_records.py`）が、コマンドの実行ごとの記録（`sweep_record_*.json`。比較scriptは`comparison_record_*.json`）にだけ書く。1 runの要約（3手法の`*_run_artifact_files.py`。形式の版1）は、run設定・指標・環境・表のファイルの名前だけを持ち、moduleの説明に「来歴（commitほか）は、ここでは扱わない」とある。
- 済んだrunの判定（`find_existing_run_state`）は、要約と表の両方があり、要約のrun設定が計画と同じなら`completed`（飛ばす）。runの識別子は、run設定の全部のhashを含むので、既定の値が変われば、識別子も変わる（例: 案Yの既定で`sine2_interval3_seed0_72f2b5b47a95`→`..._ecef3c8ebd73`）。したがって、run設定で表せる変更では、古い結果を誤って飛ばすことは無い。漏れるのは、設定を変えずに動作が変わる変更（基準版が上がる変更の一部。例: 設定の項目を足す前の修正、計算の順の変更）。
- 期待する動作の基準版は、testのhelper（`tests/refactoring/behavior_golden_cases.py`の`BASELINE_VERSION`）にだけあり、srcからは読めない（srcがtestをimportしないため）。そこで、srcへ定数を置き、testで一致を確かめる。
- 追跡しないファイルを数えないので、gitに入っていない新しいsourceを使っても「変更なし」になる。文書では「コードの版を完全に表す」とは書かない。
- 未commitの変更の有無は、`git status --porcelain`で、追跡しないファイルを含む。メインの作業ツリーには、コミット判断を保留した3文書（追跡しない）が常にあり、そこで実行すると、いつも「変更あり」になる。
- 要約を読むもの: 掃引の表（JSONを直接読む）、再開の判定（3手法の読込み）、比較の分析（`run_summary.csv`を`DictReader`で読み、決まった指標の列だけを使う。列が増えても影響しない）。

## 判断
- 基準版の違いで、掃引の全体を始める前に止める（runごとの失敗にしない）。ユーザーが2026-10-14に「違えば、警告して止める」案を選んだため。run設定の食い違い（`conflicting`）は、基準版が同じ要約に限り、今までどおり、そのrunだけの失敗とする。
- 基準版は3手法で共通の1つ（goldenの版が共通のため）。FedDrift・FedAvgの結果も、版が上がれば再利用しない（安全側）。
