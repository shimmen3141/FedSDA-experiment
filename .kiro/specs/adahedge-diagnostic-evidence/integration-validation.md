# 統合検証

## 現在の状態

Task 1・2承認済み。Windows基準の全回帰とTask 3、別sessionのfeature最終GOは未実施。

Task 2独立Luna（medium）は対象42件と新規のexact依存注入22件の64 passed、fresh process、対象source/testのRuffを再実行した。変異全ログ、元byteとHEAD一致、等価分類を照合しAPPROVED。既存依存suite全体・全pytest・Pyright・goldenは独立再実行していない。

## 要求と証拠の対応

| 要求 | 部品での検証 |
| --- | --- |
| 1.1 | 初期状態、独立owner、readonly状態の実旧対照 |
| 1.2 | 初回・取得のみの集合変更・直接updateによる集合変更と計数 |
| 1.3 | ID順序の置換、有限/無限学習率、最小値の同率、単一モデル |
| 2.1 | 実AdaHedgeRouterとの重み・累積損失・gapの厳密一致。制限損失、ゼロ重み、任意の受理重みを含む |
| 2.2 | 証拠の消去と再始動計数、集合変更計数の保持、空状態を含む繰返し再始動 |
| 2.3 | 独立owner、Python/NumPy/torch乱数全状態の不変、exact AST依存境界 |
| 3.1 | 不正ID・bool・空・重複の状態変更前拒否、負ID受理 |
| 3.2 | 型・非有限・overflow・重み範囲/総和・ID不対応・入力読取り例外で4状態不変 |
| 3.3 | 入出力copy、返却loss/weightを変更した後の状態不変 |
| 4.1 | 実旧を差し替えず正常列を対照。丸め・近似・新実装をoracleにする代用なし |
| 4.2 | 固定旧差分、Windows全pytest/JUnit、旧11・最終3goldenをTask 3で確認する |

Task 1はWindows基準の対象と依存境界で2860 passed、変更3ファイルのRuff成功、Pyright 0 errors / 0 warnings。Task 1独立担当は実測logを読取り照合し、test再実行はしていない。Task 2のtest追加後の最終結果は下記へ記録する。

## 測定環境と成果物

Windows基準のPythonは`../../venv/Scripts/python.exe`。OpenMP/MKLは1 thread、MNISTは`../../data/mnist`、matplotlibは既存`../../venv/matplotlib-cache`、TMP/TEMPは`../../venv/refactoring-tests/adahedge-diagnostic-evidence`を使用する。SAC保護設定・venv・goldenは変更しない。

実測log・変異script/report・fresh script・JUnitは`../../venv/refactoring-tests/`とその`adahedge-diagnostic-evidence/`配下（このPCのGit管理外）。別PCで追試する場合は、対象commitをcheckoutし、上の環境で対象testと依存境界testを実行する。変異はtasks.mdの各検査/演算を1つずつ変え、対象testの失敗を確認した後に原byteへ復元する。

## 検証範囲

このspecは単一AdaHedge診断証拠ownerだけを扱う。通知の時機・複数ownerへの配布、保存診断の集計全体、集約後再較正、検出episode、client進行、新全体runのgolden一致は未検証で後続specへ残る。Linux用goldenも作成していない。

## Task 2の主担当側の実測

Windows Python 3.13.15で変異24種を測定。非等価20種はすべて対応testの失敗で検出し、collection失敗は0。公開契約内で結果が変わらない4種は未検出のまま理由を記録し、非等価の分母へ含めない。空検査を消しても後段のmin(empty)がcommit前にValueErrorを出す。入力copy2種はvalidatorが別dictを作り、同集合候補copyは成功後commit用の算出値だけに使うため、逐次公開操作では差がない。callbackによる入力変更・再入・並行更新・資源不足はこの等価分類の対象外。実sourceでは設計どおりのcopy手順を維持する。

原byteのSHA-256は`76169fb139fb28448bcadae7fcbe62282c1c59cd3d818de28cd4cfc0a68760cd`。各変異後と最後にbyte復元を確認し、HEADのsourceとも一致した。復元後は対象42件と依存境界2821件の2863 passed（5.13s）。追加3件はMapping以外のlist-of-pairs2条件と、総和1でも範囲外の重み1条件。既存拒否test内で計算例外前の未commitも確認する。新しい束縛名は追加していない。

fresh processの5操作が成功し、旧実装・test・torch・numpy・pytestをimportしないこと、新規依存が対象packageとstdlibだけであることを実測した。証拠は`mutations.py`、`report.json`、`mutation-<変異名>.log`、`restored-suite.log`、`fresh_smoke.py`、`fresh-smoke.log`。理由の文字化けを修復した`classify_report.py`も保存する。

| Mutation | Detected | Classification |
| --- | --- | --- |
| id_type | True | non_equivalent |
| empty_ids | False | equivalent_within_public_contract |
| duplicate_ids | True | non_equivalent |
| numeric_type | True | non_equivalent |
| numeric_finite | True | non_equivalent |
| numeric_overflow | True | non_equivalent |
| loss_mapping_type | True | non_equivalent |
| weights_mapping_type | True | non_equivalent |
| id_sets | True | non_equivalent |
| weight_range | True | non_equivalent |
| weight_sum | True | non_equivalent |
| sort_ids | True | non_equivalent |
| gap_formula | True | non_equivalent |
| pool_count | True | non_equivalent |
| restart_count | True | non_equivalent |
| restart_clear | True | non_equivalent |
| output_copy | True | non_equivalent |
| loss_input_copy | False | equivalent_within_public_contract |
| weight_input_copy | False | equivalent_within_public_contract |
| candidate_loss_copy | False | equivalent_within_public_contract |
| get_validation_after_sync | True | non_equivalent |
| update_validation_after_sync | True | non_equivalent |
| get_commit_before_calculation | True | non_equivalent |
| update_commit_before_calculation | True | non_equivalent |
