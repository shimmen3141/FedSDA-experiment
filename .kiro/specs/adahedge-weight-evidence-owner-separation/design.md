# Design Document

## Overview
AdaHedgeの証拠の計算の部品と、全標本の証拠のownerを、手法の層（`methods/fedsda/prediction_combination/`）へ移す。真の概念別の証拠は、評価の層の、別のownerに分ける。`runtime/`は、これまで1つだった引数（`adahedge_weight_evidence_collection`）を、全標本の側のowner（全部の受取り先）と、真の概念別の側のowner（更新・再始動する処理と、その通り道だけ）に分けて受け渡す。計算・順序・設定は変えない。

### Goals / Non-Goals
- Goals: 予測に使いうる重みの出どころを、手法の層にする。真の概念を使う比較用の証拠が、手法の判断（警報の処理、候補検証の進行、証拠の再始動）へ渡らないことを、引数の形で示す。値を変えない。
- Non-Goals: 証拠の計算の変更、処理の順序の変更、設定・記録の表・指標の名前の変更、Fixed-Shareの部品の変更、提案構成で、全標本の証拠を計算しないようにする最適化。

## Boundary Commitments
- This Spec Owns: 下の3つのmodule（1つは移動だけ、2つは、前の1つを分けたもの）、それを受け取る`runtime/`の引数とclientのowner、対応するtest、依存境界の登録。
- Out of Boundary: `AdaHedgeWeightEvidence`の中身（1文字も変えない。置き場所だけ）、`FixedShareWeightController`、設定、固定旧。
- Revalidation Triggers: 予測に使う重みの出どころを増やすとき（別の重みの方式）、真の概念別の診断を増やすとき。

## Architecture

### いま
- `evaluation/adahedge_weight_evidence.py`: `AdaHedgeWeightEvidence`（1つの証拠。累積損失、gap、再始動・再較正の計数。標準ライブラリだけに依存）。
- `evaluation/adahedge_weight_evidence_collection.py`: `AdaHedgeWeightEvidenceCollection`（`evidence_is_updated_by_prediction`、`all_sample_weight_evidence`、`get_true_concept_diagnostic_evidence(true_concept_id=...)`、`created_true_concept_ids`）。
- 受け取るmodule（`runtime/`）と、使う口:

| module | 使う口 |
|---|---|
| `observed_sample_prediction.py` | 全標本の証拠（観測前の重み、観測後の更新）、真の概念別の証拠（同じ） |
| `post_aggregation_prediction_recalibration.py` | 全標本の証拠の再生、真の概念別の証拠の再始動（`created_true_concept_ids`の全部） |
| `training_assignment_all_sample_adahedge_weight_evidence_restart.py` | `evidence_is_updated_by_prediction`、全標本の証拠の再始動 |
| `alarm_occurrence_handling.py`、`held_candidate_validation_progress.py` | 上の再始動へ渡すだけ |
| `observed_sample_processing.py` | 予測・警報の処理・候補検証の進行へ渡すだけ |
| `fedsda_run_client.py` | 組立て（`FedsdaRunClientOwners`）と、標本1件の処理・再較正への受渡し |
| `fedsda_run_metric_derivation.py`、`fedsda_prediction_diagnostic_metric_derivation.py`、`fedsda_run_record_tables.py` | 全標本の証拠の、再較正の計数の読取り |

### これから

| module | 中身 | 依存 |
|---|---|---|
| `methods/fedsda/prediction_combination/adahedge_weight_evidence.py` | `AdaHedgeWeightEvidence`（前のmoduleを、そのまま移す。差分は、置き場所だけ） | 標準ライブラリだけ |
| `methods/fedsda/prediction_combination/all_sample_adahedge_weight_evidence_holder.py` | `AllSampleAdaHedgeWeightEvidenceHolder`（`evidence_is_updated_by_prediction`、`all_sample_weight_evidence`。前の集まりから、真の概念別の2つの口を除いたもの） | 上のmoduleだけ |
| `evaluation/true_concept_adahedge_diagnostic_evidence_collection.py` | `TrueConceptAdaHedgeDiagnosticEvidenceCollection`（`get_true_concept_diagnostic_evidence(true_concept_id=...)`、`created_true_concept_ids`。前の集まりの、真の概念別の2つの口。動作は同じ——IDの型の検査の後、初回だけ作る。作成順を保つ） | 手法の層の`AdaHedgeWeightEvidence`（型と、生成） |

前の2つのmoduleは、削除する（別名を残さない。証拠の部品のmoduleは、task 1で、移すのと同時に。前の集まりのmoduleは、task 3で）。

証拠の再始動は、役割を変えずに、moduleと関数の名前へ`all_sample`を足す（`runtime/training_assignment_all_sample_adahedge_weight_evidence_restart.py`の`restart_all_sample_adahedge_weight_evidence_after_training_assignment_change`。前の名前は残さない）。

評価の層から、手法の層の部品を読む向きの依存が、1つ増える（真の概念別の診断は、手法と同じ計算を、標本を分けて行うもの。計算の部品を、2か所に置かない）。手法の層から、評価の層を読む依存は、作らない（いまも無い）。

### 引数の受渡し

| module | 全標本の側（引数`all_sample_adahedge_weight_evidence_holder`） | 真の概念別の側（引数`true_concept_adahedge_diagnostic_evidence_collection`） |
|---|---|---|
| `observed_sample_prediction.py`（予測と重みの更新） | 受け取る | 受け取る |
| `post_aggregation_prediction_recalibration.py` | 受け取る | 受け取る |
| `observed_sample_processing.py` | 受け取る（予測・警報の処理・候補検証の進行へ渡す） | 受け取る（予測へ渡すだけ） |
| `alarm_occurrence_handling.py`、`held_candidate_validation_progress.py`、`training_assignment_all_sample_adahedge_weight_evidence_restart.py` | 受け取る | 受け取らない |
| `fedsda_run_client.py` | `FedsdaRunClientOwners`のfield。組立ての引数（`evidence_is_updated_by_prediction`）は、これまでと同じ式 | `FedsdaRunClientOwners`のfield。引数なしで作る |
| 指標・記録の表の3つのmodule | `run_client.owners.all_sample_adahedge_weight_evidence_holder.all_sample_weight_evidence`を読む | 読まない |

### 変えないもの（確かめる点）
- 予測の中の順序: 全標本の証拠の観測前の重み→Fixed-Shareの観測前の重み→（真の概念IDがあれば）真の概念別の証拠の取得と観測前の重み→…→観測後の更新。ownerを2つに分けても、呼ぶ順は、行の並びを変えない。
- 再較正の中の順序: `created_true_concept_ids`の読取り→全標本の証拠の再生→真の概念別の証拠の再始動→Fixed-Shareの再生。
- ownerの型の検査: 予測と標本1件の処理は、引数名→型の表（`_REQUIRED_OWNER_TYPES_BY_ARGUMENT_NAME`）で、最初の状態の更新より前に確かめる。表の、前の1行を、全標本の側の行へ替え、その直後へ、真の概念別の側の行を足す（ほかの行の順は変えない）。再較正・警報の処理・候補検証の進行・再始動の、個別の検査も、同じ位置で、新しい型へ替える。不正な型で出る例外の型（`TypeError`）は同じ。文言は、引数名と型名が変わる。
- 真の概念別の証拠は、乱数を使わず、全標本の証拠・Fixed-Shareの重み・予測・学習へ、値を渡さない（いまも同じ）。ownerを分けても、ほかの状態は変わらない。
- 設定・保存するrun設定・runの識別子・記録の表の列名・指標の名前は、変えない。
- 全標本の側のownerの、組立ての引数の検査（`evidence_is_updated_by_prediction`は、builtinのboolだけ。`TypeError`）は、前の集まりと同じ。真の概念別の側のownerの、IDの型の検査（builtinのintだけ）も同じ。
- 予測と標本1件の処理は、ownerを、引数名→ownerの辞書へ入れてから、型の表で確かめる。新しい引数は、表と、辞書の、両方へ足す（片方だけだと、`TypeError`でなく`KeyError`になる）。再較正の、個別の型の検査にも、真の概念別の側の行を足す（足さないと、不正なownerが、`AttributeError`になる）。不正なownerを、最初の更新より前に、`TypeError`で拒否することを、引数ごとのtestで確かめる。
- 再較正は、これまでどおり、`created_true_concept_ids`で得たIDの証拠を、`get_true_concept_diagnostic_evidence`で取り出して、再始動する（既にあるIDなので、新しく作らない）。専用の口は、足さない。

## Testing Strategy
- 部品のtest（`test_adahedge_weight_evidence.py`）: `AdaHedgeWeightEvidence`のtestは、importの場所だけを替える。集まりのtestは、2つのownerのtestへ分ける。task 1では、新しい2つのownerを、前の集まり（同じ操作の列を与えた、動作の正解）と比べるtestも書く（同じIDの列で、作成順と、IDごとの証拠の値が同じ。全標本の証拠の値が同じ）。この比較は、前の集まりを削除するtask 3で、前の集まりから読み取った期待（作成順、初回だけ作る、IDごとに独立）を、直接確かめるtestへ直して、残す。集まりの動作を確かめるtestは、ほかのファイルにもある（`test_training_assignment_adahedge_weight_evidence_restart.py`の、集まりの状態を読むhelperなど）。削除せずに、新しい2つのownerへ移す。
- （上の続き）集まりのtestの中身（全標本の側: 引数の型の拒否、同じ証拠を返すこと、真の概念別の口が無いこと。真の概念別の側: IDの型の拒否、初回だけ作ること、作成順、IDごとに独立、全標本の証拠が無いこと）。
- 受け取る処理のtest: 引数と、状態の読取り（testの、状態のsnapshot）を、2つのownerへ替える。実旧との対照（標本1件の処理、client、全体run、構成、再較正）は、期待値を変えずに通る。型の拒否のtest（不正なownerを、最初の更新より前に拒否）へ、真の概念別の側のownerを足す。
- 依存境界のtest（`tests/refactoring/test_single_run_dependency_boundaries.py`）: 前の名前を持つ箇所の全部を、新しい置き場所と名前へ替える（下の表。行は、`7ac8e16`でのもの）。手法の層の2つのmoduleが、評価の層と`runtime/`をimportしないことは、moduleごとの許可集合で確かめられる。

| 箇所 | いま | これから | 替えるtask |
|---|---|---|---|
| moduleごとの許可集合（`dependency_is_allowed`の分岐。5326〜5331行の、前の証拠の部品） | 前の証拠の部品（標準ライブラリの2つのmoduleから、3つの名前） | 手法の層の証拠の部品（同じ3つの名前） | 1 |
| 同（5317〜5320行の、前の集まり） | 評価の層の証拠の部品だけを読む | task 1で、手法の層の証拠の部品だけを読む、へ替える。task 3で、分岐ごと削除する | 1、3 |
| 同（新しい2つのowner） | 無い | 全標本の側のowner（手法の層の証拠の部品だけ）、真の概念別の側のowner（手法の層の証拠の部品だけ）。分岐は、既存と同じ、`return imported_module_name in (...)`の1文の形で書く | 1 |
| 同（5321〜5325行の、再始動） | 前の集まりと、変更の記録 | 全標本の側のownerと、変更の記録（moduleの名前も替える） | 2 |
| 受け取る`runtime/`のmoduleの許可集合（6043・6100・6165・6197・6245・6299行の、前の集まり。6271・6300行の、再始動の関数） | 前の集まり、前の関数名 | 全標本の側のowner。予測・再較正・標本1件の処理・clientは、真の概念別の側のownerも。再始動の、新しいmoduleと関数名 | 2 |
| importの形を厳しく確かめるmoduleの一覧（7210〜7212行、7331〜7333行） | 前の2つのmoduleと、再始動 | task 1: 前の証拠の部品を、手法の層の証拠の部品へ替え、新しい2つのownerを足す（前の集まりは、残す——一覧から外すと、importの解決の仕方が変わり、許可集合と合わなくなる）。task 2: 再始動のmoduleの名前を替える。task 3: 前の集まりを外す | 1、2、3 |
| `math`を許す例外（7433行） | 前の証拠の部品 | 手法の層の証拠の部品 | 1 |
| importの文ごとの期待の表（17〜67行、118〜138行、718〜778行）と、moduleの契約のtest（12229〜12345行） | 前の2つのmoduleと、その読み手 | 証拠の部品の行（67行、778行、契約のうち、証拠の部品のもの）は、task 1で、新しい場所へ。前の集まりと、再始動の行（17〜62行、118〜138行、718〜738行、契約のうち、集まりと再始動のもの）は、task 2で、新しい2つのownerと、新しい名前の再始動へ（同じ形の期待を、新しい名前で）。前の集まりのmoduleだけに関する行は、task 3で削除する。1つのparametrizeの中に、taskの違う行が並ぶ箇所は、実装のときに、行ごとに、替えたtaskを、実装の記録へ書く | 1、2、3 |

- 要求2.2・4.4の確認: 警報の処理（`handle_alarm_occurrence`）、候補検証の進行（`held_candidate_validation_progress.py`の入口の関数）、証拠の再始動（`restart_all_sample_adahedge_weight_evidence_after_training_assignment_change`）の引数の一覧（`inspect.signature`）に、真の概念別の側のownerの引数が無いことを、testで確かめる。真の概念が、真の概念に依る診断だけを変え、ほかの状態・指標を変えないことは、既存の全体runのtest（`test_fedsda_stream_protocol_run.py`の`test_true_concepts_affect_only_concept_dependent_diagnostics_of_whole_run`）が、期待を変えずに通ることで確かめる。
- 重要動作のgolden（基準版1〜3）、構成の照合、掃引のrunの識別子の期待値: ファイルと期待値を変えずに通る（全pytest）。
- fresh processのsmoke: 組立てを、2つのownerへ替える。

## Risks
- 受渡しの替え漏れ（片方のownerを、別の場所で作り直してしまい、状態が分かれる） → clientのownerは、それぞれ1か所で作る。実旧との対照（真の概念別の診断の列を含む、記録の表の照合）と、基準版のgoldenで検出できる。
- 型の検査の順が変わり、複数の不正がある入力で、出る例外が変わる → 表の行の位置を保つ。既存の拒否のtest（引数ごと）が、最初の更新より前の拒否を確かめる。
- task 2が大きい（`runtime/`の10のmodule、testの十数ファイル、fresh processのsmoke、依存境界の登録を、一度に替える） → 分けない理由は、ownerが、clientの1か所で作られ、全部の処理が同じものを受け取ること（一部だけを先に替えると、同じ証拠を、前の集まりと新しいownerの両方が持つ状態を、途中に作る。前の集まりから新しいownerへ委譲する、一時的な仕組みを入れれば分けられるが、消すためだけのコードを足すことになるので、採らない）。変更は、引数とfieldの名前の付替えが中心で、moduleごとに、対象のtestで確かめながら進め、最後に、まとめて通す。
- 名前の変更が広い（testを含めて、十数ファイル） → 機械的な置換の後、`grep`で、前の名前（`adahedge_weight_evidence_collection`、`AdaHedgeWeightEvidenceCollection`、`evaluation.adahedge_weight_evidence`、`evaluation/adahedge_weight_evidence`、`training_assignment_adahedge_weight_evidence_restart`、`restart_adahedge_weight_evidence_after_training_assignment_change`——新しい名前の一部として現れる箇所は、除く——）が残っていないことを確かめる。範囲は、`src`、`tests/refactoring`、直下のscript、`docs/`（このspecと、IMPROVE-013の記録が、経緯として書く箇所を除く）。過去のspec（`.kiro/specs/`の、ほかのfeature）と、引継ぎの履歴は、当時の記録なので、替えない。再始動のtestのファイル名（`tests/refactoring/test_training_assignment_adahedge_weight_evidence_restart.py`）も、moduleの名前に合わせて替える（task 2）。
