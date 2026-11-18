# Design Document

## Overview
**Purpose**: FedHASPの1クライアントの純粋な形を、Hoeffding treeを学習器にして、単一ストリームの既存手法と同じ条件で動かすパッケージ`single_stream_hard_assignment_soft_prediction`を作る。
**Users**: FedHASPの仕組みを、単一ストリームのオンライン分類の既存手法と比べる研究者。
**Impact**: 製品（`federated_learning_experiments`）は変えない。新しいパッケージは`src/`の下の別の最上位パッケージで、製品の判断の部品を一方向にimportする。

### Goals
- 論文の4.2節（柔らかい予測のうち局所の版の混合）・4.3節（固い割り当て）の手順を、Hoeffding treeで実装する。
- 既存手法と同じ条件（事前学習なし、最初の標本から評価、標本ごとに1回学習、既定値）で、書き出した系列を処理する。
- 判断の部品は製品のコードをそのまま使い、同じアルゴリズムであることを部品の共有で示す。

### Non-Goals
- 神経網の学習器、連合学習、集約、統合、同期の版の混合。
- 製品の全体の流れとの出力の一致の照合（判断の部品の共有と、流れのtestで代える）。
- 新しいデータの生成、比較手法の追加と再実行（`river_baselines.py`の事前学習なしの版を含む）、結果の集計。

## Boundary Commitments

### This Spec Owns
- `src/single_stream_hard_assignment_soft_prediction/`の全module。
- このパッケージのtest（`tests/single_stream_hard_assignment_soft_prediction/`）と、Riverを入れた実行環境の要件のファイル`requirements-single-stream.txt`。

### Out of Boundary
- 製品の全module。再利用する判断の部品の挙動（製品のtestが保証）。
- RiverのHoeffding treeの挙動。研究ツール（`tools/research_single_client_stream_baselines/`）。

### Allowed Dependencies
- 製品から（どれもtorchを読み込まないことを、WSLのRiver環境で確かめた。research.md）:
  - `methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring.OverallAndTrueClassLossMonitor`
  - `methods.fedsda.loss_change_detection.loss_change_detection_settings`（`LossChangeDetectionSettings`と方針の名前の定数）
  - `methods.fedsda.prediction_combination.fixed_share_weights.FixedShareWeightController`
  - `methods.fedsda.prediction_combination.prediction_combination_settings.PredictionCombinationSettings`
  - `methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection`（`select_assignment_by_minimum_description_length`、`select_new_model_learner`、`compute_switching_code_length`、`MinimumDescriptionLengthAssignmentSelection`、学習者の名前の定数）
  - `methods.fedsda.loss_statistics.loss_baseline_selection.select_loss_monitoring_baseline_mean_loss`
  - `learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore`
  - `data.dataset_definitions.get_dataset_definition`
- 外部: `river`（0.27.0）、`numpy`、標準ライブラリ。
- 製品は新しいパッケージをimportしない。

### Revalidation Triggers
- 再利用する製品の部品の公開の関数・引数・挙動が変わったとき。
- 製品のFedHASPの既定（下の設定の一覧で「製品と同じ」としたもの）が変わったとき（設定の一致のtestが落ちる）。

## Architecture

### 論文の手順との対応（1クライアント・同期なし）

| 論文の手順（4章） | 製品 | このパッケージ |
|---|---|---|
| モデル: 共有の特徴抽出器＋adapter＋出力層 | 神経網 | Hoeffding tree（共有部なし）。確率を一様分布と0.01で混ぜて平滑化 |
| 予測: 全モデルをFixed-Shareの重み$w_m$で混ぜ、同期の版と局所の版を$v$で混ぜる | 両方（既定は`mixture_of_synchronized_and_local_parameters`） | 局所の版だけ（製品と違う点1） |
| 重みの更新: 誤り$e_m=1-p_m(x)_y$でFixed-Share（T=120、ηは累積の分散から） | `FixedShareWeightController` | 同じ部品。保有モデルの集合が変わると、この部品が重みを一様に戻す（新しいモデルを作った警報の後）。集合が変わらない警報では重みを保つ |
| 監視: 現在のモデルの誤りをe-detectorで、全体とクラスごとに監視（α=0.001、低下で基準を下げる） | `OverallAndTrueClassLossMonitor`（事前学習の5分割交差検証の統計で開始） | 同じ部品と設定。200件のウォームアップの後、現在のモデルの損失の統計で開始（製品と違う点5） |
| 帰属の保留: FIFO 30、容量を超えた最古の標本を現在のモデルへ | 保留の部品（torchの標本の型と組） | 同じ規則を、このパッケージの小さいFIFOで持つ |
| 警報: 合図があり、min(n,k)≥5（待つ方針） | 同じ | 同じ |
| MDL: (i)そのまま、(ii)保有モデルへ、(iii)新しいモデル（keep・resetを新しい順に1件ずつ予測→学習、短い方＋ln 2）、$L_{sw}=\ln T+\ln$（帰属先の候補の数＋1） | 符号長の計算はtorch、比較は`select_assignment_by_minimum_description_length` | 符号長は木で計算し、比較は同じ部品。$L_{sw}$のMは製品と同じ「帰属先の候補の数＋1」。保有モデルはすべて損失の統計を持つので、これは保有モデルの数（現在のモデルを含む）と等しく、論文の式$M=|\mathcal{M}|$と同じ |
| 選んだ後: 前のn−k件を現在のモデル、後のk件を帰属先へ。FIFOを空に、帰属先の損失の統計で監視を始め直す | 同じ | 同じ |
| 新しいモデル: 区間の損失が最小の保有モデルを初期値に、区間で数エポック学習 | 候補の学習 | MDLで測った学習者そのもの（製品と違う点3） |
| 学習: 毎標本、各モデルのデータストアからミニバッチを作り、全モデルを1回更新 | 再生のミニバッチ | 割り当てた時点に、その標本でそのモデルを1回学習（製品と違う点2） |
| 集約・統合・集約の後の重みの作り直し | あり | 無し |

### 製品と違う点（論文・設計に明記する）
1. **予測は局所の版だけ。** 製品の既定は同期の版と局所の版を混ぜるが、同期の無い1クライアントでは同期の版が使われない。根拠: 1クライアント×5,000標本・間隔5000で、製品のFedHASPと「同期の版を混ぜない」構成（`fedsda-local-parameter-prediction`）の精度が全300 runで一致した（`docs/research/fedhasp-single-client-pilot-v16.md`の「結果（2026-11-17、主担当）」の節）。
2. **学習は割り当てた時点の1回だけ。** 既存手法の「標本ごとに1回」に合わせる。製品の再生のミニバッチ（毎標本、全モデルのデータストアから標本を取り直して学習する）は行わない。
3. **新しいモデルは、MDLで測った学習者そのもの。** 警報の時点の空の木（reset）か現在のモデルの写し（keep）に、後のk件を新しいものから順に1件ずつ学習させる。これはMDLの前置きの和を測ったときの学習者の、k件目までの状態と同じになる。製品は、区間の損失が最小の保有モデルの写しから数エポック学習するが、木は数エポックの学習をせず、過去のコンセプトで作った分岐を捨てられないため、測った学習者をそのまま使う。
4. **共有の特徴抽出部とadapterが無い。** 学習器は木1本。
5. **事前学習の代わりに、最初の200件をウォームアップにする。** この間は監視も警報もせず、予測は評価に含める。200はRiverの`HoeffdingTreeClassifier`の既定のgrace period（WSLのriver 0.27.0で確かめた）と同じで、木が最初の分岐を考えるまでの件数。
6. **確率を一様分布と0.01で混ぜて平滑化する。** 木は確率0を出しうるので、MDLの対数損失を有限にする。最大のクラスは変わらない。

### 設定の一覧（`SingleStreamHardAssignmentSoftPredictionSettings`）

| field | 値 | 製品との関係 |
|---|---|---|
| `pending_assignment_buffer_capacity_samples` | 30 | 製品と同じ |
| `minimum_change_interval_sample_count` | 5 | 製品と同じ |
| `fixed_share_weight_redistribution_time_scale_samples`（T） | 120 | 製品と同じ。`switching_probability=1/T`。`L_sw`の「帰属先の候補の数＋1」は製品と同じで、保有モデルの数（現在のモデルを含む）と等しく、論文4.3節の$M=|\mathcal{M}|$と同じ。論文4章の実装の注記（「保持モデルの数+1」）の方がずれているので、論文側の注記を直す |
| `loss_change_detection_settings` | 製品の既定の`LossChangeDetectionSettings`（e_sr、`overall_and_true_class_losses`、α=0.001、`lower_baseline_on_loss_decrease`、待つ方針、ADWINの値は使わない） | 製品と同じ（全field） |
| `loss_monitor_betting_fractions` | (0.05, 0.1, 0.2, 0.4, 0.8) | 製品と同じ |
| `loss_monitor_maximum_retained_candidate_count` | 1000 | 製品と同じ |
| `prediction_combination_settings` | `PredictionCombinationSettings`。このパッケージで働くのは`prediction_combination_strategy`（`fixed_share_weighted_prediction`）とTだけ | 働くfieldは製品と同じ。`prediction_model_parameter_source`・集約の後の作り直しは使わない（製品と違う点1、集約なし） |
| 新しいモデルの学習者 | resetとkeepの2つ、変化区間の探索は1件刻みの逆順の1本の流れ | 製品と同じ |
| `warm_up_sample_count` | 200 | このパッケージだけ（製品と違う点5） |
| `probability_smoothing_weight` | 0.01 | このパッケージだけ（製品と違う点6） |
| 木のハイパーパラメータ | Riverの既定（grace_period 200、delta 1e-7、leaf_prediction nba、nb_threshold 0） | このパッケージだけ |

要求7.3のtestは「製品と同じ」の行だけを、製品の既定の組立て（`build_final_configuration_fedsda_run_settings`、torchを読み込むのでWindowsの固定venvで実行）から取り出した値と比べる。

### 1標本の処理の順（`SingleStreamHardAssignmentSoftPredictionProcessor.process_sample`）
製品の`process_observed_sample`の順（予測→重みの更新→監視→保留→警報の判定→割り当て・学習）と同じにする。
1. **予測**: 保有モデル（作った順）の平滑化した確率を求め、`get_weights_before_label_observation(model_ids=保有の順)`の重みで混ぜ、最大のクラス（同点は小さい番号）を予測とする。
2. **重みの更新**: 各モデルの監視用の損失と、1の重み（ラベル観測前の重み）を渡して`update_weights_after_loss_observation`を呼ぶ（製品の`observed_sample_prediction.py`の呼び方と同じ）。
3. **監視**: ウォームアップ中は監視の部品を呼ばず、記録の対数e値の欄だけをNaNにする。ウォームアップの後なら、現在のモデルの監視用の損失、正解クラス、標本の番号、`select_loss_monitoring_baseline_mean_loss(loss_moments=現在のモデルの統計の`overall_loss_moments`)`を渡して`observe_loss_after_label_observation`を呼ぶ。ウォームアップ中は呼ばない（対数e値はNaN）。
4. **保留**: 標本をFIFOへ加える。
5. **警報の判定**: 検出器が合図すれば、FIFOの列W（n件）で`select_alarm_interval_assignment`を呼ぶ。min(n, k)<5なら警報としない（`too_short`を記録）。変化区間の探索はk≥5なので、実質はn<5（警報の直後で保留が少ない）のときだけ起きる。このときは7へ進み、検出器は始め直さないので、次の標本で合図が続けば、その標本を最後に含むWで選択をやり直す。
6. **警報**: 要求5.5の(a)〜(d)の順。
7. **警報でない**: FIFOが容量を超えていれば、最古の標本を現在のモデルへ割り当てる（統計に記録→学習）。
8. **ウォームアップの終わり**: 200件目の処理の最後に、現在のモデルの統計から選んだ基準値で監視の部品を作る。

### MDLの符号長（`select_alarm_interval_assignment`）
- Wの各標本について、全保有モデルのMDL用の損失を求める（保有モデルは変えない）。
- 帰属先の候補は、現在のモデル以外で、損失の統計を持つ保有モデル（保有の順）。
- 新しいモデル: n≥5なら、Wを逆順にし、resetの学習者（空の概念モデル）とkeepの学習者（現在のモデルの写し）それぞれに、「MDL用の損失を足す→学習」を1件ずつ行い、前置きの和の列を作る。k=5..nについて`select_new_model_learner`で短い方＋ln 2と学習者の名前を得る。
- `compute_switching_code_length(switching_probability=1/T, destination_count=帰属先の候補の数+1)`。
- `select_assignment_by_minimum_description_length(..., change_interval_grid_sample_count=5, change_interval_step_sample_count=1)`。
- 戻り値: 製品の`MinimumDescriptionLengthAssignmentSelection`（そのまま）。新しいモデルは、要求5.5(a)で、選ばれた学習者と同じ作り方で作り直す。「測った学習者のk件目の状態」は、同じ初期状態（警報の時点の空の木か現在のモデルの写し）から、Wの新しい順のk件を学習し終えた状態と定義する。前置きの和のk件目の値は、その学習の前の予測で測る値なので、状態とは1件ずれる。学習者の状態を保存しないので、同じ手順で再現する。

### 指標（`compute_prequential_metrics`）
- prequential精度: 正解の割合。
- Kappa: (p0−pc)/(1−pc)。pcは予測クラスと観測クラスの周辺分布の積の和。
- Kappa-M: (p0−pm)/(1−pm)。pmは「その標本より前に最も多く観測したクラス」（観測が無いときと同点のときは、小さいクラス番号）を予測する分類器のprequential精度。
- Kappa-Temporal: (p0−pt)/(1−pt)。ptは「直前の標本の観測クラス」（最初の標本はクラス0）を予測する分類器のprequential精度。
- 分母が0のときはNaN。
- 真の変化点の後の精度: 真のコンセプトが変わった標本から、次の変化点までの最初の30・100・500件（区間がそれより短ければ区間の全部）について、全変化点の該当標本をまとめた正解の割合（標本数で重み付け）。
- 概念の同定（警報としたものだけ）: 各モデルの概念は、最後の時点でそのモデルに割り当てた全標本の最多の真のコンセプト（同点は小さい番号。割り当てた標本の無いモデルは概念なし）。各警報の「新しい概念」は、Wの末尾k件（警報の標本を含む）の最多の真のコンセプト（同点は小さい番号）。分類: `reuse`は、帰属先の概念＝新しい概念なら正しい再利用、違うか概念なしなら誤った再利用。`new`は、その警報の時点で保有していたモデル（現在のモデルを含む。その警報で作る新しいモデルは含まない）の中に、最後の時点の概念が新しい概念と同じモデルがあれば「再来なのに新規作成」、無ければ「初めての概念での新規作成」。`stay`はそのまま。

## Components and Interfaces

| module | 公開するもの | 役割 |
|---|---|---|
| `single_stream_settings.py` | `SingleStreamHardAssignmentSoftPredictionSettings`、`build_default_single_stream_settings` | 設定の型と既定値。riverとtorchをimportしない |
| `hoeffding_tree_concept_model.py` | `HoeffdingTreeConceptModel` | 木を1つ持ち、平滑化した確率、監視用・MDL用の損失、1件の学習、写し、空の新しいモデル、節点の数 |
| `alarm_interval_assignment.py` | `select_alarm_interval_assignment` | 列Wで符号長を求め、製品の比較の関数で選ぶ。保有モデルを変えない |
| `single_stream_processing.py` | `SingleStreamHardAssignmentSoftPredictionProcessor`（公開メソッド`process_sample`）、`SampleProcessingRecord`、`AlarmRecord` | 1標本の処理と記録 |
| `single_stream_run.py` | `run_single_stream`、`SingleStreamRunResult`、`compute_prequential_metrics` | 系列1本の処理、指標、経過時間 |
| `single_stream_artifacts.py` | `save_single_stream_run`、`load_single_stream_run` | JSON・NPZの保存と読込み（上書きしない、pickleなし） |
| `single_stream_command.py` | `main` | 系列のファイルを並列に処理し、成果物と一覧のCSVを書く |

## File Structure Plan
- 新規（src）: `src/single_stream_hard_assignment_soft_prediction/__init__.py`と上の7 module。
- 新規（test）: `tests/single_stream_hard_assignment_soft_prediction/`に、moduleごとのtestと境界のtest。riverを使うtestは`pytest.importorskip("river")`で、riverが無い環境（Windowsの固定venv）では飛ぶ。設定の一致のtestは製品の既定の組立て（torch）を使い、Windowsで動く（WSLのRiver環境ではtorchが無いので飛ぶ）。
- 新規（環境）: `requirements-single-stream.txt`（river==0.27.0、numpy）。WSLの`~/river-env`で、`PYTHONPATH=src`にして実行する。
- 変えない: 製品の全ファイル、`pytest.ini`。

## Error Handling
- 系列のNPZ・JSONの欠け（必要な配列・fieldが無い、長さの不一致、`array_sha256`の不一致、ラベルが0以上K未満でない、Kが決まらない）は、処理の前に`ValueError`。
- 設定の範囲外（容量<1、最小の変化区間<1、ウォームアップ<1、平滑化が0以下か1以上、T≤1）は、設定の作成時に`ValueError`。
- 成果物の保存先に同名のファイルがあれば`FileExistsError`（書かない）。

## Testing Strategy
- 設定: 既定の「製品と同じ」の値が製品の既定と同じ（Windows）。範囲外の拒否。
- 概念モデル: 空の木は一様分布、平滑化で最大のクラスが変わらない、和が1、木が返さないクラスが0から平滑化される、損失の式、写しが独立に学ぶ、新しいモデルが空、特徴の変換が`river_baselines._as_feature_dictionary`と同じ（WSL）。
- MDL: 手で作った列で、(a)現在のモデルが合うと`stay`（k=n）、(b)保有モデルが合うとそのモデルへ、(c)どれも合わないと新しいモデル、(d)保有モデルの状態が変わらない、(e)前置きの和が逆順の1件ずつの「損失を足す→学習」で作られる（同じ手順を手で行った値と一致）、(f)作り直した新しいモデルが、測った学習者のk件目の状態と同じ予測をする。
- 1標本の処理: 手で作ったA→B→Aの系列で、ウォームアップ中は監視も警報も無い、容量を超えた最古の標本だけが学習される、各標本が1回だけ学習される（学習の回数を数える）、警報の後に保留が空になり監視が始め直される、再来で保有モデルが再利用される、n<5で警報とせず次の標本で合図が続けば選択をやり直す、Fixed-Shareの重みが再利用の警報では保たれ新しいモデルの警報では一様に戻る、真のコンセプトのIDを替えても予測・学習が変わらない。
- run・指標・成果物: 指標が手で計算した値と一致（Kappa系の定義の端の場合を含む）、概念の同定の分類が手で作った記録で期待どおり、保存→読込みの往復、上書きの拒否、不正な系列・hashの不一致の拒否。
- 境界: 新しいパッケージのmoduleが許可した製品のmodule以外の製品のmoduleとtorchをimportしない（ソースの静的な検査）、製品のmoduleが新しいパッケージをimportしない（Windows）、WSLのRiver環境でパッケージをimportしてもtorchが読み込まれない（WSL）。
- 実データでの確認: 書き出した系列（SINE2・SEA4・MNIST4、5,000標本、seed 0〜2）で実行コマンドが動き、成果物と一覧のCSVができる（WSL）。
