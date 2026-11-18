# Design Document

## Overview
**Purpose**: FedHASPの1クライアントの純粋な形を、Hoeffding treeを学習器にして、単一ストリームの既存手法と同じ条件で動かすパッケージ`single_stream_hard_assignment_soft_prediction`を作る。
**Users**: FedHASPの仕組みを、単一ストリームのオンライン分類の既存手法と比べる研究者。
**Impact**: 製品（`federated_learning_experiments`）は変えない。新しいパッケージは`src/`の下の別の最上位パッケージで、製品の判断の部品を一方向にimportする。

### Goals
- 論文の4.3節（固い割り当て）・4.2節（柔らかい予測のうち局所の版の混合）の手順を、Hoeffding treeで実装する。
- 既存手法と同じ条件（事前学習なし、最初の標本から評価、標本ごとに1回学習、既定値）で、書き出した系列を処理する。
- 判断の部品は製品のコードをそのまま使い、同じアルゴリズムであることを部品の共有で示す。

### Non-Goals
- 神経網の学習器、連合学習、集約、統合、同期の版の混合（1クライアントの純粋な形では働かない）。
- 製品の全体の流れとの出力の一致の照合（判断の部品の共有と、流れのtestで代える）。
- 新しいデータの生成、比較手法の追加、結果の集計。

## Boundary Commitments

### This Spec Owns
- `src/single_stream_hard_assignment_soft_prediction/`の全module（概念モデル、設定、1標本の処理、run、成果物、実行コマンド）。
- このパッケージのtest（`tests/single_stream_hard_assignment_soft_prediction/`）と、Riverを入れた実行環境の要件のファイル。

### Out of Boundary
- 製品の全module。再利用する判断の部品の挙動（製品のtestが保証）。
- RiverのHoeffding treeの挙動。

### Allowed Dependencies
- 製品から: `methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring`（`OverallAndTrueClassLossMonitor`）、`methods.fedsda.loss_change_detection.loss_change_detection_settings`（`LossChangeDetectionSettings`）、`methods.fedsda.prediction_combination.fixed_share_weights`（`FixedShareWeightController`）、`methods.fedsda.prediction_combination.prediction_combination_settings`（`PredictionCombinationSettings`）、`methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection`（`select_assignment_by_minimum_description_length`、`select_new_model_learner`、`compute_switching_code_length`、学習者の名前の定数、`MinimumDescriptionLengthAssignmentSelection`）、`methods.fedsda.loss_statistics.loss_baseline_selection`（`select_loss_monitoring_baseline_mean_loss`）、`learning.loss_statistics.model_and_class_loss_statistics`（`ModelAndClassLossStatisticsStore`）。どれもtorchを読み込まないことを、WSLのRiver環境で確かめた（research.md）。
- 外部: `river`（0.27.0）、`numpy`、標準ライブラリ。
- 製品は新しいパッケージをimportしない。

### Revalidation Triggers
- 再利用する製品の判断の部品の公開の関数・引数・挙動が変わったとき。
- 製品のFedHASPの既定（監視・Fixed-Share・MDLの設定）が変わったとき（設定の一致のtestが落ちる）。

## Architecture

### 論文の手順との対応（1クライアント・同期なし）

| 論文の手順（4章） | 製品 | このパッケージ |
|---|---|---|
| モデル: 共有の特徴抽出器＋adapter＋出力層 | 神経網 | Hoeffding tree（共有部なし）。確率を一様分布と0.01で混ぜて平滑化 |
| 予測: 全モデルをFixed-Shareの重み$w_m$で混ぜる。同期の版と局所の版の混合$v$ | 両方 | 局所の版だけ（同期なしでは製品でも同期の版は使われない。間隔5000で製品のFedHASPと「同期の版を混ぜない」構成が全300 runで一致） |
| 重みの更新: 誤り$e_m=1-p_m(x)_y$でFixed-Share（T=120、ηは累積の分散から） | `FixedShareWeightController` | 同じ部品 |
| 監視: 現在のモデルの誤りをe-detectorで、全体とクラスごとに監視（α=0.001、低下で基準を下げる） | `OverallAndTrueClassLossMonitor`（事前学習の5分割交差検証の統計で開始） | 同じ部品。200件のウォームアップの後、現在のモデルの損失の統計の平均で開始 |
| 帰属の保留: FIFO 30、容量を超えた最古の標本を現在のモデルへ | `PendingTrainingAssignmentBuffer`など | 同じ規則を、このパッケージの小さいFIFOで持つ（製品の保留の部品はtorchの標本の型に結びつくため） |
| 警報: 合図があり、min(n,k)≥5 | 同じ | 同じ |
| MDL: (i)そのまま、(ii)保有モデルへ、(iii)新しいモデル（keep・resetの写しを新しい順に1件ずつ予測→学習、短い方＋ln 2）、$L_{sw}=\ln T+\ln M$ | 符号長の計算はtorch、比較は`select_assignment_by_minimum_description_length` | 符号長は木で計算し、比較は同じ部品 |
| 選んだ後: 前のn−k件を現在のモデル、後のk件を帰属先へ。FIFOを空に、帰属先の損失の平均で監視を始め直す | 同じ | 同じ |
| 新しいモデル: 区間の損失が最小の保有モデルを初期値に、区間で数エポック学習 | 候補の学習 | **違う**: MDLで短かった学習者の種類で作り（resetは空の木、keepは現在のモデルの写し）、後のk件を古い順に1回ずつ学習（下の「製品と違う点」） |
| 学習: 毎標本、各モデルのデータストアからミニバッチを作り、全モデルを1回更新 | 再生のミニバッチ | **違う**: 割り当てた時点に、その標本でそのモデルを1回学習（既存手法の条件） |
| 集約・統合・再較正 | あり | 無し |

### 製品と違う点（論文・設計に明記する）
1. 学習器はHoeffding treeで、共有の特徴抽出部とadapterが無い。
2. 学習は、割り当てた時点の1回だけ（データストアからの再生をしない）。既存手法の「標本ごとに1回」に合わせる。
3. 新しいモデルの初期値は、MDLで短かった学習者の種類に合わせる。木は数エポックの学習をしないので、MDLで測った学習者（1件ずつ学ぶ）と、作るモデルの学び方が同じになる。区間の損失が最小の保有モデルの写しから始めると、木は過去のコンセプトで作った分岐を捨てられないため。
4. 事前学習の代わりに、最初の200件をウォームアップにする（警報を出さない。予測は評価に含める）。200はRiverのHoeffding treeの既定のgrace periodと同じ。
5. 確率を一様分布と0.01で混ぜて平滑化する（木は確率0を出すことがあり、MDLの対数損失を有限にするため）。

### 1標本の処理の順（`SingleStreamHardAssignmentSoftPredictionProcessor.process_sample`）
1. 予測: 保有モデル（作った順）の平滑化した確率を求め、`FixedShareWeightController.get_weights_before_label_observation(model_ids=...)`の重みで混ぜ、最大のクラスを予測とする。記録する。
2. 重みの更新: 各モデルの監視用の損失で`update_weights_after_loss_observation`を呼ぶ。
3. 監視: ウォームアップの後なら、現在のモデルの監視用の損失と正解クラスで`observe_loss_after_label_observation`を呼ぶ（基準値は`select_loss_monitoring_baseline_mean_loss`で現在のモデルの統計から）。ウォームアップ中は監視しない。200件目の処理の最後に監視の部品を作る。
4. 保留: 標本をFIFOへ加える。
5. 警報の判定: 検出器の合図があれば、FIFOの列Wで`select_alarm_interval_assignment`（このパッケージの関数）を呼ぶ。min(n,k)<5なら警報としない。
6. 警報なら: 前のn−k件を現在のモデルへ割り当て（統計に記録→学習）、新しいモデルなら作って保有へ加え、後のk件を帰属先へ割り当て、現在のモデルを替え、FIFOを空にし、`reset(baseline_loss_mean=帰属先の平均)`で監視を始め直す。
7. 警報でなければ: FIFOが容量を超えたら最古の標本を現在のモデルへ割り当てる（統計に記録→学習）。

製品の処理の順（予測→重みの更新→監視→保留→警報の判定→割り当て・学習）と同じ順にする。

### MDLの符号長（`select_alarm_interval_assignment`）
- Wの各標本について、全保有モデルのMDL用の損失を求める。
- 帰属先の候補は、現在のモデル以外で、損失の統計を持つ保有モデル（保有の順）。
- 新しいモデル: n≥5なら、Wを逆順にし、resetの学習者（空の概念モデル）とkeepの学習者（現在のモデルの写し）に、1件ずつMDL用の損失を足してから学習させ、前置きの和の列を作る。k=5..nについて`select_new_model_learner`で短い方＋ln 2と学習者の名前を得る。
- `compute_switching_code_length(switching_probability=1/T, destination_count=帰属先の候補の数+1)`。
- `select_assignment_by_minimum_description_length(..., change_interval_grid_sample_count=5, change_interval_step_sample_count=1)`。

### 指標
- prequential精度: 正解の割合。
- Kappa: (p0−pc)/(1−pc)、pcは予測と観測のクラスの周辺分布から。
- Kappa-M: (p0−pm)/(1−pm)、pmは「それまでに最も多く観測したクラス」を予測する分類器のprequential精度。
- Kappa-Temporal: (p0−pt)/(1−pt)、ptは「直前の標本のラベル」を予測する分類器のprequential精度（最初の標本はクラス0）。
- 分母が0のときは欠損（NaN）とする。

## Components and Interfaces

| module | 公開するもの | 役割 |
|---|---|---|
| `hoeffding_tree_concept_model.py` | `HoeffdingTreeConceptModel` | 木を1つ持ち、平滑化した確率、監視用・MDL用の損失、1件の学習、写し、空の新しいモデルの作成、節点の数 |
| `single_stream_settings.py` | `SingleStreamHardAssignmentSoftPredictionSettings`、`build_default_single_stream_settings` | 手法の設定（保留の容量、最小の変化区間、ウォームアップ、平滑化、T、監視の設定、Fixed-Shareの設定）。riverをimportしない |
| `alarm_interval_assignment.py` | `select_alarm_interval_assignment` | 列Wで符号長を求め、製品の比較の関数で選ぶ。保有モデルを変えない |
| `single_stream_processing.py` | `SingleStreamHardAssignmentSoftPredictionProcessor`、`SampleProcessingRecord`、`AlarmRecord` | 1標本の処理と記録 |
| `single_stream_run.py` | `run_single_stream`、`SingleStreamRunResult`、`compute_prequential_metrics` | 系列1本の処理、指標、経過時間 |
| `single_stream_artifacts.py` | `save_single_stream_run`、`load_single_stream_run` | JSON・NPZの保存と読込み（上書きしない、pickleなし） |
| `single_stream_command.py` | `main` | 系列のファイルを並列に処理し、成果物と一覧のCSVを書く |

## File Structure Plan
- 新規（src）: `src/single_stream_hard_assignment_soft_prediction/__init__.py`と上の7 module。
- 新規（test）: `tests/single_stream_hard_assignment_soft_prediction/`に、moduleごとのtestと境界のtest。riverが無い環境では、riverを使うtestを`pytest.importorskip("river")`で飛ばす。設定の値の一致のtest（製品の既定と比べる）はriverを使わず、Windowsの固定venvで動く。
- 新規（環境）: `requirements-single-stream.txt`（river==0.27.0、numpy）。WSLの`~/river-env`で実行する。
- 変えない: 製品の全ファイル、`pytest.ini`（新しいtestの場所は既存の収集の範囲に入る）。

## Error Handling
- 系列のNPZの欠け（必要な配列が無い、長さの不一致、クラスが0未満）は、処理の前に`ValueError`。
- 設定の範囲外（容量<1、最小の変化区間<1、ウォームアップ<1、平滑化が0以下か1以上、T≤1）は、設定の作成時に`ValueError`。
- 成果物の保存先に同名のファイルがあれば`FileExistsError`（書かない）。

## Testing Strategy
- 概念モデル: 空の木は一様分布、平滑化で最大のクラスが変わらない、確率の和が1、損失の式、写しが独立に学ぶ、新しいモデルが空。
- 設定: 既定の値が製品のFedHASPの既定と同じ（監視の設定、Fixed-Shareの設定、保留の容量30、最小の変化区間5、T=120、switching_probability=1/T）。範囲外の拒否。
- MDL: 手で作った小さい列で、(a) 現在のモデルが合うと「そのまま」、(b) 保有モデルが合うとそのモデルへ、(c) どれも合わないと新しいモデル、(d) 保有モデルの状態が変わらない、(e) 前置きの和が1件ずつの予測→学習で作られる（逆順）。
- 1標本の処理: 手で作った系列（2つのコンセプトがA→B→Aと切り替わる）で、ウォームアップ中は警報が出ない、FIFOの容量を超えた最古の標本だけが学習される、警報の後にFIFOが空になり監視が始め直される、再来でモデルが再利用される、真のコンセプトのIDを替えても予測・学習が変わらない、各標本が1回だけ学習される。
- run・成果物: 指標の式（手で計算した値と一致）、保存→読込みの往復、上書きの拒否、不正な系列の拒否。
- 境界: 新しいパッケージのmoduleが、許可した製品のmodule以外の製品のmoduleと`torch`をimportしない（ソースの静的な検査）。製品のmoduleが新しいパッケージをimportしない。WSLのRiver環境でパッケージをimportしても`torch`が読み込まれない。
- 実データでの確認: 書き出した系列（SINE2・SEA4・MNIST4、5,000標本、seed 0〜2）で実行コマンドが動き、成果物ができる。
