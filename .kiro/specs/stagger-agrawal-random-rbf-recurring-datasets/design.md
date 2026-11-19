# Design Document

## Overview
3データ（`stagger3`・`agrawal4`・`randomrbf4`）を、製品の既存の合成データ（SEA・SINE・CIRCLE）と同じ経路に足す。生成器は`data/`の下に1データ1moduleで置き、runが貸すNumPyの`RandomState`で標本を作る。
概念の中身は単一ストリームの版が使うRiver 0.27.0の定義に合わせ、Riverには依存しない。Riverとの一致は、Riverのある環境（WSLの`~/river-env`）だけで動くtestで確かめる。
既存の10データの経路・値・乱数の消費は変えない。

## Boundary Commitments
- このspecが持つもの: 3つの生成器module、データの定義と実行条件の許される値、生成器の表、3手法のデータ別の設定の表、掃引の重さの表、単一ストリームの版の名前の検査、文書。
- 持たないもの: 概念の時系列の生成（`data/concept_schedules/`）、事前学習・評価・各手法の処理（変えない）。3データでの本規模の実験とδの調整（後続の研究）。
- 許す依存: 新しい生成器moduleは`numpy`・`random`（Random RBFの中心）・`math`と`data/observed_streams.py`の`ObservedSample`だけをimportする。`river`はtestだけがimportする（`pytest.importorskip`）。
- 再検証が要る変更: 生成器の乱数の消費順を変えると、3データの全結果が変わる（既存10データには影響しない）。

## 根拠（River 0.27.0、`river/datasets/synth/`）
- `stagger.py`の`STAGGER.__iter__`: `size = rng.randint(0, 2)`、`color = rng.randint(0, 2)`、`shape = rng.randint(0, 2)`の順。分類の関数は`_classification_function_zero/one/two`（要求1.3）。`balance_classes=False`では1回で決まる。
- `agrawal.py`の`Agrawal.__iter__`（`balance_classes=False`、`perturbation=0.05`）:
  - 生成の順序: `salary = 20000 + 130000 * random()`、`commission = 0 if salary >= 75000 else 10000 + 75000 * random()`、`age = randint(20, 80)`、`elevel = randint(0, 4)`、`car = randint(1, 20)`、`zipcode = randint(0, 8)`、`hvalue = (8 - zipcode) * 100000 * (0.5 + random())`、`hyears = randint(1, 30)`、`loan = random() * 500000`。
  - ラベル: `_classification_function_0`〜`_3`を、摂動の前の値で計算する。
  - 摂動の順序: salary（範囲20000〜150000）→ commission（0より大きいときだけ、10000〜75000）→ age（20〜80、`round`）→ hvalue（`_perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)`）→ hyears（1〜30、`round`）→ loan（0〜500000）。
  - `_perturb_value(val, val_min, val_max, val_range=None)`: `val_range`が無ければ`val_max - val_min`。`val += val_range * (2 * (random() - 0.5)) * perturbation`の後、`val_min`未満なら`val_min`、`val_max`超なら`val_max`。
  - hvalueの呼び出しは、引数の位置から`val_min=(9 - zipcode) * 100000`、`val_max=0`、`val_range=135000`になる。結果のhvalueは`(9 - zipcode) * 100000`か0のどちらかになる（MOAの`perturbValue(val, range, min, max)`の引数の並びを、Riverが`(val, min, max, range)`として受けたためと見られる）。単一ストリームの版と同じ定義にするため、この挙動のまま再現し、文書に開示する。分類の関数0〜3はhvalueを使わないので、ラベルには影響しない。
- `random_rbf.py`の`RandomRBF`:
  - `_generate_centroids`: `rng_model = random.Random(seed_model)`。中心iごとに、位置10個（`random()`）、ラベル（`randint(0, n_classes - 1)`）、標準偏差（`random()`）、重み（`random()`）の順。
  - `_generate_sample`: `random_index_based_on_weights`（`val = random() * sum(weights)`、累積が`val`を超えた最初の添字）で中心を選び、特徴iごとに`random() * 2 - 1`、大きさの正規化、`gauss(0, 1) * std_dev`の大きさで、`centre[i] + att[i] * scale`。

## 生成器（新しいmodule）
| module | 型 | 概念数の検査 |
|---|---|---|
| `data/stagger/stagger_sample_generation.py` | `StaggerSampleGenerator` | 1〜3 |
| `data/agrawal/agrawal_sample_generation.py` | `AgrawalSampleGenerator` | 1〜4（River の関数0〜3） |
| `data/random_rbf/random_rbf_sample_generation.py` | `RandomRbfSampleGenerator` | 1〜4 |

- 構築: `SeaSampleGenerator`と同じく`numpy_random_generator`と`concept_count`を受け、型と範囲を検査する。構築では乱数を使わない（Random RBFの中心は、構築のときに概念0〜concept_count−1の分を`random.Random(c)`で作る。runの乱数は使わない）。
- `generate_sample(concept_id=...)`: 概念を検査してから乱数を使う。
- Pythonの`random.Random`とNumPyの対応: `random()`→`random_sample()`、`randint(a, b)`（両端を含む）→`randint(a, b + 1)`、`gauss(0, 1)`→`standard_normal()`。1標本の中の乱数の使用の順序は、上の根拠のRiverの順序と同じにする。値の列はRiverと同じにならない（要求の範囲）。
- 丸め: Agrawalのage・hyearsの`round`は、Pythonの`round`（偶数への丸め）をそのまま使う。
- Random RBFの中心の選択: Riverの`random_index_based_on_weights`と同じく、Pythonのfloatで重みを先頭から逐次に足し、`<=`の比較で添字を決める（`numpy.cumsum`・`searchsorted`は使わない。境界の丸めで添字が変わりうるため）。重みの合計はPythonの`sum`で構築のときに計算する。
- 分類の関数とRandom RBFの中心の生成は、Riverとの照合のtestから呼べるよう、公開の関数にする（`compute_stagger_class_label`、`compute_agrawal_class_label`、`generate_random_rbf_centroids`）。`generate_random_rbf_centroids`は、不変の記録`RandomRbfCentroid`（位置のtuple、ラベル、標準偏差、重み）のtupleを返す。
- 3つのpackage（`data/stagger/`・`data/agrawal/`・`data/random_rbf/`）は、説明だけの`__init__.py`を持つ（再exportしない。既存の`data/sea/`と同じ）。
- 特徴の尺度（要求1.4・2.5）:
  - STAGGER: 値/2。
  - Agrawal: salary (x−20000)/130000、commission x/75000、age (x−20)/60、elevel x/4、car (x−1)/19、zipcode x/8、hvalue x/900000、hyears (x−1)/29、loan x/500000。範囲は、摂動0.05のときの出力が取りうる値の範囲（hvalueは上の挙動で0〜900000。摂動なしなら最大1.2×10⁶になるが、製品は摂動0.05だけを使う）。
  - Random RBF: 写さない。値に上限・下限が無い（中心からの距離は標準正規×標準偏差）ので、固定の範囲を決められないため。実測の範囲（seed 0の各概念の2万標本の最小・最大）を文書に書く。
  - 写す理由: Agrawalは特徴の桁が1〜5×10⁵まで違い（salary・loan・hvalue）、そのままでは神経網の学習が成り立ちにくい。STAGGERは0・1・2の順序の値で、写さなくても学習できるが、Agrawalと同じく0〜1にそろえる。SEA（0〜10）は既存のとおり写さない（既存の値を変えない）。全データの方針ではなく、3データの個別の判断である。
- 特徴は、尺度を写した後にfloat32へ丸め、Pythonのfloatのtupleで`ObservedSample`へ入れる（SEAと同じ）。

## 製品への接続
- `dataset_definitions.py`: `stagger3`（入力3、概念3、クラス2）、`agrawal4`（9、4、2）、`randomrbf4`（10、4、2）を、既存の10データの後に足す。
- `experiment_run_conditions.py`: 直書きの許される値のtupleに、3データを既存の10データの後へ同じ順で足す（`DEFINED_DATASET_NAMES`と一致することを既存のtestが確かめる）。
- `observed_sample_generation.py`: 生成器の表と構築の分岐、`is_observed_sample_generator_of_dataset`の概念数の照合に3つの型を足す。
- `fedsda_final_configuration_run_settings.py`・`feddrift_default_run_settings.py`: 3データを合成データの値（隠れ層(32, 32)、学習率0.01）にする。FedDriftのδは0.10（公式の`default_deltas`は`sea`0.04・`sine`0.20・`circle`0.10・`MNIST`0.10。3データの値は無いので、4つのうち2つが使う0.10にする）。本規模の前に、評価に使わないseedでδを調整する（[前回の調整](../../../docs/research/feddrift-delta-tuning-new-datasets-v16-20261101.md)と同じ手順。このspecの外）。
- `fedavg_run_settings.py`: 3データは画像でない側（合成データの設定）になることをtestで確かめる。分岐の書き方は変えない（明示の画像データの集合に含まれない）。DWM（`dynamic_weighted_majority_run_settings.py`）はFedAvgの設定を使うので、同じ値になる（testで確かめる）。
- `experiment_sweep_execution.py`の重さ: 3データは表に無い（0）、今の合成データと同じ。`experiment_sweep_command.py`の画像の読込みの前検査には入らない。
- 単一ストリームの版（`river_synthetic_recurring_streams.py`の`_validate_arguments`）: 製品と同じ名前の拒否を、製品の定義の`concept_count`・`class_count`・`input_feature_count`と単一ストリームの定義（`concept_count`・`class_count`・`len(feature_names)`）の一致の検査に替える。単一ストリームの系列の生成（Riverの生成器）は変えない。
- 単一ストリームの読込み（`single_stream_input.py`の`_resolve_class_count`）は変えない。系列のJSONの`class_count`（2）が製品の定義（2）と一致するので、3データの既存の系列はそのまま読める。
- 既存の単一ストリームのtestで直すもの: `tests/single_stream_hard_assignment_soft_prediction/test_river_synthetic_recurring_streams.py`の「製品の定義が無い」ことを確かめるtestを、定義の一致の検査（一致すれば生成でき、概念数などが違う定義を拒否する）のtestへ替える。`test_single_stream_metrics_and_input.py`の、`stagger3`を製品の外の名前として`class_count=3`を受け入れるtestは、製品に無い名前へ替える。
- 既存の製品のtestで直すもの: データ名の一覧を完全一致で比べる`tests/refactoring/test_dataset_definitions.py`・`test_run_settings_validation.py`（`allowed_parameter_values`）・`test_synthetic_sample_generation.py`（生成器の表）。依存境界の`tests/refactoring/test_single_run_dependency_boundaries.py`は、3packageを`package_boundary_paths`とpackageごとの一覧へ足し、3moduleの許可集合（`numpy`・`math`・`random`と`ObservedSample`の、実際にimportする名前の組）を既存の`sea_sample_generation`と同じ形で登録する。

## 開示する差（文書に書く）
- 標本の乱数の列: Riverの`random.Random`ではなく、runが貸すNumPyの`RandomState`。分布は同じで、値の列は単一ストリームの版と違う。
- 概念の時系列とclientへの分け方: 今の製品のデータと同じ（ランダムな変化点、全手法で共通）。単一ストリームの版の1本の系列とは違う。
- Agrawalのhvalueの摂動: Riverの引数の並びのまま（上の根拠）。
- 特徴の尺度: 神経網のため、STAGGER・Agrawalを固定の範囲で0〜1へ写す。単一ストリームの木の版は写していない（木は尺度に依らない）。
- FedDriftのδの既定0.10は公式の値ではなく、仮定である。公式のFedDriftの表（手元の`references/code/FedDrift/fedml_api/distributed/fedavg_ens/FedAvgEnsDataLoader.py`の1274行の`default_deltas`と、455行の`MultiModelAcc_data_loader`の`deltas`。同じ値: sea 0.04・sine 0.20・circle 0.10・MNIST 0.10）に3データは無い。274行の表（sea 0.02・sine 0.10・circle 0.05）はDriftSurfのもので使わない（2026-11-20、主担当が確認）。0.10は4値のうちcircle・MNISTの値で、選ぶ根拠は弱い。本規模の前の調整で選び直す。

## Testing Strategy
- 定義（Windows venv）: 要求1〜3の値の範囲・ラベルの規則・尺度、要求4.1の再現性（同じseedで同じ標本）、4.2の拒否、構築で乱数を使わないこと。
- Riverとの照合（WSLの`~/river-env`、`pytest.importorskip("river")`）: 要求6.2の(i)〜(iii)。
  - (i) 摂動なし（`perturbation=0.0`）のRiverの`Agrawal`と、`STAGGER`の生成器の出す特徴とラベルを、製品の分類の関数へ渡し、各概念の1万標本で全てのラベルが一致すること（Agrawalは摂動の後の特徴ではラベルが決まらないため、摂動なしで比べる）。
  - (ii) Random RBFの全中心（位置・ラベル・標準偏差・重み）の一致。
  - (iii) 各概念で、製品の生成器（固定seed）とRiverの生成器（摂動0.05、固定seed）の2万標本ずつで、クラス1の割合と各特徴の平均の差が、差の標準誤差（両方の標本の分散から計算）の4倍以内。Riverの特徴は製品と同じ尺度へ写してから比べる。seedは固定なので結果は決まる。4倍は、正しい実装で偶然に外れる確率が十分小さく（正規近似で約6×10⁻⁵／比較）、平均のずれ（例: 範囲の数%）を見逃さない幅として選ぶ。
  - (iii)の比較は約100あり、全体で偶然に1つ外れる確率は約0.6%。外れたときは、別のseedで同じ比較をして、同じ比較が再び外れるかを確かめ、結果をtasks.mdのImplementation Notesへ書く（許容を広げて通さない）。
  - (i)のSTAGGERは、27通り×3概念の全てで比べる。
  - (i)〜(iii)は全て3.1のtestが持つ（Riverの環境で動く1つのmodule）。
  - このtestのmoduleはtorchをimportしない。
- fresh process: 共用script`tests/refactoring/fresh_process_smoke.py`へ、3データのFedSDAの全体run（小さい条件）の流れを足す。
- 設定と全体run: 要求7.2の(iii)〜(v)。(iv)は既存の`test_fmnist_and_rotated_mnist_runs.py`と同じ形（各データ1 run、10 client、各client 100標本、集約間隔50）で、FedSDA・FedDrift・FedAvg・DWMを流す。データ名を固定の一覧で持つ既存のtest（`tests/refactoring/test_dataset_definitions.py`の`DATASET_NAMES`、`test_fedavg_experiment_run.py`・`test_feddrift_experiment_sweep.py`のデータの一覧、`fresh_process_smoke.py`のデータの一覧）は、目的が全データを覆うことなら3データを足し、既存のデータの値を固定する目的なら変えない（実装のときに各箇所の目的を読んで決め、Implementation Notesへ書く）。
- 単一ストリームの版: 名前の一致の検査で、製品の定義と一致すれば生成でき、一致しない定義を拒否すること。既存の単一ストリームのtest（系列の値）が変わらず通ること（WSL）。
- 回帰: 全版のgoldenが通る。依存境界: 新しい3moduleを`test_single_run_dependency_boundaries.py`の許可集合へ登録する。

## 範囲の外
- `balance_classes`、Agrawalの関数4〜9、Random RBFの中心の移動（`RandomRBFDrift`）。
- 本規模の実験、δの調整、論文の表の更新。
