# Design Document

## Overview
単一ストリームのパッケージ`single_stream_hard_assignment_soft_prediction`に、RiverのSTAGGER・Agrawal・Random RBFから再来の系列を作って書き出す2つのmoduleと1つのコマンドを足し、系列の読込みがJSONのクラス数を使えるようにする。比較手法の研究ツール`river_baselines.py`には、事前学習の標本の有無の検査を足す。製品は変えない。

### Goals
- 文献の定番の合成データ3つの再来の系列を、既存の系列と同じ形式・同じ概念列の規則で作る。
- 木の版・比較手法・比較の集計が、新しい系列を読める。

### Non-Goals
- 製品のデータセットの定義・概念列の生成の変更。実データ。系列での実験の実行と集計。

## Boundary Commitments

### This Spec Owns
- `random_concept_schedule.py`（概念列。riverとtorchを使わない）
- `river_synthetic_recurring_streams.py`（データの定義と、系列の配列の生成。riverを使う）
- `river_synthetic_stream_command.py`（書き出しのコマンド）
- `single_stream_input.load_single_stream_input`のクラス数の扱い
- `tools/research_single_client_stream_baselines/river_baselines.py`の事前学習の標本の有無の検査

### Out of Boundary
- 製品（`src/federated_learning_experiments/`）。木の版の処理・指標・成果物（変えない）。比較の集計（変えない）。

### Allowed Dependencies
- 新しいmoduleは、標準ライブラリ、numpy、river（`river_synthetic_recurring_streams`とコマンドだけ）、同じパッケージの`single_stream_input.compute_stream_array_digest`、製品の`federated_learning_experiments.data.dataset_definitions`（名前の重なりの検査）だけをimportする。新しい製品のmoduleの許可は足さない。
- testは、概念列の照合のために製品の`generate_random_client_concept_traces`・`ExperimentRunConditions`・`RandomConceptScheduleSettings`と、製品の`fedsda`の既定の概念列の設定（研究ツール`stream_export`と同じ計画の作り方）を使う。

### Revalidation Triggers
- 製品の概念列の規則・値が変わったとき（概念列の照合のtestが落ちる）。
- Riverの版が変わったとき（同じseedで同じ系列にならない可能性。JSONにRiverの版を残す）。

## Architecture

### 概念列（`generate_random_concept_schedule`）
製品の`generate_random_client_concept_traces`の1クライアント分を写す（再来は、ランダムな変化の結果として生じる）。概念0、直前の変化の番号0から始める。引数は標本数、概念の数、Pythonの`random.Random`、最小の間隔（既定300）、変化の確率（既定0.0015）。各標本で、`標本の番号−直前の変化の番号 > 最小の間隔`のときだけ`random()`を引き、確率未満なら、今の概念以外の概念の一覧から`choice`で選ぶ（候補が1つでも`choice`を呼ぶ）。乱数の消費の順は製品と同じ。製品の関数はデータセットの定義から概念の数を取るのでSTAGGER（3概念）には使えず、写しを持つ。既定値は製品の`fedsda`の既定の概念列の設定（公開の`build_final_configuration_fedsda_run_settings`が返す`execution_settings.concept_schedule_settings`）と同じであることをtestで確かめる。

### データの定義（`RIVER_SYNTHETIC_STREAM_DEFINITIONS`）

| データ | 概念 | 生成器（概念iのとき） | クラス | 特徴 |
|---|---|---|---|---|
| `stagger3` | 3 | `STAGGER(classification_function=i, seed=s_i)`（他は既定: クラスの均衡なし） | 2 | size・color・shape（0〜2の整数をfloatに） |
| `agrawal4` | 4 | `Agrawal(classification_function=i, seed=s_i, perturbation=0.05)`（ラベルを決めた後に、数値の特徴を値域の最大±5%乱す。MOAの既定と同じ割合） | 2 | salary・commission・age・elevel・car・zipcode・hvalue・hyears・loan（9） |
| `randomrbf4` | 4 | `RandomRBF(seed_model=i, seed_sample=s_i, n_classes=2, n_features=10, n_centroids=50)`（Riverの既定） | 2 | 0〜9（10） |

- 概念ごとの生成器のseed: `s_i = 1000 × 系列のseed + i`。Random RBFの概念の中身（中心の配置）は`seed_model=i`でseedによらず固定（SEAの概念がseedによらないのと同じ）、標本だけがseedで変わる。
- 特徴の順は、生成器が返す辞書の鍵の順（各データの定義に名前の一覧として持ち、生成した標本の鍵を`str()`した列と一致しなければ止める。Random RBFの鍵は整数0〜9なので、名前は`"0"`〜`"9"`）。JSONの`feature_names`は文字列の列。
- 製品のデータセットの定義と同じ名前は持たない（testで確かめる）。

### 系列の生成（`generate_river_synthetic_stream`）
1. `random.Random(系列のseed)`で概念列を作る（標本数、データの概念の数）。
2. 概念ごとの生成器を最初に使うときに作り、標本ごとにその概念の生成器から次の1件を取る（再来は続きから）。
3. `features`（float64、標本数×特徴数）、`labels`（int64）、`concept_ids`（int64）を返す。

### 書き出し（`river_synthetic_stream_command.main`）
- 引数: `--datasets`（必須、定義にある名前だけ）、`--seeds`（整数の列）、`--sample-count`（既定5000）、`--output`、`--workers`（既定1）。
- ファイル名は既存と同じ`<データ>_seed<seed>_n<標本数>.npz`と同じ名前の`.json`。書く前に、対象の全系列について、NPZとJSONの片方だけがある系列が無いことをまとめて確かめ、あれば何も書かずに止める（片方だけの系列の名前を示す）。両方ある系列は飛ばす。各系列は、NPZを`.tmp.npz`に書いて置き換え、続けてJSONを`.tmp.json`に書いて置き換える（途中で止まるとNPZだけが残り、次の実行は止まる。そのときは残ったNPZを消してやり直す）。木の版・比較手法は`.tmp.npz`を読まない。
- JSON: `export_format_version`（1）、`stream_source`（`river_synthetic_generator`）、`dataset_name`、`random_seed`、`sample_count`、`class_count`、`concept_count`、`feature_count`、`feature_names`、`array_sha256`（`compute_stream_array_digest`）、`concept_change_count`、`generator_settings`（生成器の名前・固定の引数・seedの決め方）、`concept_schedule_settings`（最小の間隔・確率）、`river_version`。

### 読込み（`load_single_stream_input`）
- JSONに`class_count`があれば、bool以外の2以上のintであることを確かめて使う。同じ名前の製品の定義があれば、そのクラス数と一致しなければ止める。
- 無ければ、これまでどおり製品の定義から取り、定義が無ければ止める（既存の系列の読込みは変わらない）。

### 比較手法の研究ツール（`river_baselines.py`）
- `--without-pretraining`が無いとき、系列の一覧を作った直後、出力先の作成と出所の記録の書き込みより前に、全系列のNPZの配列の名前を確かめ、`pretraining_features`か`pretraining_labels`が無い系列があれば、その名前を示して止める（出力先に何も作らない）。

## Components and Interfaces
- `generate_random_concept_schedule(*, sample_count: int, concept_count: int, python_random_generator: random.Random, minimum_sample_index_gap_before_change_trial: int = 300, per_eligible_sample_concept_change_probability: float = 0.0015) -> tuple[int, ...]`。sample_countは0以上、concept_countは2以上のint。
- `RiverSyntheticStreamDefinition`（frozen dataclass）: `dataset_name`、`concept_count`、`class_count`、`feature_names`、`generator_name`、`fixed_generator_arguments`（記録用の辞書）。
- `RIVER_SYNTHETIC_STREAM_DEFINITIONS: dict[str, RiverSyntheticStreamDefinition]`
- `generate_river_synthetic_stream(*, dataset_name: str, random_seed: int, sample_count: int) -> dict[str, np.ndarray]`
- `river_synthetic_stream_command.main(command_arguments: list[str] | None = None) -> int`

## File Structure Plan
- 新規 `src/single_stream_hard_assignment_soft_prediction/random_concept_schedule.py`
- 新規 `src/single_stream_hard_assignment_soft_prediction/river_synthetic_recurring_streams.py`
- 新規 `src/single_stream_hard_assignment_soft_prediction/river_synthetic_stream_command.py`
- 変更 `src/single_stream_hard_assignment_soft_prediction/single_stream_input.py`
- 変更 `tools/research_single_client_stream_baselines/river_baselines.py`
- 新規 `tests/single_stream_hard_assignment_soft_prediction/test_random_concept_schedule.py`（Windows）
- 新規 `tests/single_stream_hard_assignment_soft_prediction/test_river_synthetic_recurring_streams.py`（WSL）
- 新規 `tests/single_stream_hard_assignment_soft_prediction/test_river_synthetic_stream_command.py`（書き出しと、比較手法の研究ツールの検査。WSL）
- 変更 `tests/single_stream_hard_assignment_soft_prediction/test_single_stream_metrics_and_input.py`（読込み）、`test_single_stream_boundaries.py`（riverを使わないmoduleの一覧へ概念列のmoduleを足す）

## Error Handling
- 引数の検査（知らないデータ、負のseed・標本数、概念の数<2、bool）は、乱数を引く前・ファイルを書く前に行う。
- 生成した標本の鍵が定義の特徴の名前と違えば止める（Riverの版の違いの検出）。

## Testing Strategy
- 概念列（Windows）: 製品の`generate_random_client_concept_traces`（1クライアント、`sine2`の2概念と`sea4`の4概念、seedを変えて数通り、標本数5000）と、同じseedの`random.Random`から作った列が一致し、乱数の消費後の状態も一致する。既定値が製品の`fedsda`の既定の概念列の設定と同じ。引数の拒否（乱数を引く前）。概念列の生成が、渡した`Random`以外の乱数（`random`モジュールの全体の状態、numpyの全体の状態）を変えない（要求1.3）。
- 系列（WSL）: 3データで、形・型・クラスの範囲・特徴の名前。定義の`fixed_generator_arguments`に生成器の固定の引数とseedの決め方（`1000 × seed + 概念の番号`）がある（要求2.3）。概念列が`generate_random_concept_schedule`（`Random(seed)`）と一致。概念ごとの標本の並びが、同じ概念の生成器（同じseed）を単独で回した最初のn件と一致（再来で続きから取ること）。同じ引数で同じhash、seedが違えば違うhash。Agrawalの摂動の引数が生成器に渡っていること（各概念の最初の標本が、同じseedで摂動0の生成器の最初の標本と、同じラベルで数値の特徴が違う）。製品の名前と重ならない。
- 書き出し（WSL）: 書いたNPZ・JSONを`load_single_stream_input`で読め、hash・クラス数が合う。JSONの鍵の集合がdesign.mdの一覧と一致し、`generator_settings`がデータの定義と同じ（要求3.1）。両方あれば飛ばし、片方だけの系列が1つでもあれば何も書かずに止める。知らないデータの拒否。木の版の`run_single_stream`が1本を処理できる。
- 読込み（Windows）: JSONのクラス数の採用、製品の定義との不一致・不正な値（1、bool、文字列）の拒否、クラス数の無い既存の形式の系列は従来どおり。
- 比較手法の研究ツール（WSL）: 事前学習の標本の無い系列を`--without-pretraining`なしで渡すと、結果を書く前に止まり、付けると処理できる。
- 境界: 既存の境界のtest（許可した製品のmodule、torchを読まない）に新しいmoduleが含まれて通る。概念列のmoduleは、別processでimportしたときに`river`も`torch`も`sys.modules`に入らないことを確かめる（要求6.3。既存のtestは`river`を許可した第三者のpackageとして扱うので、概念列のmoduleだけ`river`も確かめるtestを足す）。
