# Design Document

## Overview
`LocalTrainingSettings`へ、学習のbatchの抜き出し方の3つの項目（既定値つき）を足し、共同更新の繰り返し（`perform_held_model_joint_training_iterations`）が、抜き出しの関数へ渡す。抜き出しの関数は、選び方の引数を省略すれば今と同じ。掃引の計画へ、FedSDAの構成の変種を作る軸を足す。

## Boundary Commitments
### This Spec Owns
- `learning/training/local_training_settings.py`: 3つの項目と検査。
- `learning/training/held_model_training_batch_sampling.py`: 選び方の引数と、直近K件・指数減衰の抜き出し。
- `learning/training/held_model_joint_training_iterations.py`: 設定を抜き出しへ渡す。
- `runtime/fedsda_run_client_settings.py`: Kとbatchの件数の組合せの検査。
- `sweep/experiment_sweep_plan.py`・`experiment_sweep_command.py`・`experiment_sweep_records.py`: 軸、引数、表の列。
- `tests/refactoring/behavior_golden_cases.py`: 新しい項目を、版1〜5の「後に足された項目」に登録する。
- 文書。
### Out of Boundary
候補の学習・事前学習・FedDrift（抜き出しの関数を、選び方の引数なしで呼ぶので、今のまま）、標本の保存、警報を区切りに使う方法、比較script・分析script、既定の値。
### Allowed Dependencies
新しいmoduleは無い。抜き出しのmoduleは、`from math import floor, log`と`from bisect import insort`で書き（依存境界の許可は`math.floor`・`math.log`・`bisect.insort`の名前で足す）、同じpackageの`LocalTrainingSettings`を読む。依存境界のtestは、今、このmoduleの`import math`を、禁止の例（`test_single_run_dependency_checker_rejects_forbidden_imports`の入力。学習標本の記録のmoduleと共通のparametrize）として持っているので、このmoduleの分だけ外し（記録のmoduleの禁止の例は残す。このmoduleには別の禁止の依存——例えば`numpy`——の例を置く）、許可の一覧へ`math.floor`・`math.log`と`LocalTrainingSettings`を足す。
### Revalidation Triggers
FedSDAのrun設定の保存の形（項目が増えるので、FedSDAのrunの識別子のhashが全部変わる）、goldenの設定の比較、抜き出しの関数の呼出し。

## Architecture
### 設定（`LocalTrainingSettings`）
- `training_sample_selection_policy: str = "uniform_over_all_stored_samples"`（今の項目と同じく、`allowed_parameter_values`のmetadataで許す値を宣言し、共有の検査に任せる）。許す値: `uniform_over_all_stored_samples`（既定）、`uniform_over_most_recent_stored_samples`、`recency_decayed_stored_samples`。
- `most_recent_stored_sample_count: int | None = None`（直近K件のときだけ、bool以外の正のbuiltin int。ほかはNone）。
- `stored_sample_recency_decay_factor: float | None = None`（指数減衰のときだけ、0<α<1の有限のbuiltin float。ほかはNone）。
- 共有の検査`core/settings_field_validation.py`の`validate_settings_field_values`は、今は宣言の型が`str`・`int`・`float`以外なら`TypeError`を出す。ここへ、`int | None`と`float | None`の扱いを足す（値がNoneなら通し、Noneでなければ`int`・`float`と同じ検査。範囲のmetadataも同じ）。今は、Noneを許す宣言の設定の項目は無い（宣言すると型の誤りになる）ので、既存の設定の検査は変わらない。`validate_settings_field_values`のtestへ、この2つの型の受理・拒否を足す。
- `LocalTrainingSettings.__post_init__`は、共有の検査の後に、値の型を厳密に確かめ（Noneでなければ、Kは`type(K) is int`、αは`type(α) is float`。共有の検査は、int宣言で派生型を、float宣言でintも受けるため。`fedsda_run_client_settings.py`の件数の項目の検査と同じ書き方）、選び方と2つの値の組合せ（直近K件ならKだけ、指数減衰ならαだけ、全履歴ならどちらもNone）を確かめる。誤りは`RunSettingsValidationError`。範囲（K≥1、0<α<1）は、metadata（`minimum_allowed_value`ほか）で共有の検査に任せる。
- Kとbatchの件数（`FedsdaRunClientScalarSettings.local_training_batch_sample_count`）の組合せ（K≥batchの件数）は、両方を持つ`FedsdaRunClientSettings.__post_init__`で確かめる。

### 抜き出し（`sample_training_batches_for_held_models`）
- keyword引数`local_training_settings: LocalTrainingSettings | None = None`を足す。型は、乱数を使う前（入れ物の検査と同じ時点）に確かめる。Noneか既定の選び方なら、今と同じ処理（`python_random_generator.sample(training_samples, batch_sample_count)`）。
- 直近K件: `python_random_generator.sample(training_samples[-K:], batch_sample_count)`。列の長さがK未満なら、列の全部（今と同じ母集団）。Kがbatchの件数より小さい設定を直接渡されたら、乱数を進める前に`ValueError`で拒否する（run設定では、`FedsdaRunClientSettings`の検査が先に拒否する）。
- 指数減衰: 1件ずつ、batchの件数だけ、正確に選ぶ（引き直しはしない）。位置は「末尾から何件前か」のa（0が最新、L−1が最古。Lは列の長さ）で表す。選んだaは、昇順の列（`bisect.insort`で挿入。1件あたり選んだ件数に比例）に保つ。各回の手順:
  1. 未選択の範囲を、選んだ位置で区切られた区間[s, e)の列にする（新しい順。長さ0の区間は作らない）。最初の区間の始まりをa0（未選択の最新の位置）とする。
  2. 区間の重みの和（a0の重みを1とする相対値）: `S = α^(s − a0)·(1 − α^(e − s)) / (1 − α)`。`α^(s − a0)`が0へ丸められた区間は、和が0で、選ばれない。全区間の和をWとする（最初の区間の和は1以上なので、W≥1）。
  3. `u = python_random_generator.random()`、`t = u·W`。区間を新しい順にたどり、区間ごとに`t`から、その区間の和を引いていく。`t < S`となった区間で止め、その時点の`t`（区間の中の残り`t_in`）で、区間の始まりsからの距離を求める: `x = 1 − t_in·(1 − α) / α^(s − a0)`、`d = floor(log(x) / log(α))`（x>0のとき）。xが0以下（丸め）なら、dは区間の長さ−1。dは`[0, e − s − 1]`へ収める。選ぶ位置は`a = s + d`。
  4. 丸めで、どの区間でも止まらなかったら、和が正の最後の区間の、最後の位置を選ぶ。
  手数は1件あたり選んだ件数（batchの件数未満）に比例し、列の長さに依らない。乱数は1件あたり`random()`を1回だけ使う。各回の条件付き確率は、未選択の位置の中で`α^a`に比例する（要求1.3。区間の和と逆関数は、幾何級数の閉じた式）。αが極端に小さくても、Wは1以上で、0での除算や`log`の定義域外は起きない（`α^(s − a0)`が0の区間は通り過ぎるだけで、その中では逆関数を使わない）。選んだ位置の標本は`列[L − 1 − a]`。batchの順は選んだ順。
- 学習に使えるモデルの条件（列の長さ≥batchの件数）と、検査・失敗時の乱数の状態の戻しは、今のまま。
- FedDriftの呼出し（`feddrift_client_round_training.py`）は、引数を足さない。

### 共同更新の繰り返し
`perform_held_model_joint_training_iterations`は、受け取っている`local_training_settings`を、抜き出しへそのまま渡す。共有部を持たない構成（`sequential_per_model_training`）も同じ関数を通るので、同じ選び方が効く。

### 掃引
- `ExperimentSweepPlan.fedsda_training_sample_selections: tuple[str, ...] | None = None`。値は次の書式の文字列（重複なし）: `uniform`、`recent:<K>`（Kは先頭が0でない10進の数字だけ。符号・空白・小数点は不可）、`decay:<α>`（αは`0.`に続く1桁以上の10進の数字だけ。例`0.997`。`nan`・`inf`・`1`・`1.0`・指数表記・符号・空白は不可で、αが0になる`0.000`も不可）。それ以外は`ValueError`。K≥batchの件数とαの範囲は、作ったrun設定の検査が確かめる。`__post_init__`は、文字列の解析（`parse_fedsda_training_sample_selection`）と、解析した値（選び方・K・α）の重複を確かめる（`recent:300`と`recent:0300`は同じ値なので重複）。FedSDAの構成を含むかは、`build_planned_experiment_runs`の中で、作ったrun設定の型（`FedsdaRunSettings`）で確かめ、1つも無ければ`ValueError`（構成の名前は11個にまたがるので、名前では見分けない。FedDriftの閾値の軸は`__post_init__`で構成の名前を見るが、FedDriftは構成が1つなので名前で足りる。どちらも、コマンドでは終了コード2になる）。FedSDAでない構成は、この軸で増やさない。
- 文字列から設定の値への変換は、計画のmoduleの関数`parse_fedsda_training_sample_selection(text) -> tuple[policy, K, α]`。不正な文字列は`ValueError`（コマンドは引数の誤り、終了コード2）。
- `build_planned_experiment_runs`は、run設定の型ごとに変種を作る: `FeddriftRunSettings`なら今の閾値の軸、`FedsdaRunSettings`なら`local_training_settings`だけを替えた抜き出し方の軸（軸の順）、それ以外は1つ。2つの軸は別の型にしか効かないので、同じ計画で両方を指定しても入れ子にならない。軸を指定しなければ、今と同じ1つ。
- コマンドの引数`--fedsda-training-sample-selections`（`nargs="+"`）。
- 掃引の記録: 計画（`asdict(sweep_plan)`）に新しい項目が増えるので、記録の形式の版（`SWEEP_RECORD_FORMAT_VERSION`）を4へ上げる。
- 一覧の表: 要約の`run_settings.settings_type`が`FedsdaRunSettings`の行に、列`training_sample_selection`（`uniform`・`recent:<K>`・`decay:<α>`の形）を足す。読む経路は`run_settings`→`run_participant_settings`→`run_client_settings`→`local_training_settings`で、ここまでは今のFedSDAの要約に必ずある。最後の3つの項目だけを`.get`で読み、選び方の項目が無い前の結果は空の欄にする（FedDriftの列のように、無い項目を`[]`で読むと、その行が読めない要約として表から外れるため）。行の判定は`run_settings.settings_type == "FedsdaRunSettings"`。

### golden
新しい3つの項目は、版1〜5のどれでも、既定の値（`uniform_over_all_stored_samples`、None、None）。`_CLIENT_SETTINGS_ADDED_AFTER_BASELINE_VERSION`の版1〜5へ、経路`("local_training_settings", 項目)`と既定の値を足す（版4・5の行を新しく作る）。`build_case_settings`は変えない（版の値を一般に入れる仕組みは無く、個別に書いた項目だけを置き換える。新しい項目は既定の値＝版の値なので、置き換えは要らない）。既定の値＝版の値であることは、比較の前の検査（`comparable_run_settings`）が確かめてから除く。helperの表のコメント（「その版の設定を組み立てるときは、この値を入れ」）は、この事実に合わせて直す。照合の値は変わらないので、基準版は増やさない。`BEHAVIOR_BASELINE_VERSION`も5のまま。

## 旧と違う点
FedSDAのrun設定の保存の形に3つの項目が増えるので、FedSDAのrunの識別子が全部変わる。前の結果の出力先へ、新しいコードで実行すると、前のrunは計画に入らない（識別子が違うので、済んだrunとしても、基準版の確認の対象としても扱われない）。前のrunのファイルは`runs/`に残り、一覧の表に載り、新しい列は空の欄になる。比べるときは、別の出力先を使う。既定の動作と値は変わらない。

## Validation Design
- 設定: 3つの選び方の正しい組合せを受け、不正な組合せ・型・範囲を拒否する。Kとbatchの件数の組合せ。
- 抜き出し: 既定（引数なし・既定の設定）は今と同じbatchと乱数の状態。直近K件は末尾K件からだけ抜き出す（K≥列の長さなら全部の標本から）。指数減衰は、重複なし、件数どおり、決まったseedで再現する。偏りの確認（決まったseed、列の長さ200・batch 8で、2000回の抜き出し）: 新しい半分の位置の選ばれた回数が、古い半分より多い。αが小さいほど、選ばれた位置の平均の「何件前か」が小さい。α=0.999では、新しい半分と古い半分の回数の比が1.2未満（ほぼ一様）。列の長さ＝batchの件数でα=0.5のとき、全部の位置を1回ずつ選ぶ。α=1e-200でも、件数どおり重複なしで選び（最新から順に選ばれる）、例外を出さない。乱数の消費が列の長さに依らない: 列の長さ200と100000（batch 32、α=0.5）で、抜き出しの後の乱数の状態が、同じseedから`random()`をbatchの件数だけ呼んだ後の状態と一致する（時間では測らない）。条件付き確率の確認: 列の長さ3・batch 1で、多数回の抜き出しの頻度が`α^a`の比に近い（許容差を、回数から決めた標準誤差の4倍とする）。
- 共同更新: 設定の選び方が抜き出しへ渡る（直近K件で、末尾K件の外の標本が使われない）。
- 掃引: 軸の変換と検査、変種の数と識別子の違い、FedSDAでない構成を増やさない、表の列、コマンドの引数。
- 回帰: goldenの版1〜5、実旧との照合、全pytest。FedSDAのrunの識別子の期待値の変更だけを許す。
- 手元の当たり付け（実装とspecの完了の後に行う計測。specの完了の条件には含めない）: sea2・circle2・sine2、seed 0〜2、間隔50（sea2は200も）で、`uniform`・`recent:300`・`decay:0.997`を比べ、試作の結果と同じ傾向かを確かめる。

## File Structure Plan
- 変更: 上の`This Spec Owns`のsource、`core/settings_field_validation.py`と、`tests/refactoring/`の対応するtest（`test_held_model_training_batch_sampling.py`、`test_held_model_joint_training_iterations.py`、設定のtest、掃引のtest、FedSDAのrunの識別子を期待するtest、依存境界）、`behavior_golden_cases.py`。
- 文書: `docs/experiments/experiment-sweep.md`、研究バックログの7、再開案内。

## Requirements Traceability
| 要求 | 設計 |
|---|---|
| 1.1–1.5 | 設定、抜き出し、共同更新の繰り返し |
| 2.1–2.3 | 掃引 |
| 3.1–3.2 | golden、抜き出しのtest |
| 4.1 | 文書 |
