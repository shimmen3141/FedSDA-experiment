# 実測: 警報時点の参照モデルの固定

2026-10-07、共有../../venvのWindows CPU基準環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。主担当はClaude Code、独立レビューはcodex exec -m gpt-6-lunaで起動したレビュー担当。

## 設計前の確認

実旧_snapshot_reference_modelsの後のtorch乱数状態が、新の分類器を保有順に同じ構造で生成した後の状態と一致した。実旧_begin_forward_validationを候補の学習だけ無効化して実行できた（結果はresearch.md）。

## Task1

- RED: testを書いた後、実装ファイルが存在しない状態でpytestを実行し、ModuleNotFoundError、1 collection error/1.83秒、exit1。実装はその後に書いた。
- 実装後の初回実行は24 passed/18 failed/3.68秒。原因は全てtest側の準備で、productionは初回から無変更。(a)12条件testで実旧clientのphase_secondsを共同学習より前に設定していなかった。(b)test helperが実旧の統計dictへ空のclass_statsを入れておらず、上流の比較helperがKeyErrorになった。(c)12条件testの共同更新で新側の学習計数を記録しておらず、共同学習で計数を加算する実旧と不一致になった。3点を直して42 passed/3.46秒。
- 検出力の確認: `../../venv/refactoring-tests/post_alarm_reference_model_fixation_red_evidence.py`で実装を一時的に差し替え、元へ戻して実装のLF sha256 `6dd12f9a53c5dd7d3c1a9bd115cc5aa18f061c127bf436850fc44e227f60b06b`が変わらないことを確認した。

|差し替えた実装|結果|
|---|---|
|空のrecordを返すstub|42 failed|
|保有モデルの分類器そのものを返す|24 failed|
|値を複製しない|24 failed|
|保有モデルと共有特徴抽出部を共有する|24 failed|
|一覧と逆の順で生成する|30 failed|
|分類器を1つ余分に生成する（乱数を余分に消費）|24 failed|
|値の検証より前に分類器を生成する|3 failed|
|件数1の統計も履歴平均に含める|22 failed|
|平均0の履歴を落とす|2 failed|
|統計未登録のモデルを0として含める|2 failed|
|保有0件を受理する|1 failed|
|正しい実装|42 passed|

- 実旧の参照複製との対照6条件（class2/4×保有一覧(4,)/(4,9)/(9,-3,4)）。同じtorch乱数状態から実旧と新を実行し、乱数が実際に消費されたことと処理後の状態が一致することを確認。参照IDの順序、全parameter値（実旧の参照と保有モデルの両方と一致）、出力、学習mode、requires_grad、grad未設定を照合。独立性: 共有特徴抽出部と全parameterの実体が保有モデル・他の参照と別で、固定後に保有モデルを書き換えても参照が不変。保有一覧のrecord identity・保有モデルのparameter/grad・全optimizer・統計・Python/NumPy乱数の不変。recordのfield付替え拒否。
- 履歴平均8条件（class2/4×統計4通り: 両方に履歴/1件は除外し平均0は保持/0件と1件/統計未登録）を、実旧_begin_forward_validation（候補の学習だけ無効化）のsessionの対応と値・順序で照合。
- 拒否10条件（一覧と統計storeの型各3、保有0件、非有限parameter3: 先頭/末尾/共有部）で、torch/random/NumPy乱数・保有モデル・一覧・統計の不変を確認。
- AST RED: 注入契約追加直後14 failed/1003 passed/1.05秒、exit1。exact6symbol guard追加後、対象＋AST 1059 passed/4.19秒。
- Ruff check成功/format 141files整形済み、Pyright基準venv明示0 errors/0 warnings。

## Task2

- session開始→観測→確定の照合6条件（class2/4×統計3通り）。実旧は実session開始（候補の学習だけ無効化）→観測4件→到達時の確定。新は候補（test側で現行モデルと同じ値の独立分類器を用意）→参照の固定→損失収集の開始→観測4件→評価（固定が返した履歴平均を使用）→確定。候補の生成と参照の固定を合わせたtorch乱数の消費が実旧のsession開始と一致。固定した参照が実旧sessionの参照と一致。結果種別は統計の与え方どおり現行維持/別モデル再利用/棄却となり、全owner状態が実旧と一致。
- 12条件（class2/4×Adam標準/AMSGrad/SGD×共有部更新有無）: 共同更新→固定→保有モデルをさらに2回共同更新→観測と確定（再利用）。保有モデルが学習で変わっても参照は固定時の値のままで、実旧sessionの参照とも一致し続ける。観測・評価・確定で乱数不変。全状態が実旧と一致。
- fresh `../../venv/refactoring-tests/post_alarm_reference_model_fixation_cpu_smoke.py`成功、exit0。class2/4で学習→固定（乱数消費、独立性、履歴平均は2件以上のモデルだけ）→学習継続中も参照不変→観測→評価→確定（再利用）→学習。旧importなし。
- Task1/Task2はレビュー担当がAPPROVED、独立1059 passed/smoke/Ruff成功、指摘なし。
