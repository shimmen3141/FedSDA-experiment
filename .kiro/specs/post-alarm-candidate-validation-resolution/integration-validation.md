# 実測: 警報後の候補検証の確定

2026-10-07、共有../../venvのWindows CPU基準環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。主担当はClaude Code、独立レビューはcodex exec -m gpt-6-lunaで起動したレビュー担当。

## Task1

- RED: testを書いた後、実装ファイルが存在しない状態（runtime/にpost_alarm*が0件）でpytestを実行し、ModuleNotFoundError、1 collection error/1.79秒、exit1。実装はその後に書いた。
- GREEN: 実装後の初回実行で対象120 passed/4.24秒。
- 検出力の確認: `../../venv/refactoring-tests/post_alarm_candidate_validation_resolution_red_evidence.py`で実装を一時的に差し替え、対象testを実行し、元へ戻して実装のLF sha256 `9f50916e292da1527b7a7d4421559bd3c70cba0bb628d930e1109920bf472572`が変わらないことを確認した。

|差し替えた実装|結果|
|---|---|
|何もせず棄却を返すstub|117 failed/3 passed|
|棄却で吸収しない|6 failed|
|再利用で現在IDを切り替えない|18 failed|
|再利用で元の現行モデルへ吸収する|16 failed|
|維持と再利用の結果種別を取り違える|20 failed|
|棄却を維持と報告する|4 failed|
|採用で保留標本の概念も計数する|14 failed|
|採用を吸収で代用する|22 failed|
|再利用で吸収より先に現在IDを切り替える|6 failed|
|概念ID列の要素検証を省く|2 failed|
|正しい実装|120 passed|

- 実旧対照16条件（class2/4×旧4分岐×保留標本0/4件）。同じ損失列・履歴平均・閾値を移植済み評価関数へ与えて評価結果を作り、旧判定acceptedと一致することを確認してから適用。結果種別↔旧action↔旧戻り値（2/1/0）、帰属先、変更記録↔旧_on_local_model_changeの引数、切替位置の有無、採番次値、保有一覧と全モデルの値/optimizer、統計、送信保留、学習計数、割当概念計数、標本列を照合。採用では保留標本の概念を計数しないこと、採用以外では候補の接続先・候補optimizer・保有一覧・学習計数・採番が変わらないこと、乱数不変を確認。結果recordの不変性と結果種別の検証1。
- 拒否: 入力検証15種×4結果種別＝60（評価結果の型/派生型/フィールド型/採用と再利用の両立、現在ID ownerの型、保留標本列の型、概念ID列の型/長さ/要素）。委譲先の拒否15（採用: 待機・計数・候補・ラベル・使用済み一時ID・採番owner、再利用: 未保有の再利用先・複数行record・ラベル範囲外・registry型、維持/棄却: 特徴数・要素型・owner型）。各拒否で現在IDと採番次値を含む全状態の不変を確認。
- 呼出順4（採用は採用の組立だけ、再利用と維持は吸収→現在ID、棄却は吸収だけ）。
- AST RED: 注入契約追加直後23 failed/928 passed/1.22秒、exit1。exact16symbol guard追加後、対象＋AST 1071 passed。
- Ruff check成功/format 137files整形済み、Pyright基準venv明示0 errors/0 warnings。

## Task2

- 24条件（class2/4×Adam標準/AMSGrad/SGD×共有部更新有無×採用/再利用）は初回実行でGREEN。共同更新（parameterが変わる）→実旧確定処理/新の評価と確定→各自の標本storeの全標本を使う共同更新2回。loss、全値/grad、optimizer state、学習計数、標本列、統計、乱数が一致。
- fresh `../../venv/refactoring-tests/post_alarm_candidate_validation_resolution_cpu_smoke.py`成功、exit0。class2/4×4結果種別で学習→評価→確定→学習。旧importなし。
- Task1/Task2はレビュー担当がAPPROVED、独立1071 passed/smoke/Ruff成功、指摘なし。
