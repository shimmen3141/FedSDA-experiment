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

## Task3

- 検証対象実装commit: c53dfe0。要求revision1・設計revision2・命名revision2のLF hashは承認値と一致。tasksは承認時hash（revision2）を維持し、check後hashを別fieldへ記録。
- tracked Python＋2goldenの237パスをパス順、パスUTF8＋NUL＋内容CRLF→LF＋NULで連結したSHA256: `bf007838a3ee020c6901dcd593a49b0f45ab3d995371e378f5005ace8518f8a7`（前specの235パスに新module/新testの2件を加えた数）。
- 全適用対象Ruff成功/format137files、Pyright基準venv明示0 errors/0 warnings、pip check成功。
- 固定旧基準748c3aaからHEADへの、federated_drift_experiment/・2golden・旧回帰test2本・tools/の差分は空。
- 全pytest（主担当実測）: 6200passed/3skipped/1既存warning、128.38秒、exit0。前spec完了時6038に今回の162（対象120＋AST契約42）を加えた件数と一致。旧11条件と最終3条件の固定goldenを含む。goldenは更新していない。
- JUnit: `../../venv/refactoring-tests/post-alarm-candidate-validation-resolution-full.xml`。
- 全pytestの独立再現はsteering/agent-handoff.mdの基準（2026-10-07ユーザー決定）に従い必須としない。レビュー担当側での全pytest再現は試みていない。新全体runを実行したとは扱わない。

## 要件trace

|要求|証拠|
|---|---|
|1.1|旧create分岐との全状態一致、結果種別/帰属先/変更記録、採用を吸収で代用する誤実装の検出|
|1.2|旧reuse分岐との一致、現在ID切替えと変更記録↔旧通知引数、切替えない/旧モデルへ吸収する誤実装の検出|
|1.3|旧maintain分岐との一致、変更記録なし、種別取り違えの誤実装の検出|
|1.4|旧create_rejected分岐との一致、吸収しない/維持と報告する誤実装の検出|
|1.5|採用以外で候補接続先・候補optimizer・保有一覧・学習計数・採番・送信保留が不変、採用で概念計数が不変、乱数不変|
|2.1|評価結果7種と現在ID owner2種の拒否×4結果種別、全状態不変|
|2.2|保留標本列と概念ID列の拒否6種×4結果種別、全状態不変（採用での概念ID検証を省く誤実装の検出）|
|2.3|委譲先の拒否15（未保有の再利用先を含む）で現在IDと採番次値を含む全状態不変|
|2.4|呼出順4、吸収より先に現在IDを切り替える誤実装の検出|
|3.1|状態なしruntime関数と結果record、exact16symbol AST guardと注入契約42（評価関数・収集・設定・登録・登録確認・予測重みを拒否）|
|3.2|実旧_finalize_forward_validationの4分岐との一致、24条件の確定後学習、fresh新CPU smoke|

損失の収集と評価の呼出し、判定recordの保存、切替位置・検出エピソード・適応イベントの記録、学習帰属変更の通知（予測重み再始動）、候補sessionの開始と破棄、参照も学習させる方針の分岐、計算量診断、通信/new client/runは後続。新たな旧挙動の記録はない（採用時の概念計数の非対称は既存のLEGACY-014）。旧golden成功は新全体runの検証と区別する。
