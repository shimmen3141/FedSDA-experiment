# 実測: 採用候補のローカル採用

2026-10-07、共有../../venvのWindows CPU基準環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。主担当はClaude Code、独立レビューはcodex exec経由のGPT-6 Luna。

## 設計時の確認

実旧`SharedBackboneClassConditionalESRFedSDAClient`へ設計に列挙した属性と実ForwardValidationSessionを与え、forward_persistentで実_finalize_forward_validationを実行できることを設計レビュー中に確認した（結果はresearch.md）。

## Task1

- RED: 未実装moduleによるModuleNotFoundError、1 collection error/1.88秒、exit1。
- GREEN初回90 passed/1 failed/3.78秒（-xで停止）。失敗は呼出順testで、登録関数のwrapperが参照する変数を後続のループが再束縛していたtest側の誤り。変数名を分けて修正。productionは無変更。
- 実旧対照48条件（class2/4×保有一覧(4,)/(9,-3,4)×保留標本0/1/4件×候補計数(0,0)/(11,3)×別ID既存保留なし/あり）と連続2回の採用2条件。実旧の確定処理が採用と判定したこと（戻り値2、判定recordのaccepted、session破棄）を前提確認し、一時ID、保有一覧、共有部と全モデルの値/grad/optimizer state、統計、送信保留と待機、学習計数、標本列の順序とTensor identity、現在ID、返す変更record、採番次値、旧_on_local_model_changeへ渡された(変更前ID, 変更後ID)を照合。
- 増分0の計数: 旧defaultdictは+=0でもkeyを作る。新record_completed_model_trainingも増分0でkeyを作り、候補計数(0,0)の条件で両実装の計数辞書（key順を含む）が一致した。
- 保留標本の追加で、既存モデルの統計・標本・計数と全モデルの割当概念計数が変わらないこと、新モデルの統計が登録時の初期統計のままであることを確認（実旧も採用分岐では_absorb_into_storeを通らず変えない）。torch/random/NumPy乱数不変。
- 拒否40条件: 本関数の検証点26（3owner＋現在IDownerの型11、2計数8、保留標本3、標本store/計数store2種/現在IDの使用済みID4）と登録の拒否代表14（一覧/統計/送信保留の使用済みID、保有0件、待機2、登録先owner型3、候補型、管理器不一致、特徴空、ラベル範囲外、非有限parameter）。各拒否で採番次値・一覧record identity・共有部値と接続先・全parameter/grad・全optimizer・統計・送信保留・標本store・計数store・現在ID・乱数の不変を確認。呼出順1（登録→採番→計数→標本→現在ID）。
- AST RED: 注入契約追加直後20 failed/854 passed/1.08秒、exit1。exact13symbol guard追加後、対象＋AST 977 passed/4.88秒。
- Ruff check成功/format 133files整形済み、Pyright基準venv明示0 errors/0 warnings。

## Task2

- 12条件（class2/4×Adam標準/AMSGrad/SGD×共有部更新有無）は初回実行でGREEN。保有2モデルの共同更新→独立した共有部を持つ候補だけの学習→実旧確定処理/新採用→3モデルの共同更新2回→実旧confirm/既存新confirm→(4,9,12)での共同更新。両実装とも各自の標本storeの全標本をモデルごとの固定batchにし、採用で追加された保留標本が新モデルの学習に使われる。各段階でloss、全値/grad、個別・共有optimizer state、共有参照、出力、学習計数、標本列、統計、乱数を照合。採用で現在IDが一時IDになるため、上流testで必要だった現在ID切替えのtest-only補助なしで正式ID確認へ接続できた。
- fresh `../../venv/refactoring-tests/adopted_candidate_local_adoption_cpu_smoke.py`成功、exit0。class2/4で保有モデル学習→候補学習→採番と採用（一時ID-101、次値-102、現在ID、計数、標本identity、既存統計不変）→標本storeからの後続学習→正式ID確認→学習。旧importなし。
- 実Luna Task1/Task2 APPROVED、独立977 passed/smoke exit0/Ruff成功、指摘なし。

## Task3

- 検証対象実装commit: af5c5b8。要求revision2・設計revision2・命名revision2のLF hashは承認値と一致。tasksは承認時hash（revision2）を維持し、check後hashを別fieldへ記録。
- tracked Python＋2goldenの233パスをパス順、パスUTF8＋NUL＋内容CRLF→LF＋NULで連結したSHA256: `8dd79ea76b1406ef2baf0787b8e0f6ca9f0cc58ab9d17ffa1b03b88661fdbc36`（前specの231パスに新module/新testの2件を加えた数）。
- 全適用対象Ruff成功/format133files、Pyright基準venv明示0 errors/0 warnings、pip check成功。
- 固定旧基準748c3aaからHEADへの、federated_drift_experiment/・2golden・旧回帰test2本・tools/の差分は空。
- 全pytest（主担当実測）: 5876passed/3skipped/1既存warning、125.64秒、exit0。前spec完了時5731に今回の145（対象103＋AST契約42）を加えた件数と一致。旧11条件と最終3条件の固定goldenを含む。goldenは更新していない。
- JUnit: `../../venv/refactoring-tests/adopted-candidate-local-adoption-full.xml`。
- 全pytestの独立再現はsteering/agent-handoff.mdの基準（2026-10-07ユーザー決定）に従い必須としない。Luna側での全pytest再現は試みていない。新全体runを実行したとは扱わない。

## 要件trace

|要求|証拠|
|---|---|
|1.1|実旧対照48条件の一時ID・一覧・値・統計・保留/待機、採番次値、連続2回の採用|
|1.2|学習計数の一致（増分0のkey生成を含む）、標本列の順序とTensor identity、保留標本0件で空の標本列|
|1.3|返す変更recordと実旧_on_local_model_changeの引数の一致、呼出順testで現在ID切替えが最後|
|1.4|既存モデルの統計/標本/計数と全モデルの割当概念計数の不変、新モデル統計が初期統計のまま、乱数不変、12条件の継続学習|
|2.1|3owner＋現在IDownerの型拒否11、全状態不変|
|2.2|2計数の拒否8、保留標本の拒否3、全状態不変|
|2.3|標本store/計数store（学習計数・割当概念計数）/現在IDの使用済みID拒否4、全状態不変|
|2.4|登録の拒否代表14で採番次値を含む全状態不変|
|2.5|呼出順test（登録→採番→計数→標本→現在ID）|
|3.1|状態なしruntime関数、返り値は変更recordだけ、exact13symbol AST guardと注入契約42|
|3.2|実旧確定処理の採用分岐との全状態一致、12条件で採用前学習→採用→学習→正式ID確認→学習の全数値一致、fresh新CPU smoke|

採否判定と判定記録、切替位置・検出エピソード・適応イベントの記録、現在ID変更の通知（予測重み再始動）、棄却/再利用/維持分岐、候補sessionの開始・収集・破棄、計算量診断、通信/new client/runは後続。新たな旧正常経路の不具合は観測していない。旧golden成功は新全体runの検証と区別する。
