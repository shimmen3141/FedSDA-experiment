# Claude・Codex共通の引継ぎ手順

現在有効な規則だけを書く。決定の経緯・過去の事例・導入時の確認記録は[経緯の記録](agent-handoff-history.md)（読む必要はない。規則の理由を確かめたいときだけ参照する）。

## 作業場所と正本

両方のエージェントで、このリファクタリングworktreeを作業フォルダとして開く。元checkoutのmainとは区別する。Claude用に別のspecや進捗台帳を作らない。同じworktreeを編集する主担当は一度に一人とする。

| 情報 | 正本・入口 |
| --- | --- |
| 共通規約・方針 | [AGENTS.md](../../AGENTS.md)、[refactoring-policy.md](../../docs/research/refactoring-policy.md) |
| 現在地・次の候補 | [resume.md](resume.md)、[roadmap.md](roadmap.md) |
| 採用構成・依存方向・配置 | [product.md](product.md)、[tech.md](tech.md)、[structure.md](structure.md) |
| 個別仕様・承認・task | 対象specのREADME、spec.json、tasks.md、requirements.md、design.md、naming.md |
| レビューと検証証拠 | 対象specのreview.md、integration-validation.md、Gitの対象commitと実diff |
| 旧実装の問題／改善候補 | [implementation-findings](../../docs/research/implementation-findings/README.md)／[improvement-candidates](../../docs/research/improvement-candidates/README.md) |

resumeは要約であり、承認状態を単独で決める根拠にしない。resumeに書くのは、現在地、次の一手、決定事項、実行環境の注意、未解消の事項、次の候補、未移植の細目、完了specの一覧（1行ずつ、リンクと一言）だけにする。完了specの検証commit・件数・hash・レビューの経緯は各specのintegration-validation.mdとreview.mdが正本で、resumeとroadmapへ写さない（同じ状態を複数の文書へ書くと更新漏れが起きるため）。Claudeの入口は[CLAUDE.md](../../CLAUDE.md)、CodexはAGENTS.mdからresumeへ進む。

## 開始時

1. 作業ディレクトリ、ブランチ（期待は`refactor/architecture`）、HEAD、`git status --short`を確認する。別作業の差分を破棄・上書き・一括stageしない。
2. resumeを読み、対象specのREADME→spec.json/tasks.md→要求/設計/命名→レビュー/検証を読む。未完了task・レビュー待ちがあれば、新しいspecより先に扱う。
3. 対象revision・内容hashと承認記録が合うか確認する。過去の別revisionの承認や、別commitでのテスト成功を流用しない。
4. 交代前のセッションの編集・子エージェントの処理が終了していることを確認する。thread IDは証拠の来歴であり、再利用できる接続先ではない。

開始依頼の文面:

```text
このリファクタリングworktreeのAGENTS.md、.kiro/steering/resume.md、
.kiro/steering/agent-handoff.mdを読み、Gitと対象specから現在地を確認してください。
未完了・レビュー待ちを優先し、共有規約と対象の難度に応じたLuna/Haiku 5.5の選択・相互代替による独立レビュー条件を維持して続けてください。
```

## 承認と独立レビュー

要求→設計・命名→tasks→実装task→全回帰と証拠のtask、の各ゲートを維持する。別sessionのfeature最終レビューは行わない（2026-10-09ユーザー決定。それまでの最終レビューの指摘は、記録の更新漏れと、判定基準の読み違いによるものだった）。最後のtask（全回帰と証拠）のレビューが完了の判定を兼ねる: 実装taskのレビューに関わっていない別sessionが、要求を1項目ずつ実装・testと照合し、`spec_checks.py identity`と`progress`を独立に実行する。承認されたら、specを完了にする。主担当が必要と判断した場合（実装taskのレビューで採否の割れた指摘が残る、複数のspecにまたがる変更など）は、別sessionのレビューを1回足してよい。独立レビューと、主担当による有用な指摘の反映を承認として扱う。自己レビュー、cc-sddの自動承認・inline/off、`kiro-spec-quick --auto`、`-y`で置き換えない。[命名レビュー規約](../settings/rules/naming-review.md)に従う。

| 対象 | 優先モデル | 利用不能時 |
| --- | --- | --- |
| 日常的・分量の多いレビュー（命名/文書/要件、機械的証拠の照合など） | GPT-6 Luna（effort `medium`） | Claude Haiku 5.5 |
| やや複雑・難易度の高い実装レビュー（NN数値/RNG、可変状態、更新順序、複数部品の接続など） | Claude Haiku 5.5（effort `medium`） | GPT-6 Luna |

- 分量より難易度を優先し、対象の実際の判断難度から主担当が選ぶ。両方利用不能ならレビュー待ちとして対象specのreview.mdへ記録して引き継ぐ。自己承認やSonnetへの自動代替はしない。
- effortは明示指定し、代替で使うときも各モデルの上の指定とする。指定が拒否されたら既定値で続行せず他方へ代替する。
- 毎回、実装担当の会話を継続せず独立したreviewerとして起動し、再委譲しない役割を明記する。permission bypassや認証・ユーザー設定の変更は行わない。
- 記録するもの: 選択理由、実際のモデルとeffort、代替時の利用不能の事実、対象revision/hash、結果、指摘の採否と理由。Lunaはログのmodel行と`reasoning effort`行、HaikuはJSONの`is_error=false`と`modelUsage`の実モデルで確認する（Haikuの実効effortは出力されないので指定値として記録する）。呼出しの成功だけを承認に数えない。
- 未承認段階のapprovedをtrueにしない。未レビューのtaskを完了扱いしない。レビュー後に対象が変わったらrevision/hashを更新して承認を取り直す。過去の承認記録は書き換えない。
- 未承認の名前・仕様を先取りして実装しない。命名表へ登録するのは、sourceの名前（module、関数、class、引数、field、局所名）と、testのmodule直下の名前（test module、helper関数、定数、importの別名）。test関数の名前（`test_`で始まるmodule直下の関数。2026-10-09ユーザー決定）、test関数の中の局所名・引数・parametrize引数、条件名になるdictのkeyは登録しない（`spec_checks.py names`も同じ範囲を照合する）。既存名の再利用は元と同じ役割であることを示す。記録用wrapperに、判定を行う関数と読める名前を付けない。 例外（2026-10-09ユーザー決定）: 命名表を作るために、source・testをリポジトリ外で下書きし、作業ツリーの複製（`git archive`）で実行して、名前と実行可能性を確かめてよい。下書きはworktreeへ置かず、worktreeのtestも実行しない。複製での結果は承認の証拠に使わない（判定は、承認の後にworktreeへ適用した実装の実測とレビューによる）。下書きを作った事実はresearch.mdへ書く。

起動例（PowerShell。依頼文は`$reviewPrompt`へ。複数行は単一引用符のhere-string）。読取りレビュー用で、testを実行させるときは対象と権限を別途限定する（Lunaは`--sandbox workspace-write`）。

```powershell
claude -p $reviewPrompt --model claude-haiku-5-5 --effort medium --output-format json --no-session-persistence --tools Read,Glob,Grep --permission-mode plan
codex exec -m gpt-6-luna -c 'model_reasoning_effort="medium"' --sandbox read-only --ephemeral $reviewPrompt
```

### レビューに出す前と依頼文

- 機械的に照合できるものは先に`.kiro/settings/scripts/spec_checks.py`で確かめ、出力を依頼文へ貼る（使い方はscript冒頭）。`names`は束縛名と命名表の照合、`identity`は承認hash・固定旧差分・source hash・作業ツリー・JUnitの照合。`progress`はtasks.mdの完了数とspec.jsonの進捗・phaseの照合、状態を重ねて書いていないことの検査（下の「状態を書く場所」）、再開案内・roadmapに残る「待ち」「未実施」の表示で、最後のtaskのレビューへ出す前（再開案内を現在の状態へ更新してから）と完了の記録後に実行する。名前の役割や検査の順序の正しさは判定しない。
- 状態を更新する処理は、レビューの前に`.kiro/settings/scripts/mutation_check.py`を実行する（「検査を更新の後へ移す」変異を含む。下の「REDと検出力」）。
- 依頼文に書くこと: (1)worktreeの絶対パス・期待するブランチ・HEAD。(2)対象ファイルとLF hash、主担当の実測、手順上の逸脱。(3)全pytestの判定基準の原文（下記）。(4)「自分のモデル名を確認できないことを判定理由にしない」こと。(5)出力形式（1行目に判定、番号付き指摘、独立に実行したもの・していないもの）。
- 設計レビューの確認項目: (a)例外を出す検査（戻り値のrecordのconstructorを含む）がすべて最初の状態更新より前にあるか。上流の型が検査しないfieldに手で組み立てた値（boolのID、None、fieldの不対応）が入っても更新前に拒否されるか。(b)要求を1文ずつ設計の条件と照合し、設計側にだけある例外句がないか。
- 実装レビューでは(a)(b)を1回目から実コードで確認させる。変異は`report.json`の場所と、未検出の変異とその扱い（testを足した、または等価と判断した理由）を渡し、検出力の指摘は「どの変異でも覆われていない検査」に限らせる。
- 重大度: Minorは外から観測できる挙動または検出できる欠陥が変わるもの。書き方の好みは「任意」として分けて出させる。
- 指摘は全て採用する必要はない。前提が事実と違う指摘は根拠を示して不採用とし、理由をreview.mdへ書く。NO-GOの理由が実装・testにないときも、GOへ書き換えず、事実を示して再判定を依頼する。
- レビュー担当にtestを実行させた後は、`git status --short`が依頼前と同じであることを確かめる。
- レビュー依頼は対象段階・実diff・対応する要求と契約・関連する証拠を入口にする。上流の実装や過去specは判断に必要な箇所だけ追加で読む。役割が異なるレビューへ同じ全資料を毎回読ませない。
- 同じcommit・承認hash・検証条件の機械照合は一度の出力を再利用できる。対象変更・証拠不足・独立確認が必要な場合は再実行する。最後のtaskのレビュー担当は統合判定に集中し、既に独立再現された検証の重複実行は必要性を判断する。

### 全pytestの独立再現の基準（2026-10-07ユーザー決定）

codex execのWindows sandboxでは一時ディレクトリの操作が拒否され、独立レビュー担当が全pytestを再現できないため、次の基準で判定する。

- 全pytest（旧11・最終3goldenを含む）は主担当が固定環境で実測し、件数・exit・JUnitの場所・対象commit・source hashを記録する。
- 独立レビュー担当は、対象test＋依存境界test・fresh CPU smoke・Ruffを独立実行し、JUnit集計とgolden回帰testcaseの成否、固定旧差分が空であること、承認hashを照合する。
- レビュー担当による全pytestの再現は必須としない。再現していない事実は対象specの記録へ明記し、再現済みと書かない。
- sandboxを外した実行（danger-full-access等）は、ユーザーの個別の明示承認なしに行わない。

## specの進め方

- 1つのspecは、旧の1メソッドまたはその一部に対応する小さな単位にする。文書はREADME・brief・requirements・design・naming・tasks・research・review・mutation-and-cpu-evidence・integration-validation・spec.json。spec.jsonの既存構造を守り、別台帳を増やさない。
- 状態を書く場所（2026-10-09ユーザー決定。同じ状態を複数の文書へ書くと、段階が進むたびに書き換え漏れが出る）: specの現在の状態（どのrevisionが承認済みか、どのtaskが済んだか、完了か）は、spec.json（approvals、implementation_progress、phase、task_reviews）とtasks.mdのcheckboxだけに書く。レビューの経緯と採否はreview.mdだけに書く。READMEは文書の案内だけにし、状態の行を置かない。integration-validation.mdは検証の事実（対象commit、件数、hash、要求対応、未検証事項）だけにし、レビューの判定や担当を書かない。再開案内は「現在地」の1行と完了specの一覧（リンクと一言）だけを更新する。`spec_checks.py progress`が、READMEの状態の行とintegration-validation.mdのレビューの節を見つけたら失敗にする（この規則より前からあるspecは対象外）。
- 設計とtasksには、testの条件数・変異の種類の一覧・件数を書かない（確かめる観点だけを書く）。数は証拠文書（mutation-and-cpu-evidence.md、integration-validation.md）に書く。testの条件を足しただけで承認済みの文書のrevisionを上げることにならないようにするため。
- 要求・設計・命名・tasksは1回の依頼でまとめてレビューに出してよい。判定は段階ごとに受け、spec.jsonへ段階ごとのrevision・LF hash・sessionを記録する。改訂した段階だけ再レビューする。
- 文面だけの指摘は、修正して再レビューを省いてよい（2026-10-09ユーザー決定）。対象は、レビューの指摘が文書の記述の正確さ・明確さだけに関わり、反映しても処理・契約・検査の順・例外・testの方針・要求の意味・source・testが変わらないもの。主担当が修正し、review.mdへ指摘ごとの修正内容を書き、spec.jsonへ修正後のrevisionとhashを、最後にレビューを受けたrevision・sessionと「文面だけの修正」である旨とともに記録する。次のどれかに当たる場合は再レビューを受ける: Blocker・Majorの指摘、処理・契約・要求の意味・testを変える修正、指摘を不採用にする場合、文面だけかどうか迷う場合。省いた事実は次のレビュー（task、feature最終）の依頼文へ書き、そのレビューが修正後の文書を読む。
- tasksの分け方（2026-10-09ユーザー決定）: 実装taskと、最後の「全回帰と証拠」のtaskに分ける。実装taskは、cc-sddの基準（下位taskは1〜3時間の実行単位で、確かめられる成果物と、観測できる完了条件を持つ）に従って、実装を進める順に下位task（1.1、1.2…）へ分ける。例: test→source→依存の許可集合と共用script→変異toolによる検出力の確認。関数1つ・module 1つ程度で全体が1〜3時間に収まるspecは、実装taskを1つにしてよい（その場合も、進める順を箇条書きで書く）。検出力の確認は実装taskに含め、独立したtaskにしない。レビューは最上位のtaskごとに受ける（下位taskごとには受けない）。task作成前に止まらず、spec完了（最後のtaskの承認）を区切りにする。やむを得ず中断するときは、対象specとresumeへ未完了の状態と次の一手を残す。
- 検査と数値の生成を状態変更より前に置く。旧と処理順が変わる場合は、成功時の観測値（値・順序・乱数の消費）が変わらないことを実旧との対照testで示し、research.mdへ理由を書く。
- 不正入力・上流状態・乱数の検証範囲は、その部品が受け取る入力と触れ得る状態から決める。関係しない状態の組合せまで網羅しない。上流の契約testを再利用するときは、その証拠が今回の接続・更新順も覆うか確認する。
- 旧の契約外入力での挙動（部分更新など）は、実旧で再現してからimplementation-findingsへ記録する。正常経路の非対称はユーザーへ確認する。処理の簡略化・効率化・局所的な調整の案は改善候補へ1候補1ファイルで記録し、旧挙動を維持する移植へ混ぜない。
- 固定旧実装とgoldenを環境差だけで更新しない。旧golden成功・部品の新旧一致・新全体runの一致を区別して記録する。未検証の成功や未レビューの承認を記録しない。

### REDと検出力

- testは実装より先に書く。新しいmoduleのtestが「moduleがない」ために収集失敗するだけのREDは記録しない。既存の処理を変えるspecでは、変更後のtestを変更前のsourceで実行して、失敗するtestを記録する。依存境界は、guardを登録する前に失敗することを確かめる。仕様化の段階ではworktreeのtestを実行しない。
- GREENの後、`.kiro/settings/scripts/mutation_check.py`で検出力を確かめる（使い方はscript冒頭）。specごとに変異scriptを書かない。scriptは、対象関数の文から、検査や呼出しを1つ消す・例外を出しうる文を後続の代入の後へ移す・隣り合う文を入れ替える・exact型検査をisinstanceへ緩める、の変異を機械的に作り、1種ずつ対象testを実行して元byteへ戻す。機械的に作れない変異（引数の差替えなど）は、その接続に固有の危険があるときだけ`--extra`で足す。
- 未検出の変異は、testの穴か等価な変異（例: 読取りだけの代入の後へ検査を移した）のどちらかである。穴ならtestを足す。等価なら理由を証拠文書へ書く。証拠文書には、実行したコマンド、検出数/総数、未検出の変異とその扱いを書き、変異ごとの表は載せない（`report.json`が正本）。
- 対象は、そのspecで新しく書いた関数と変更した関数だけにする。変更していない関数や過去specの変異を再実行しない。変異scriptは実装ファイルを書き換えるので、レビュー担当には実行させず`report.json`を読ませる。
- 新実装だけで動くことの確認（fresh process）は、specごとにscriptを書かず、Git管理下の共用script（`tests/refactoring/fresh_process_smoke.py`。旧実装とtest moduleをimportしない）へ、そのspecの接続を通る流れを足して育てる。全pytestから別processで実行するtestを置き、sourceの変更で動かなくなったらすぐ分かるようにする。2026-10-09より前のspecの個別scriptはGit管理外の当時の証拠で、現在のsourceでは動かないものがある。新全体runを接続したら、全体runのtestへ置き換えて廃止する。
- 依存境界は、新しいmoduleの許可集合を、既存と同じ形（`dependency_is_allowed`の中のmoduleごとの名前の組と、exact検査の対象の一覧）で登録するだけにする。symbolごとに5通りの書き方を並べる注入契約testは新しく足さない（import module・private・starの拒否は、symbolによらない共通の仕組みで、既存のtestが確かめている。許可は、実際のsourceがimportして全体の依存testが通ることで確かめられる）。許可が広すぎないこと（許可集合の各名前が実際にimportされていること）は、全moduleを対象にした1件のtestが確かめる。そのspecで特に禁じたい依存があれば、拒否の例を数件だけ足す。既存の登録と注入契約testは移さず、消さない（2026-10-09ユーザー決定。方式の一本化は見送り。[IMPROVE-009](../../docs/research/improvement-candidates/improve-009-unify-dependency-boundary-tests.md)）。

### 実旧を使うoracle

- 式や手順をtestへ複製して正解にしない。実旧clientを`__new__`で作り、対象メソッドが読む属性だけを与えて実旧メソッドを実行する。新しいメソッドを使うときは、REDより前に実行できることを確かめてresearch.mdへ書く。
- test helperは上流のtestから再利用する（`tests/refactoring/`内で互いにimportする）。入口は`test_adopted_candidate_initial_local_registration.py`の`build_initial_registration_oracle`から、`test_alarm_response_completion.py`の`build_response_completion_oracle`まで連なる。どのhelperが何を差し替えているかは各testのdocstringと各specのresearch.mdにある。
- モデルの生成は初期化でCPUのtorch乱数を消費する。生成の順序と個数を旧と一致させる。乱数の不変を確かめるときは、モデル生成後の状態を基準にし、torch・Python・NumPyの状態全体を比べる。
- 学習の継続は、実旧の共同学習と全値・grad・optimizer state・乱数を照合する形を続けている。

## 検証コマンド

worktreeルートで実行する（Git Bash）。PowerShellでは`$env:NAME='値'`で同じ環境変数を設定し、`$PY`を`../../venv/Scripts/python.exe`へ読み替える。このPCの共有Pythonは`../../venv/Scripts/python.exe`。日本語ファイルを読むPowerShellでは`Get-Content -Encoding UTF8`を使う。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMP="$(cd ../../venv/refactoring-tests && pwd -W)"; export TEMP="$TMP"
export MPLCONFIGDIR="$(cd ../../venv/matplotlib-cache && pwd -W)"
export FDE_MNIST_DATA_DIR="$(cd ../../data/mnist && pwd -W)"
PY=../../venv/Scripts/python.exe

$PY -m pytest tests/refactoring/test_<対象>.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
$PY -m pytest tests -q -p no:cacheprovider --junitxml="$TMP/<feature名>-full.xml"   # 約3分
$PY -m ruff check src tests/refactoring; $PY -m ruff format --check src tests/refactoring
$PY -m pyright --pythonpath ../../venv/Scripts/python.exe; $PY -m pip check
$PY .kiro/settings/scripts/spec_checks.py identity <feature名> --junit "$TMP/<feature名>-full.xml"
```

- 全pytestと品質検査は、全taskの実装をcommitした後に実行し、そのcommitを検証対象として記録する。検証時に未コミット差分がないこと。
- 長い検証の前に、必要な一時ファイル操作と実行環境を短い確認で確かめる。既知の権限エラーがある条件で全suiteを試し直さない。必要な権限は通常の承認手順で扱い、sandbox解除・認証変更を回避策にしない。
- 承認hashは各mdのLF sha256（tasksは完了のcheckboxを未完了へ戻した内容）。source hashはtracked Pythonと2goldenを、パス昇順でLF内容として連結したもの。どちらも`spec_checks.py identity`が計算・照合する。
- 全pytestの件数は、前specの件数に今回追加したtest数を足した数と一致することを確かめる。

## Git・成果物・終了時

- commit/pushは対象ファイルを明示し、日本語メッセージ・AI coauthorなしで行う。保留中の3資料を自動でstageしない。commit後にGit状態を確認する。
- pushはtaskごとに通常pushを1回だけ試す。失敗時は原因探索・連続再試行をせず、次taskのpush成功時に未送信commitも送る。
- `../../venv/refactoring-tests/`（元checkoutのvenv配下、このPCだけ、Git管理外）に、各specの変異の`report.json`とlog、全pytestのJUnit、レビューの出力を置く（2026-10-09より前のspecは、個別のfresh CPU smokeと変異scriptもここにある）。別PCでは存在しないので、必要なら対象specのintegration-validation.mdから作り直す。results・保留資料・venvもGitで共有されるとは限らない。
- 中断・交代の前に、対象specへ「完了済み/進行中/レビュー待ち/検証待ち」と実差分・次の一手を記録し、resumeを更新する。

## cc-sdd

Codex Skills版（`.agents/skills/`）とClaude Skills版（`.claude/skills/`）はどちらも3.1.0。仕様・承認・方針は共通の`.kiro/`とdocsで管理する。更新するときは隔離した場所で生成してdiffを確認し、本worktreeへforce上書きで再導入しない（生成元のCLAUDE.mdと`.kiro/settings/`は取り込まない）。
