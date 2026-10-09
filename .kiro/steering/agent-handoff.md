# Claude・Codex共通の引継ぎ手順

現在有効な規則だけを書く。決定の経緯・過去の事例・導入時の確認記録は[経緯の記録](agent-handoff-history.md)（読む必要はない。規則の理由を確かめたいときだけ参照する）。

## 作業場所と正本

両方のエージェントで、このリファクタリングworktreeを作業フォルダとして開く。元checkoutのmainとは区別する。Claude用に別のspecや進捗台帳を作らない。同じworktreeを編集する主担当は一度に一人とする。

| 情報 | 正本・入口 |
| --- | --- |
| 共通規約・方針 | [AGENTS.md](../../AGENTS.md)、[refactoring-policy.md](../../docs/research/refactoring-policy.md) |
| 現在地・次の候補 | [resume.md](resume.md)、[roadmap.md](roadmap.md) |
| 採用構成・依存方向・配置 | [product.md](product.md)、[tech.md](tech.md)、[structure.md](structure.md) |
| 個別仕様・進め方・状態 | 対象specのspec.json、requirements.md、design.md、tasks.md、naming.md（cc-sddの標準の構成と、命名表。必要に応じてbrief.md・research.md）。2026-10-09までのspecは、これにREADME・review・証拠文書を足した形 |
| レビューと検証証拠 | spec.jsonの承認とレビューの記録、tasks.md末尾の「Implementation Notes」（指摘の採否、検証結果、未検証）、Gitの対象commitと実diff |
| 旧実装の問題／改善候補 | [implementation-findings](../../docs/research/implementation-findings/README.md)／[improvement-candidates](../../docs/research/improvement-candidates/README.md) |

resumeは要約であり、承認状態を単独で決める根拠にしない。resumeに書くのは、現在地、次の一手、決定事項、実行環境の注意、未解消の事項、次の候補、未移植の細目、完了specの一覧（1行ずつ、リンクと一言）だけにする。完了specの検証commit・件数・レビューの経緯は各specの文書（tasks.mdの「Implementation Notes」。2026-10-09までのspecはintegration-validation.mdとreview.md）が正本で、resumeとroadmapへ写さない（同じ状態を複数の文書へ書くと更新漏れが起きるため）。Claudeの入口は[CLAUDE.md](../../CLAUDE.md)、CodexはAGENTS.mdからresumeへ進む。

## 開始時

1. 作業ディレクトリ、ブランチ（期待は`refactor/architecture`）、HEAD、`git status --short`を確認する。別作業の差分を破棄・上書き・一括stageしない。
2. resumeを読み、対象specのspec.json→tasks.md→requirements.md／design.md／naming.mdを読む（2026-10-09までのspecは、README→spec.json/tasks.md→要求/設計/命名→レビュー/検証）。未完了・レビュー待ちがあれば、新しいspecより先に扱う。
3. 未コミットの差分があれば、どのspecのどの下位taskのものかを、spec.jsonとtasks.mdに照らして確認する。別commitでのテスト成功を流用しない。
4. 交代前のセッションの編集・子エージェントの処理が終了していることを確認する。thread IDは証拠の来歴であり、再利用できる接続先ではない。

開始依頼の文面:

```text
このリファクタリングworktreeのAGENTS.md、.kiro/steering/resume.md、
.kiro/steering/agent-handoff.mdを読み、Gitと対象specから現在地を確認してください。
未完了・レビュー待ちを優先し、共有規約と、対象の難度に応じたLuna/Haiku 5.5の選択・相互代替による独立レビュー（全taskの後にまとめて1回）を維持して続けてください。
```

## 承認と独立レビュー

cc-sdd（kiro）のskillと標準の構成を基本とする（2026-10-10ユーザー決定）。進め方は次のとおり。(1)`/kiro-spec-init`→`/kiro-spec-requirements`→`/kiro-spec-design`→`/kiro-spec-tasks`で、requirements.md・design.md・tasks.mdを作る（各skillのSKILL.mdと`.kiro/settings/templates/specs/`の雛形に従う。調査が要るときは`/kiro-validate-gap`）。命名表（naming.md）は設計と一緒に作る。(2)`/kiro-impl <feature> --review off`の手順で、tasks.mdの下位taskを1つずつ、testを先に書いて実装する（下位taskごとのレビューはしない）。(3)全taskの後に、独立レビューを1回受ける。実装していない別sessionが、検証コマンドだけを許可した状態で、未コミットまたは対象commitの差分と3文書・命名表を読み、対象test・共用script・Ruffを独立に実行して、要求1項目ずつと実装・testの照合、旧との対応、検査の位置を確かめる。(4)指摘を反映し、`/kiro-validate-impl`の観点（全test、要求の網羅、設計との一致）で全pytestと品質検査を通して、完了にする。Blocker・Majorの指摘を直したときだけ、直した内容を別sessionが確認する。Minor・文面の指摘の反映は、採否を記録するだけで、再レビューを受けない。要求・設計・tasksの各段階の承認（spec.jsonの`approvals`）は、ユーザーの委任により、主担当が内容を確かめて記録する（段階ごとの独立レビューは置かない。最後の独立レビューが3文書も読む）。子エージェントは使わない（2026-10-10ユーザー決定。skillが子エージェントへの委譲を勧める箇所——調査、実装、レビュー——は、調査と実装を主担当自身が行い、レビューを下の独立CLIで行う）。独立レビューと、主担当による有用な指摘の反映を完了の条件とする。自己レビュー、cc-sddの`inline`、`kiro-spec-quick --auto`で独立レビューを置き換えない。

| 対象 | 優先モデル | 利用不能時 |
| --- | --- | --- |
| 日常的・分量の多いレビュー（命名/文書/要件、機械的証拠の照合など） | GPT-6 Luna（effort `medium`） | Claude Haiku 5.5 |
| やや複雑・難易度の高い実装レビュー（NN数値/RNG、可変状態、更新順序、複数部品の接続など） | Claude Haiku 5.5（effort `medium`） | GPT-6 Luna |

- 分量より難易度を優先し、対象の実際の判断難度から主担当が選ぶ。両方利用不能ならレビュー待ちとして対象specのspec.jsonへ記録して引き継ぐ。自己承認やSonnetへの自動代替はしない。
- effortは明示指定し、代替で使うときも各モデルの上の指定とする。指定が拒否されたら既定値で続行せず他方へ代替する。
- 毎回、実装担当の会話を継続せず独立したreviewerとして起動し、再委譲しない役割を明記する。permission bypassや認証・ユーザー設定の変更は行わない。
- 記録するもの: 選択理由、実際のモデルとeffort、代替時の利用不能の事実、対象revision/hash、結果、指摘の採否と理由。Lunaはログのmodel行と`reasoning effort`行、HaikuはJSONの`is_error=false`と`modelUsage`の実モデルで確認する（Haikuの実効effortは出力されないので指定値として記録する）。呼出しの成功だけを承認に数えない。
- 独立レビューの承認より前に、specを完了（spec.jsonのphaseを`completed`）にしない。過去の承認記録は書き換えない。
- 名前（2026-10-10ユーザー決定）: naming.mdに、sourceの公開する名前（module、class、公開の関数・メソッド）と、役割・似た名前との違いを書く。引数・field・局所名・非公開の名前・testの名前は書かない（`spec_checks.py names`も同じ範囲を照合する）。名前は、最後の独立レビューで確認を受ける（命名だけを先に承認する段階は置かない）。既存名の再利用は元と同じ役割にする。記録用wrapperに、判定を行う関数と読める名前を付けない。

起動例（PowerShell。依頼文は`$reviewPrompt`へ。複数行は単一引用符のhere-string）。下は読取りだけの形。レビュー担当にtestを実行させるとき（通常のレビュー）は、Haikuは`--tools Read,Glob,Grep,Bash`と`--allowedTools`で検証コマンドだけ（固定venvのpytest・Ruff・共用script・`spec_checks.py`、`git status`、`git diff`）を許可し、ファイルの変更やその他のコマンドは許可しない。Lunaは`--sandbox workspace-write`。

```powershell
claude -p $reviewPrompt --model claude-haiku-5-5 --effort medium --output-format json --no-session-persistence --tools Read,Glob,Grep --permission-mode plan
codex exec -m gpt-6-luna -c 'model_reasoning_effort="medium"' --sandbox read-only --ephemeral $reviewPrompt
```

### レビューに出す前と依頼文

- 機械的に照合できるものは先に`.kiro/settings/scripts/spec_checks.py`で確かめ、出力を依頼文へ貼る（使い方はscript冒頭）。`names`はsourceの公開する名前とnaming.mdの照合、`identity`は固定旧差分・source hash・作業ツリー・JUnitの照合、`progress`はspec.jsonのphaseとtasks.mdのcheckboxの対応の照合。名前の役割や検査の順序の正しさは判定しない。
- 依頼文に書くこと: (1)worktreeの絶対パス・期待するブランチ・基点のHEAD、未コミットの差分が対象であること。(2)対象ファイル、主担当の実測、手順上の逸脱。(3)実行してよい検証コマンド。(4)「自分のモデル名を確認できないことを判定理由にしない」こと。(5)出力形式（1行目に判定、番号付き指摘、独立に実行したもの・していないもの）。全pytestは、レビューの後に主担当が実行する（下の基準）。
- レビューの確認項目: (a)例外を出す検査（戻り値のrecordのconstructorを含む）がすべて最初の状態更新より前にあるか。上流の型が検査しないfieldに手で組み立てた値（boolのID、None、fieldの不対応）が入っても更新前に拒否されるか。(b)要求を1項目ずつ実装・testと照合し、実装側にだけある例外や、要求にあって実装にない条件がないか。(c)旧の処理を1行ずつ実装と照合し、design.mdに書いた「旧と違う点」以外の違いがないか。(d)testが実旧をoracleにしているか、比較していない状態や常に成功する比較がないか。
- 重大度: Minorは外から観測できる挙動または検出できる欠陥が変わるもの。書き方の好みは「任意」として分けて出させる。
- 指摘は全て採用する必要はない。前提が事実と違う指摘は根拠を示して不採用とし、理由をtasks.mdの「Implementation Notes」へ書く。却下の理由が実装・testにないときも、承認へ書き換えず、事実を示して再判定を依頼する。
- レビュー担当にtestを実行させた後は、`git status --short`が依頼前と同じであることを確かめる。
- レビュー依頼は対象段階・実diff・対応する要求と契約・関連する証拠を入口にする。上流の実装や過去specは判断に必要な箇所だけ追加で読む。役割が異なるレビューへ同じ全資料を毎回読ませない。
- 同じcommit・検証条件の機械照合は一度の出力を再利用できる。対象変更・証拠不足・独立確認が必要な場合は再実行する。

### 全pytestの独立再現の基準（2026-10-07ユーザー決定）

codex execのWindows sandboxでは一時ディレクトリの操作が拒否され、独立レビュー担当が全pytestを再現できないため、次の基準で判定する。

- 全pytest（旧11・最終3goldenを含む）は主担当が固定環境で実測し、件数・exit・JUnitの場所・対象commit・source hashを記録する。
- 独立レビュー担当は、対象test＋依存境界test・fresh CPU smoke・Ruffを独立実行し、JUnit集計とgolden回帰testcaseの成否、固定旧差分が空であること、承認hashを照合する。
- レビュー担当による全pytestの再現は必須としない。再現していない事実は対象specの記録へ明記し、再現済みと書かない。
- sandboxを外した実行（danger-full-access等）は、ユーザーの個別の明示承認なしに行わない。

## specの進め方

- specの大きさ（2026-10-09ユーザー決定）: 1つのspecは、旧の「ひとまとまりの流れ」（例: 標本1件の処理全体）に対応させ、つなぐ先まで含める。「呼ぶ位置は後続」のような未接続の境界を、specを分けるためだけに残さない。数値やアルゴリズムの一致が繊細な部分（損失・学習・乱数の順序など）は、そこだけ小さいspecへ切り出してよい。
- specの文書（2026-10-10ユーザー決定）: cc-sddの標準の構成（spec.json、requirements.md、design.md、tasks.md。skillが作る場合はbrief.md・research.md）に、naming.mdを足す。README・review.md・mutation-and-cpu-evidence.md・integration-validation.mdは作らない（2026-10-09までのspecは、作ったときの形のまま残す）。独立レビューの担当・session・判定はspec.jsonへ、指摘の採否と理由・検証結果（対象commit、全pytestの件数）・未検証と残る制約は、tasks.md末尾の「Implementation Notes」へ短く書く。次のspecへ引き継ぐ未検証事項は、再開案内の「次の候補」へ書く。文書のrevisionとhashは管理しない（commitが記録）。コードを読めば分かること（検査の一覧表、引数の一覧、局所名）と、testの条件数は、文書へ写さない。同じ状態を複数の場所へ書かない。
- tasks.mdは、cc-sddの基準（下位taskは1〜3時間の実行単位で、確かめられる成果物と、観測できる完了条件を持つ）に従って、責務の境界で下位taskへ分ける（1つの下位taskへ別の境界の変更を混ぜない）。実装は下位taskの順に進め、済んだ下位taskのcheckboxを付ける。段の間で止まらず、spec完了（独立レビューの承認と全pytest）を区切りにする。やむを得ず中断するときは、spec.jsonとresumeへ未完了の状態と次の一手を残す。
- 検査と数値の生成を状態変更より前に置く。旧と処理順が変わる場合は、成功時の観測値（値・順序・乱数の消費）が変わらないことを実旧との対照testで示し、research.mdへ理由を書く。
- 不正入力・上流状態・乱数の検証範囲は、その部品が受け取る入力と触れ得る状態から決める。関係しない状態の組合せまで網羅しない。上流の契約testを再利用するときは、その証拠が今回の接続・更新順も覆うか確認する。
- 旧の契約外入力での挙動（部分更新など）は、実旧で再現してからimplementation-findingsへ記録する。正常経路の非対称はユーザーへ確認する。処理の簡略化・効率化・局所的な調整の案は改善候補へ1候補1ファイルで記録し、旧挙動を維持する移植へ混ぜない。
- 固定旧実装とgoldenを環境差だけで更新しない。旧golden成功・部品の新旧一致・新全体runの一致を区別して記録する。未検証の成功や未レビューの承認を記録しない。

### testと検出力

- testを先に書く（`/kiro-impl`のTDD）。移植では、実旧との対照testを、その下位taskのtestとして書く。新しいmoduleだけの下位taskでは、moduleがないための収集失敗をREDとして記録しない。
- 変異テスト（`.kiro/settings/scripts/mutation_check.py`）は、既定では実行しない（2026-10-10ユーザー決定。時間に対して、sourceを変える発見がなかった）。入力の検査が多い関数など、主担当が必要と判断したときだけ実行する。そのときは、`--mutant-pytest-args`で速いtest（拒否・順序のtestなど）に絞り、5分以内に収める。未検出は、testの穴ならtestを足し、等価なら何もしない（文書へ理由を書かなくてよい）。実行した事実と結果の要点だけを、tasks.mdの「Implementation Notes」へ1行書く。
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

- 全pytestと品質検査は、独立レビューの指摘を反映した後に実行する。通ったらcommitし、そのcommitと件数をtasks.mdの「Implementation Notes」へ記録する（source・testを変えたら、やり直す）。下位taskごとのcommitは、対象testが通った時点で行ってよい。
- 長い検証の前に、必要な一時ファイル操作と実行環境を短い確認で確かめる。既知の権限エラーがある条件で全suiteを試し直さない。必要な権限は通常の承認手順で扱い、sandbox解除・認証変更を回避策にしない。
- source hashは、tracked Pythonと2goldenを、パス昇順でLF内容として連結したもの（`spec_checks.py identity`が計算する）。2026-10-09までのspecは、文書ごとの承認hashも持つ。
- 全pytestの件数は、前specの件数に今回追加したtest数を足した数と一致することを確かめる。

## Git・成果物・終了時

- commit/pushは対象ファイルを明示し、日本語メッセージ・AI coauthorなしで行う。保留中の3資料を自動でstageしない。commit後にGit状態を確認する。
- pushはcommitの区切りごとに通常pushを1回だけ試す。失敗時は原因探索・連続再試行をせず、次のpush成功時に未送信commitも送る。
- `../../venv/refactoring-tests/`（元checkoutのvenv配下、このPCだけ、Git管理外）に、各specの変異の`report.json`とlog、全pytestのJUnit、レビューの出力を置く（2026-10-09より前のspecは、個別のfresh CPU smokeと変異scriptもここにある）。別PCでは存在しないので、必要なら対象specのintegration-validation.mdから作り直す。results・保留資料・venvもGitで共有されるとは限らない。
- 中断・交代の前に、spec.jsonへ未完了の状態（進行中／レビュー待ち／検証待ち）と次の一手を記録し、resumeを更新する。

## cc-sdd

Codex Skills版（`.agents/skills/`）とClaude Skills版（`.claude/skills/`）はどちらも3.1.0。仕様・承認・方針は共通の`.kiro/`とdocsで管理する。更新するときは隔離した場所で生成してdiffを確認し、本worktreeへforce上書きで再導入しない（生成元のCLAUDE.mdと`.kiro/settings/`は取り込まない）。 2026-10-10から、文書の作成と実装は、cc-sddのskillの手順と雛形に従う（それまでは、形式だけを合わせて手で書いていた）。
