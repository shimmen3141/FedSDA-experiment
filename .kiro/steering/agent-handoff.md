# Claude・Codex共通の引継ぎ手順

## 作業場所と正本

両方のエージェントで、このリファクタリングworktreeを作業フォルダとして開く。
元checkoutのmainとは区別する。Claude用に別のspecや進捗台帳を作らない。

| 情報 | 正本・入口 |
| --- | --- |
| 共通規約・方針 | [AGENTS.md](../../AGENTS.md)、[refactoring-policy.md](../../docs/research/refactoring-policy.md) |
| 現在地・次の候補 | [resume.md](resume.md)、[roadmap.md](roadmap.md) |
| 採用構成・依存方向・配置 | [product.md](product.md)、[tech.md](tech.md)、[structure.md](structure.md) |
| 個別仕様・承認・task | 対象specのREADMEにある正本一覧、spec.json、tasks.md、requirements.md、design.md、naming.md |
| レビューと検証証拠 | 対象specのreview.md、integration-validation.md等、Gitの対象commitと実diff |
| 旧実装の問題 | [implementation-findings](../../docs/research/implementation-findings/README.md) |
| 開発手順の問題 | worktreeのdevelopment-findings/ |

resumeは要約であり、承認状態を単独で決める根拠にしない。Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。
CodexはAGENTS.mdからroadmap・resumeへ進む。共通方針を入口ファイルへ複製しない。

## 開始時

1. 作業ディレクトリ、`git branch --show-current`、`git rev-parse HEAD`、`git status --short`を確認する。
   期待するブランチは`refactor/architecture`。元checkoutや別ブランチなら実装前に作業場所を確認する。
2. resumeとroadmapを読み、対象specのREADME→spec.json/tasks.md→要求/設計/命名→レビュー/検証を読む。
   未完了task・レビュー待ちがあれば、新しいspecより先にその状態を復元する。
3. 既存の未コミット差分の作成者・対象を確認する。別作業の差分を破棄・上書き・一括stageしない。
4. 対象revision・内容hashと承認記録が合うか確認し、差分と検証対象commitを照合する。
   過去の別revisionの承認や、別commitでのテスト成功を流用しない。
5. 必要なcc-sdd skillを読み、共通規約に従って次の一単位を進める。

同じworktreeを編集する主担当は一度に一人とする。交代前に旧セッションの編集・子エージェントの処理が終了していることを確認する。
セッション固有のthread IDは証拠の来歴であり、次のツールで再利用できる接続先ではない。

## 承認と独立レビュー

要求→設計・命名→tasks→実装task→統合検証・別feature最終GOの流れを維持する。
[命名レビュー規約](../settings/rules/naming-review.md)と各specの承認契約に従う。
2026-10-08のユーザー指示により、次の選択を担当ツールにかかわらず共通に適用する。独立レビューと主担当による有用な指摘の反映を承認として扱う。要求・設計・命名・tasks・実装task・別feature最終GOの各ゲートは維持する。

| 対象 | 優先モデル | 利用不能時 |
| --- | --- | --- |
| 日常的・分量の多いレビュー（通常の命名/文書/要件、機械的証拠の照合など） | GPT-6 Luna | Claude Haiku 5.5 |
| やや複雑・難易度の高い実装レビュー（NN数値/RNG、可変状態、更新順序、複数部品の接続など） | Claude Haiku 5.5 | GPT-6 Luna |

分量より難易度を優先し、対象の実際の判断難度から主担当が選ぶ。段階名だけで一律に選ばない。独立した別feature最終担当も、その対象の難度で選ぶ。両方利用不能ならレビュー待ちとして引き継ぎ、自己承認やSonnetへの自動代替はしない。
利用可能な接続・モデルを確認し、モデルを明示してレビュー担当を起動する。
優先モデルの選択理由・実際のモデル/effort・代替時の利用不能理由・対象revision/hash・結果・指摘の採否を記録する。接続なし、呼出上限、認証/実行エラー等を事実として残し、担当ツールだけを代替理由にしない。
自己レビューやcc-sddの自動承認・inline/offで置き換えない。`kiro-spec-quick --auto`や`-y`も承認を代替しない。
過去のLuna/Sonnet等の承認記録は変更しない。

### CodexからClaudeを呼ぶ手順と確認済み事項
次の例から対象に応じた優先モデルのコマンドを一つだけ実行し、利用不能時だけ他方へ切り替える。

2026-10-08、Claude Code 2.1.293をCodexから非対話実行し、`--model claude-haiku-5-5`のJSON応答はsuccess/is_error=false、modelUsageの実モデルもclaude-haiku-5-5であることを確認した。aliasや自分のモデル名に関する回答だけで同定しない。証拠は元checkoutのvenv/refactoring-tests/haiku-5-5-call-check.json。

```powershell
claude -p $reviewPrompt --model claude-haiku-5-5 --effort high --output-format json --no-session-persistence --tools Read,Glob,Grep --permission-mode plan
codex exec -m gpt-6-luna -c 'model_reasoning_effort="high"' --sandbox read-only --ephemeral $reviewPrompt
```

実装担当の会話を継続/resumeせず、毎回独立したreviewerとして起動し、再委譲しない役割を明記する。上記は読取りレビュー用で、コマンド実行による再現が必要なら対象と権限を別途限定する。permission bypassや認証・ユーザー設定の変更は行わない。出力は必要な証拠だけを保存し、秘密情報を含めない。モデル未確認、is_error=true、単なる呼出し成功は対象レビューの承認に数えない。

依頼文は対象・revision/hash・役割・確認事項を含む`$reviewPrompt`に格納する。PowerShellで複数行を渡す場合は単一引用符のhere-string（`@'`と`'@`を独立した行へ置く）を使うと、依頼中の引用符や`$`を展開せず渡せる。
2026-10-08の追加指示により、LunaとHaikuは両方ともeffortを`high`に明示指定し、代替時も維持する。Lunaのログでモデル名と`reasoning effort: high`を確認する。Haikuは`--effort high`の指定と実モデルを記録し、JSONが実効effortを出力しない場合は指定値と実効値未確認を区別する。指定が拒否された場合は既定値で続行せず、他方へ同じhigh指定で代替する。
直近alarm-buffer-responseのLunaレビュー15件はログのreasoning effortが全てmediumで、起動コマンドにはeffortの上書きがなかった。これは過去の記録であり、過去全セッションの設定を一律には断定しない。

今回の規約変更の確認記録（2026-10-08、基点`7cbb4a5`、この記録を追記する前の5文書を対象）：
- 対象hash（AGENTS.md、CLAUDE.md、naming-review.md、agent-handoff.md、resume.mdの順に相対パスとファイルbytesを連結したSHA-256）：`de4517df0b68f92847ffe5608e8b6bcaab4da2514058b63df3df26a0f943925b`。
- 通常の文書整合性レビューとしてGPT-6 Lunaを選択し、ログの実モデル`gpt-6-luna`・`reasoning effort: high`を確認。Luna固定の再開案内と、CLI例が両方実行に読める点を修正し、独立再レビューでPASS。代替は行っていない。
- Haikuは`--model claude-haiku-5-5 --effort high`の呼出しで`is_error=false`、実モデル`claude-haiku-5-5`、応答`HIGH_CALL_OK`を確認。JSONに実効effort値は出ておらず、highは指定値として記録する。この疎通確認自体をレビュー承認には用いていない。
- 生ログは基準checkoutの`venv/refactoring-tests/review-policy-luna-high.log`、`review-policy-luna-high-r2.log`、`haiku-high-call-check.json`。文書変更のみでコード・goldenのテストは再実行していない。

指定された独立レビュー担当を利用できないセッションでは、対象specの既存review.mdへ次を記録する。

- レビュー待ちの段階・task番号、対象ファイル、revision・内容hash、対象commit/未コミット差分。
- 確認してほしい契約・命名・判断、実施した検証と未実施項目。
- 前提の承認状態、未解決の指摘、レビュー後に進める具体的な次作業。

未承認段階のapprovedをtrueにせず、未レビューの実装taskを完了扱いしない。
task briefではtest用module別名と動的type名も新規名として照合する。既存名の同義再利用なら元specと同じ役割を示し、新しい別名ならコード追加前に命名表へ登録して独立承認を受ける。
2026-10-08の事例: 警報区間再利用評価のTask 2で、testを書いてから局所名（helper内の関数、state記録のkey、一時変数）を命名表へ事後登録し、独立レビューで手順逸脱と指摘された。testを書く前に、helper・parametrize引数・定数・helper内の関数・記録用の局所名まで洗い出して登録する。記録用wrapperに、選択や判定を行う関数と読める名前を付けない。2026-10-08の事例2: 警報の変化区間の解決では、testをリポジトリ外で下書きして名前を事前登録したが、目視の洗い出しで7つ漏れた。下書きをASTで走査し（関数名・引数名・代入名・内包表記の変数）、命名表の全文と機械的に照合する。条件名（parametrizeのidになるdictのkey）はtest側のdictを正本とし、個別には登録しない。
独立した承認済み作業は進められるが、未承認の名前・仕様を先取りして実装しない。
独立レビューを実行可能なセッションへ記録を引き継ぎ、実レビューの結果と指摘の採否・理由を保存する。
レビュー後に対象が変わった場合はrevision/hashを更新し、必要な承認を取り直す。

### レビュー依頼の例

```text
このworktreeのresume.mdと対象specのREADME・spec.json・review.mdを読み、
review.mdに記録したレビュー待ち対象を、日常的・大量ならLuna、やや複雑・難易度の高い実装ならHaiku 5.5で独立レビューさせてください。
優先モデルを利用できない場合は他方で代替し、選択理由・利用不能理由・実際のモデル/effortを記録してください。
対象revision/hashと実差分を照合し、有用な指摘を反映・再確認してください。
承認状態と次の作業を正本へ記録してください。
```

### 全pytestの独立再現の基準（2026-10-07ユーザー決定）

codex execのWindows sandboxでは、pytestの一時ディレクトリ走査と一時ファイル書込みがWinError 5で拒否され、
独立レビュー担当が全pytestを再現できない。ユーザーの明示決定により、今後も次の基準でTask3とfeature最終GOを判定する。

- 全pytest（旧11・最終3goldenを含む）は主担当が固定環境で実測し、件数・exit・JUnitの場所・対象commit・source hashを記録する。
- 独立レビュー担当は、対象test＋依存境界test・fresh CPU smoke・Ruffを独立実行し、JUnit集計とgolden回帰testcaseの成否、固定旧差分が空であること、承認hashを照合する。
- レビュー担当による全pytestの再現は必須としない。再現していない事実は対象specの記録へ明記し、再現済みと書かない。
- sandboxを外した実行（danger-full-access等）は、ユーザーの個別の明示承認なしに行わない。

## 検証・成果物・終了時

- 2026-10-08ユーザー訂正指示: 具体的な処理簡略化・計算/通信等の効率化・局所的な調整は[改善候補](../../docs/research/improvement-candidates/README.md)へ1候補1ファイルで記録する。広い研究方向は研究バックログ、不具合の疑いはimplementation-findings。事実・仮説・変更案・比較条件/指標/悪化の懸念・採否を分け、旧挙動維持の移植へ混ぜない。同率現行優先はIMPROVE-001（旧ALGO-001）、未採用/未検証。
- 2026-10-08ユーザー訂正指示: task作成前に停止する必要はなく、spec終了まで進める。Codexの5時間枠残量を取得できないことだけで要求/設計/命名の途中終了をしない。各独立レビューと実装・検証を完了して、spec完了を区切りにする。やむを得ず中断した場合は正本へ未完了の状態と次の一手を残す。

- 検証手順は[code-quality.md](../../docs/research/code-quality.md)、
  [refactoring-baseline.md](../../docs/experiments/refactoring-baseline.md)、対象specを参照する。
  このPCの共有Pythonはworktreeから`../../venv/Scripts/python.exe`。別PCでは基準環境を構築して選択する。
- Pyrightには必要なら選択したPythonを`--pythonpath`で指定する。
  Windows sandboxでルート探索にアクセス拒否がある場合、Ruffの現行全対象は`src tests/refactoring`。
  日本語ファイルを読むPowerShellでは`Get-Content -Encoding UTF8`を使用する。
- 固定旧実装とgoldenを環境差だけで更新しない。旧golden成功・部品の新旧一致・新全体runの一致を区別して記録する。
- results、保留中の3資料、venv、ローカルのログ/JUnitはGitで共有されるとは限らない。
  検証記録へ結果・環境・対象commitを残し、別PCで必要な証拠がなければ再実行または証拠の所在を確認する。
- 中断前に対象specへ「完了済み/進行中/レビュー待ち/検証待ち」と実差分・次の一手を記録し、resumeを更新する。
  未検証の成功や未レビューの承認を記録しない。spec.jsonの既存構造を守り、別台帳を増やさない。
- 許可済みのcommit/pushは対象ファイルを明示し、日本語メッセージ・AI coauthorなしで行う。
  commit後にGit状態とpush結果を確認する。保留資料を自動でstageしない。
- 2026-10-08ユーザー指示: GitHub障害時はtaskごとに通常pushを1回試す。失敗の原因探索・連続再試行はせず、次taskのpush成功時にそのブランチの未送信commitも送る。

### 次の主担当への開始依頼

```text
このリファクタリングworktreeのAGENTS.md、.kiro/steering/resume.md、
.kiro/steering/agent-handoff.mdを読み、Gitと対象specから現在地を確認してください。
未完了・レビュー待ちを優先し、共有規約と対象の難度に応じたLuna/Haiku 5.5の選択・相互代替による独立レビュー条件を維持して続けてください。
```

## 2026-10-07に確立した運用

Claude Code担当の7spec（resume.mdの現在地を参照）で使った進め方。次の主担当も、変更する理由がなければ同じ形で続ける。

### specの進め方

- 1つのspecは、旧の1メソッドまたはその一部に対応する小さな単位にする。文書はREADME・requirements・design・naming・tasks・research・review・integration-validation・spec.json。
- 要求・設計・命名・tasksは、1回の依頼でまとめてレビューに出してよい。判定は段階ごとに別々に受け、spec.jsonへ段階ごとのrevision・LF hash・レビューsessionを記録する。指摘で改訂した段階だけ再レビューする。
- 実装taskは、(1)組立と依存境界、(2)上流・下流の部品との接続と学習の継続、(3)固定環境の全回帰、の順。feature最終GOは、task承認とは別のレビューsessionで受ける。
- 失敗時に部分的な変更を残さないため、検証と数値の生成を状態変更より前に置く設計を採ってきた。旧と処理順が変わる場合は、成功時の観測値（値・順序・乱数の消費）が変わらないことを実旧との対照testで示し、research.mdへ理由を書く。
- 旧の契約外入力での挙動（部分更新など）を見つけたら、実旧で再現してからimplementation-findingsへ記録する。正常経路の非対称（LEGACY-014）はユーザーへ確認する。

### 実装前のREDと検出力の確認

- testを書いたら、実装ファイルを作る前にpytestを実行し、REDを記録する。実装を先に書いてしまうと、レビューで承認ゲート違反として差し戻される（assigned-training-sample-absorptionのreview.md）。
- GREENの後、実装を一時的にstubと数種の誤実装へ差し替えてtestが失敗することを確かめ、元へ戻して実装のLF hashが変わっていないことを確認する。検出されない誤実装があれば、testの条件が足りない。
- 差し替えscriptの例: `../../venv/refactoring-tests/post_alarm_reference_model_fixation_red_evidence.py`（Git管理外）。実装ファイルを書き換えるので、レビュー担当には実行させず内容を読ませる。

### 実旧を使うoracle

- 式や手順をtestへ複製して正解にしない。実旧clientのクラスを`__new__`で作り、対象メソッドが読む属性だけを与えて、実旧メソッドを実行する。
- test helperは上流のtestから再利用する（`tests/refactoring/`内で互いにimportする）。主な連鎖:
  - `test_adopted_candidate_initial_local_registration.py`の`build_initial_registration_oracle`: 同じ初期値の新owner群・候補と、実旧`SharedBackboneClassConditionalESRFedSDAClient`・旧候補。`assert_held_model_states_match_legacy`・`convert_legacy_parameter_name`もここ。
  - `test_adopted_candidate_local_adoption.py`の`build_local_adoption_oracle`: 上に採番・標本・計数ownerと実旧`ForwardValidationSession`を加える。`snapshot_adoption_state`/`assert_adoption_state_unchanged`（拒否時不変）もここ。
  - `test_post_alarm_candidate_validation_resolution.py`の`assert_resolution_matches_legacy`、`test_post_alarm_candidate_validation_sample_observation.py`の`observe_and_resolve_in_both_implementations`、`test_post_alarm_reference_model_fixation.py`の`begin_forward_validation_in_legacy_client`（実旧のsession開始。現在は候補の学習だけ無効化）。
- 実旧で実行できると確認済みのメソッド: `_register_trained_new_model`、`_absorb_into_store`、`_finalize_forward_validation`（4分岐）、`_observe_forward_validation`、`_snapshot_reference_models`、`_begin_forward_validation`（候補の学習を無効化した場合）、`_train_heads_together`、`confirm_model_registration`。新しいメソッドを使うときは、REDより前に実行できることを確かめてresearch.mdへ書く。
- 学習の継続は、class2/4×Adam標準/AMSGrad/SGD×共有部更新有無の12条件で、実旧の共同学習と全値・grad・optimizer state・乱数を照合する形を続けている。

### 乱数の消費

- モデルの生成は初期化でCPUのtorch乱数を消費する。新`ResidualAdapterClassifier`の生成は実旧と同じ順・量で消費する。実旧の参照複製と候補の生成も乱数を消費するので、生成の順序と個数を旧と一致させる（post-alarm-reference-model-fixation）。
- testで乱数の不変を確かめるときは、モデルを生成した後の状態を基準にする。同じ乱数状態から実旧と新を順に実行し、処理後の状態が一致することと、実際に消費されたことの両方を確かめる。

### 検証コマンド

worktreeルートで実行する。下はGit Bash用。PowerShellの環境変数は続けて示す。

実行する時点: (1)testを書いた直後、実装ファイルを作る前に対象testだけを実行してREDを記録する（import失敗でよい）。(2)実装後に対象testを実行する。依存境界testは、新module用の注入契約testを追加した時点で一度REDを確認し、exact guardを追加してから対象testと一緒に実行する。(3)全taskの実装をcommitした後に全pytestと品質検査を実行し、そのcommitを検証対象として記録する。仕様化の段階（要求〜tasks）ではtestを実行しない。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMP="$(cd ../../venv/refactoring-tests && pwd -W)"; export TEMP="$TMP"
export MPLCONFIGDIR="$(cd ../../venv/matplotlib-cache && pwd -W)"
export FDE_MNIST_DATA_DIR="$(cd ../../data/mnist && pwd -W)"
PY=../../venv/Scripts/python.exe

$PY -m pytest tests/refactoring/test_<対象>.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
$PY -m pytest tests -q -p no:cacheprovider --junitxml="$TMP/<feature名>-full.xml"   # 約2分
$PY -m ruff check src tests/refactoring; $PY -m ruff format --check src tests/refactoring
$PY -m pyright --pythonpath ../../venv/Scripts/python.exe; $PY -m pip check
git status --short   # 検証対象のcommit時点で未コミット差分がないこと
git diff --stat 748c3aa HEAD -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json tests/test_regression.py tests/test_proposed_regression.py tools   # commit済みの差分。空であること
git diff --stat 748c3aa -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json tests/test_regression.py tests/test_proposed_regression.py tools   # 作業中の未コミット差分も含む。空であること
```

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
$env:TMP=(Resolve-Path ../../venv/refactoring-tests).Path; $env:TEMP=$env:TMP
$env:MPLCONFIGDIR=(Resolve-Path ../../venv/matplotlib-cache).Path
$env:FDE_MNIST_DATA_DIR=(Resolve-Path ../../data/mnist).Path
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --junitxml="$env:TMP/<feature名>-full.xml"
```

PowerShellの例は環境変数の設定と全pytestだけを示している。対象test・Ruff・Pyright・pip check・git diffは、Git Bash用の各行の`$PY`を`../../venv/Scripts/python.exe`へ読み替えればPowerShellでもそのまま実行できる。

承認hash（各mdのLF sha256）とsource hash（tracked Python＋2golden）は次で計算する。`lf_sha256`は作業ツリーのファイルを読む（レビューへ出す直前と承認の記録時に計算する）。`source_sha256`は指定したcommitに含まれるファイルだけを読み、未コミットの変更は含まない。全回帰を実行したcommit（検証対象commit）に対して計算し、spec.jsonへ記録する。この手順は前specまでの記録値を再現する。確認用の期待値: `source_sha256("03f24e6")`は`(227, "e87d426ec169c676a7b6ce57b7aa468d8d434992d08bdb5dcf6ffb12a8370621")`、`source_sha256("dbaf5cc")`は`(241, "086a886ed484a41844fcd14ddacea8f29a73205efdaef1d92453edd98f944f4c")`。

```python
import hashlib, pathlib, subprocess
def lf_sha256(path):  # spec.jsonのgenerated/approved/current_sha256_lf
    return hashlib.sha256(pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
def source_sha256(rev="HEAD"):  # spec.jsonのverification_source_sha256とpath数
    names = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", rev], capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    names = sorted(n for n in names if n.endswith(".py") or n in ("tests/regression_golden.json", "tests/proposed_regression_golden.json"))
    digest = hashlib.sha256()
    for name in names:
        content = subprocess.run(["git", "show", rev + ":" + name], capture_output=True, check=True).stdout
        digest.update(name.encode("utf-8") + b"\0" + content.replace(b"\r\n", b"\n") + b"\0")
    return len(names), digest.hexdigest()
```

全pytestの件数は、前specの件数に今回追加したtest数（対象test＋依存境界の注入契約）を足した数と一致することを確かめてきた（直近は6355）。

### Git管理外の資材

- `../../venv/refactoring-tests/`（元checkoutのvenv配下、このPCだけ）: 各specのfresh CPU smoke（`*_cpu_smoke.py`、旧importなし）、実装差し替えscript（`*_red_evidence.py`）、全pytestのJUnit（`*-full.xml`）。別PCでは存在しないので、必要なら対象specのintegration-validation.mdの記述から作り直す。
- Claude Codeのセッション内だけにあった補助script（hash記録、レビュー起動）はリポジトリにない。上の検証コマンドとhash計算で同じことができる。

### レビュー依頼の注意

- Claude Codeからは、別プロセスの`codex exec -m gpt-6-luna --sandbox read-only`（testを実行させるときは`workspace-write`）へ依頼文を標準入力で渡し、最終回答を保存した。Codexが主担当のときは、Codex側の独立レビュー手段を使う。どちらでも、依頼文・session・対象hash・結果・採否を対象specへ記録する。
- 依頼文には次を書く。(1)worktreeの絶対パスと期待するブランチ・HEAD（元checkoutで実行して対象を取り違えた例がある）。(2)対象ファイルとLF hash、主担当の実測、手順上の逸脱があればその事実。(3)全pytestの判定基準の原文（基準を逆に読んで未再現をBlockerにした例がある）。(4)「自分のモデル名を確認できないことを判定理由にしない」こと（それを理由に保留・NO-GOにした例がある）。(5)出力形式（1行目に判定、番号付き指摘、独立実行した検証）。
- レビュー担当にtestを実行させた後は、`git status --short`が依頼前と同じであることを確かめる。レビュー担当が作った一時ディレクトリが残ることがある。
- 指摘は全て採用する必要はない。前提が事実と違う指摘は、根拠を示して不採用とし、review.mdへ理由を書く（例: 記載済みの要求番号を「ない」とした指摘、基準の誤読、元checkoutの取り違え）。NO-GOの理由が実装・testにないときも、GOへ書き換えず、事実を示して再判定を依頼する。

## cc-sdd導入の来歴と再生成

2026-10-07、既存Codex Skills版と同じcc-sdd 3.1.0のClaude Skills版を導入した。
両方17 skills。ツール固有のskillと補助テンプレートは各ツールのディレクトリに置き、
プロジェクトの仕様・承認・方針は共通の`.kiro/`とdocsで管理する。

別の作業用ディレクトリへ`npx cc-sdd@3.1.0 --claude-skills --lang ja --overwrite force`で生成し、
生成された`.claude/skills/`だけを本worktreeへ取り込んだ。
生成元のCLAUDE.mdと`.kiro/settings/`は取り込まず、既存spec・承認・カスタム規約を保持した。
更新時も隔離した場所で生成してdiffを確認する。本worktreeへforce上書きで再導入しない。

Claudeのimport構文は[公式memory文書](https://code.claude.com/docs/en/memory#import-additional-files)、
cc-sddの各エージェント対応は[公式リポジトリ](https://github.com/gotalab/cc-sdd)を参照する。
Claude実行時に`/context`で入口とimportの読み込みを確認する。ファイル導入だけで実行・読み込み確認済みとは扱わない。

### 導入確認記録（2026-10-07）

- 17 skills・33ファイルを生成元とバイト単位で照合し、新しい相対リンク・CLAUDE.mdのimport先を確認した。
- 既存spec・旧新コード・テスト・Codex skillsは変更なし。コード変更がないため数値回帰は再実行していない。
- GPT-6 Luna独立レビューで、共通規約の旧一律Luna指定を担当ツール別へ統一する指摘を採用し、再レビューPASS。
  自動承認オプションを代替にしない旨も明記した。Claudeの起動・Sonnet委任の実行確認は未実施。
  その後の2026-10-07ユーザー訂正でLuna優先・利用不能時Sonnet代替とし、2026-10-08に上記のLuna/Haiku 5.5の選択・相互代替へ変更した。この導入時レビューは来歴として保持する。
