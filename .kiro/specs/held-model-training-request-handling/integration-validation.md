# 学習要求の記録と保有モデルの共同学習 — 統合検証

## 対象と判定

検証対象commit: `1a03667`（本specのsource・testの唯一のcommit。全pytestを実行したcommit）。主担当Claude Code、2026-10-09。実行環境はWindowsの基準環境（固定venv、Python 3.13、torch 2.12.1+cpu、OMP/MKL各1 thread）。判定は末尾の「レビュー」に記録する。

## taskごとの証拠

| Task | 証拠 |
| --- | --- |
| 1 実装 | 新test 101、件数管理のtest 74、依存境界、共用scriptを実行するtestを合わせて3219 passed（主担当の実測と、独立レビュー担当の独立実行で同じ結果）。独立レビュー承認（経緯はreview.md） |
| 2 検出力 | 汎用の変異toolで、学習要求の処理の関数は31/35（未検出4種は等価）、件数管理の読取りの操作は3/3。[詳細](mutation-and-cpu-evidence.md) |
| 3 全回帰・品質 | 全pytest 10185 passed / 3 skipped / 2 warnings、exit 0。Ruff・Pyright・pip check成功、固定旧差分は空 |

## 要求対応（13/13）

| 要求 | 実装と検証 |
| --- | --- |
| 1.1 | `record_training_request_and_train_held_models_when_due`。間隔に達していない要求では、保留件数が実旧の`train_step`の後と一致し、共同学習の反復が呼ばれず、空のtupleを返す |
| 1.2 | 同じ関数。2値・多クラス、更新間隔3種、一要求あたりの回数3種、batchの件数4種で、要求と消化の列の各段の後に、保留件数・乱数の状態・計数・全モデルのparameterとgradとoptimizerの状態を実旧と照合。戻り値は反復が返した損失を実行順に並べたもの。Adamの別種とSGDでも対照 |
| 1.3 | `train_held_models_for_pending_training_requests`。端数の消化と空の消化を列に含めて実旧の`flush_pending_updates`と照合。保留が0件なら、学習へ渡すだけの値が不正でも成功し、何も変わらない |
| 1.4 | 回数0の条件を上の対照に含む（学習せず、間隔に達すれば保留が0件になる）。回数0なら、学習へ渡すだけの値が不正でも成功し、保留だけが消化される |
| 1.5 | `LocalTrainingRequestSchedule.has_pending_requests_reaching_update_interval`。読取りが状態を変えず、回数が0でも件数だけで決まる。変異3種（「より多い」、回数と比べる、常に真）を検出 |
| 2.1 | 共同学習の反復が共同更新1回ぶんずつ呼ばれ、各回の時点で、計数がそれまでに完了した回のぶんだけ増えている。参加モデルの計数（標本数＝batchの件数、更新回数＝1）は実旧の`model_training_examples`・`model_optimizer_steps`と項目の順を含めて一致。保有していないモデルの標本は学習にも計数にも入らない |
| 2.2 | 参加するモデルがない条件（batchの件数6）の実旧との対照で、計数が変わらず項目も増えない。反復を「損失を返さない」ものへ差し替えた条件で、標本の足りる保有モデルがあっても計数しない（後者は実旧をoracleにしていない。設計8節） |
| 2.3 | 計数への反映のどの時点でも保留が未消化。割当概念の計数が不変。標本・損失統計は既存helperの対照に含まれる |
| 3.1 | owner 4つそれぞれの別の型・派生型を2つの関数へ渡して、TypeError、反復が呼ばれず、保留件数を含む全状態が不変 |
| 3.2 | 2回目の共同更新を抽出の後・更新の前に失敗させ（実旧は2回目の抽出の直後に失敗させる）、完了した1回ぶんのparameter・optimizer・計数と、保留件数が実旧と一致。batchの件数が不正な条件で、保留件数が実旧と同じ（記録済みの要求は残る）、計数・モデル・乱数が不変 |
| 3.3 | 学習へ進まない3つの場合（間隔に達していない、保留が0件、回数が0）で、学習へ渡すだけの値を読まない。参加するモデルがない回の共同更新の入力は、反復の契約どおり検査されない（本specのtestでは確かめていない。反復のspecの範囲） |
| 4.1 | 上の1.2・2.1・3.2の実旧対照（実物の`train_step`・`flush_pending_updates`・`train_all_held_models`・`_train_heads_together`・`_sample_training_batches`） |
| 4.2 | 全pytest（旧11・最終3goldenを含む）、固定旧差分が空 |

## 同一性と品質

| 対象 | LF SHA256 |
| --- | --- |
| 要求r3 | 55afd5245e78363ed3931dbdc00320dd1a71e1f76c08d37d363654ad9a8734a8 |
| 設計r5 | 4681c58f475ba888fcee30e3038d43865f45f82fead15f97ea56d9f6f2fca0d3 |
| 命名r2 | 0e56d3bdf0c7fcf0ed6a1e9f4afb2eb0edd7977f60a74db3b5f9abef26bcaa59 |
| tasks r2（承認hashの規約どおり、完了のcheckbox `[x]`を`[ ]`へ置き換えた内容で計算した値） | cf59217419a851655c31b0b5cadb238705dd04ab6a2894e9e87b2a38c807ebba |
| source全体、`1a03667`、290パス | 0a03123dbebba4736bc29ad3fa3265616c4c7eb99844c565405ae513aee91401 |

source hashはtracked Pythonと2goldenの、パス昇順・LF内容のhash（`spec_checks.py identity --rev 1a03667`の計算）。前spec（`cbf38b6`、288パス）から2パス増えた（新しいsourceとtest）。`src`・`tests`の差分は、新しい2ファイルと、件数管理・そのtest・依存境界test・共用scriptの4ファイル。固定旧`748c3aa`から旧実装・tools・2golden・旧回帰testへのdiffは、commit済み・作業ツリーとも空。

全pytestは**10185 passed / 3 skipped / 2 warnings、342.84s、exit 0**。前spec 10080＋新test 101＋件数管理のtest 4＝10185。JUnitは10188 testcase、failure 0、error 0、skip 3。`tests.test_regression`と`tests.test_proposed_regression`はどちらも成功。skip 3件と警告2件は以前のspecと同じ既存のもの。所要時間は前spec（約190秒）より長い（原因は調べていない。件数と結果は内訳どおり）。

全pytestの実行中の作業ツリー: 開始時は`1a03667`で未コミット差分なし。実行中に、検出力の証拠の文書（`.kiro/specs/`の下のmd）を1つ追加した。source・test・設定は変更していない。

Ruff check成功、format checkは188 files already formatted。Pyrightは固定venv指定で0 errors/0 warnings/0 informations。pip check成功。

## 証拠の場所

Git管理外（元checkoutの`venv/refactoring-tests/`、このPCだけ）: `held-model-training-request-handling-full.xml`/`.log`、`held-model-training-request-handling-mutation-evidence/handling/`と`schedule/`（`report.json`と変異ごとのlog）。各レビューの依頼文と応答は主担当のsessionの一時領域（保存は保証されない）。

## 未検証・残る制約

- 2つの関数を呼ぶ位置（標本1件の処理、警報の前、ラウンド境界）、標本1件の処理全体、計算量・時間・勾配の診断、保留標本そのものを持つowner、新全体runのgolden一致は未検証。
- 旧との差が1つ残る: 1回の共同更新の途中（個別部のoptimizerを順に進めている間と、旧ではその後の診断の記録の間）で失敗した場合、旧はそこまでに進めたモデルの計数が残り、新ではその回の計数は残らない（設計2節）。旧も新も例外は標本処理の外まで伝わり、失敗した試行の結果は返らない。ユーザーの確認項目として再開案内に残す。
- 参加モデルの条件が、batchの抽出と計数の2箇所にある（IMPROVE-011）。
- 独立レビュー担当による全pytestの再実行は行っていない（2026-10-07のユーザー決定による基準）。
- 規則からの逸脱: 命名の承認より前に、sourceとtestをリポジトリ外で下書きし、作業ツリーの複製で実行した（変異toolの試行を含む）。仕様レビューで指摘され、research.mdとreview.mdに記録した。複製での結果は判定に使っていない。worktreeへは、仕様の承認の後にtest→source→依存の登録の順で適用した。今後の扱いはユーザーへ確認する。
- 変異toolはTask 1・2の独立レビューより前に実行した（レビューでの変更はなく、実行し直していない）。

## レビュー

仕様（5回）、Task 1・2は、GPT-6 Lunaが利用上限で使えないためLunaの担当分もClaude Haiku 5.5が行った。経緯と採否は[review.md](review.md)。Task 3とfeature最終の判定は、この文書の作成時点では未記録（記録後にこの節を更新する）。
