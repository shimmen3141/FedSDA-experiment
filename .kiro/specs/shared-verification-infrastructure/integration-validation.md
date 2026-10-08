# 検証基盤の整理 — 統合検証

## 対象と判定

検証対象commit: `97d9c43`（本specのtestの最終commit。全pytestを実行したcommit）。`src/`の変更はない。主担当Claude Code、2026-10-09。実行環境はWindowsの基準環境（固定venv、Python 3.13、torch 2.12.1+cpu、OMP/MKL各1 thread）。判定は末尾の「レビュー」に記録する。

## taskごとの証拠

| Task | 証拠 |
| --- | --- |
| 1 追加 | 新test・保持と進行のtest・依存境界testで3102 passed。Lunaが承認（経緯はreview.md） |
| 2 検出力 | 追加した検証を一時的に壊す確認6/6、汎用の変異toolで29/31（未検出2種は等価。修正前は26/31）。[詳細](mutation-and-cpu-evidence.md) |
| 3 全回帰・品質 | 全pytest 10031 passed / 3 skipped / 2 warnings、exit 0。Ruff・Pyright・pip check成功、`src/`と固定旧の差分は空 |

## 要求対応（9/9）

| 要求 | 実装と検証 |
| --- | --- |
| 1.1 | `tests/refactoring/fresh_process_smoke.py`。2/4 class×8つの流れ（不足、警報での再利用、警報での維持、候補検証中の警報、採用・棄却・維持・検証後の再利用とその後の次の警報）を、1組のownerで実行。各警報の後に戻り値と記録・保持・監視・保留位置・診断の整合を、確定の後に記録・保持・診断の再始動を確かめる |
| 1.2 | `test_fresh_process_smoke.py`が`sys.executable`で別processとして実行（`PYTHONPATH`を外す）。旧実装のimportを足す・流れを1つ外す・sourceの引数名を変える、のそれぞれで失敗することを確認 |
| 1.3 | Git管理下。worktreeルートから`python tests/refactoring/fresh_process_smoke.py`で単独実行して成功（レビュー担当も実行） |
| 2.1 | `test_module_allowed_dependencies_are_all_imported_by_the_module`。使っていない名前を許可へ足すと失敗することを確認（2通り） |
| 2.2 | 読めたmoduleの数の下限のassert。読取りが空振りする状態にすると失敗することを確認 |
| 2.3 | `candidate_parameter_initialization.py`の許可から`torch.Tensor`を外し、許可例のtestから`from torch import Tensor`を除いた |
| 3.1 | 応答の派生型の拒否条件と、保持のownerの派生型の拒否を追加。変異toolで、exact型検査をisinstanceへ緩める2種を検出 |
| 3.2 | 終端回収のtestへ、解除の時点で記録が追加済みであることの確認を追加。変異toolで、記録と解除の入替えを検出 |
| 4.1 | `git diff ef1be82 97d9c43 -- src`が空（変更は`tests/refactoring/`の4ファイル）。固定旧`748c3aa`から旧実装・tools・2golden・旧回帰testへのdiffが空。全pytest |

## 同一性と品質

| 対象 | LF SHA256 |
| --- | --- |
| 要求r1 | add420d43e2e7aaec471a918357221f81ad4266ce23d7c6463cb8b0f626d9ae9 |
| 設計r1 | 375754b39780beba09cf57e77c8bb4eb0e8f7561d44b6682ec9b4e68f73fc9c5 |
| 命名r1 | 7b8e4b91b37cd9705876cf7975ff1ad02451096504b4cf8bdd5b71796267ac28 |
| tasks r1（承認hashの規約どおり、完了のcheckbox `[x]`を`[ ]`へ置き換えた内容で計算した値） | 2c3a63de447bea87d3346388b922aa0d5d74a01b13062b9d79e52ca21ac841ce |
| source全体、`97d9c43`、286パス | 669709a0437c8882596f87654f306368527700af6c77b1be3729f886aac533c8 |

source hashはtracked Pythonと2goldenの、パス昇順・LF内容のhash（`spec_checks.py identity --rev 97d9c43`の計算）。前spec（`ef1be82`、283パス）から3パス増えた: 汎用の変異tool（`.kiro/settings/scripts/mutation_check.py`、`caa5b8d`で追加）と、本specの新しい2ファイル。

全pytestは**10031 passed / 3 skipped / 2 warnings、196.88s、exit 0**。前spec 10029＋新test 2（共用scriptの実行、許可集合の検査）＋応答の派生型の拒否条件1−除いた許可例1＝10031。JUnitは10034 testcase、failure 0、error 0、skip 3。`tests.test_regression`と`tests.test_proposed_regression`はどちらも成功。skip 3件と警告2件は以前のspecと同じ既存のもの。全pytestの実行時、tracked fileに未コミット差分はない（未コミットだったのは本specの証拠文書だけ）。

Ruff check成功、format checkは184 files already formatted。Pyrightは固定venv指定で0 errors/0 warnings/0 informations（Pyrightの対象は`src/`で、testとtest用scriptは対象外）。pip check成功。

## 証拠の場所

Git管理外（元checkoutの`venv/refactoring-tests/`、このPCだけ）: `shared-verification-infrastructure-full.xml`/`.log`、`shared-verification-infrastructure-detection.py`と`shared-verification-infrastructure-detection-evidence/`、`shared-verification-infrastructure-mutation-evidence/`、各レビューの出力。

## 未検証・残る制約

- 共用scriptは、新実装が旧実装なしで動くことと、ownerの状態の整合だけを確かめる。実旧との一致は各部品の対照testの役割で、新全体runのgolden一致は未検証。
- 許可集合の検査の対象は、`dependency_is_allowed`の中で名前の組として書かれた分岐（現在58 module）だけ。層ごとの規則や条件つきの許可は対象外。依存testの方式の一本化と既存の注入契約testの整理は行っていない（見送り。IMPROVE-009）。
- `candidate_parameter_initialization.py`に`torch.Tensor`の許可を足した当時のspecの意図は調べていない（sourceが使っていないことだけを確認した）。
- 共用scriptを実行するtestは全pytestに約20秒を加える。
- 独立レビュー担当による全pytestの再実行は行っていない（2026-10-07のユーザー決定による基準）。
- 手順上の事実: 変更をリポジトリ外のpatchとして下書きし、承認前に作業ツリーの複製で実行した（変異toolの試行を含む）。検出力の確認の1件は、確認の間だけ`src/`の1ファイルを書き換えて元byteへ戻した。Task 1・2のレビュー担当のtest実行は、主担当の全pytestと同時だった。

## レビュー

Task 1・2はGPT-6 Lunaが承認（2026-10-09）。Task 3とfeature最終の判定は、同じ依頼で別sessionのLunaが承認・GO（同日）。経緯と採否は[review.md](review.md)。
