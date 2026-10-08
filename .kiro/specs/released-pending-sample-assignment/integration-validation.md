# 警報のない標本での帰属確定 — 統合検証

## 対象と判定

検証対象commit: `cbf38b6`（本specのsource・testの最終commit。全pytestを実行したcommit。sourceの最終変更は`86cd7ed`で、`cbf38b6`はtestだけを追加した）。主担当Claude Code、2026-10-09。実行環境はWindowsの基準環境（固定venv、Python 3.13、torch 2.12.1+cpu、OMP/MKL各1 thread）。判定は末尾の「レビュー」に記録する。

## taskごとの証拠

| Task | 証拠 |
| --- | --- |
| 1 実装 | 新test 47、保留位置のownerのtest 69、依存境界、共用scriptを実行するtestを合わせて3160 passed。独立レビュー1回目の指摘でtestを足し、2回目で承認（経緯はreview.md） |
| 2 検出力 | 汎用の変異toolで、確定の関数は21/22（未検出1種は等価）、保留位置のownerの読取りの操作は3/3。[詳細](mutation-and-cpu-evidence.md) |
| 3 全回帰・品質 | 全pytest 10080 passed / 3 skipped / 2 warnings、exit 0。Ruff・Pyright・pip check成功、固定旧差分は空 |

## 要求対応（9/9）

| 要求 | 実装と検証 |
| --- | --- |
| 1.1 | `assign_released_pending_samples_to_current_training_model`。2/4 class×6条件（保留が容量未満・ちょうど・1件超過・複数件超過）で、学習標本・割当概念計数・損失統計を実旧の標本処理の後と照合。戻り値は渡した最古の標本そのもの。吸収は1回で、その時点で保留は未解放、渡る標本・概念ID・モデルID・ownerが正しい |
| 1.2 | 解放がない条件で全状態が不変、空のtupleを返す。解放後の再実行でも不変。解放がないとき吸収を呼ばない |
| 1.3 | 現在の学習帰属が実旧と一致して不変。モデル・optimizer・乱数が不変 |
| 1.4 | `PendingTrainingAssignmentBuffer.get_sample_indices_exceeding_capacity`。読取りが状態を変えず、直後の解放と同じ位置を同じ順で返す。変異3種（1件多く返す、新しい側から返す、容量を使わない）を検出 |
| 2.1 | ownerの別の型・派生型、保留標本の組のlist・tupleの派生型、要素の別の型・派生型、位置のintの派生型、並びの不一致（新しい側の不足・古い側の不足・逆順・位置の違い）で、吸収が呼ばれず全状態が不変。型の検査が並びの検査より先に拒否する |
| 2.2 | 解放対象の標本の特徴数が違う・概念IDがintでない・現在の学習帰属のモデルが未保有・吸収へ渡すだけのownerが別の型、の各条件で、保留を含む全状態が不変 |
| 2.3 | 解放しない保留標本の標本の中身が不正でも成功する。型と位置は全要素を検査する（設計4節） |
| 3.1 | 上の1.1の実旧対照（実旧`FedSDAClient.process_one_step`を、予測・候補検証の観測・検出・学習・計算量の記録を止めて実行）。保留に残る位置が実旧のFIFOに残る標本と一致 |
| 3.2 | 全pytest（旧11・最終3goldenを含む）、固定旧差分が空 |

## 同一性と品質

| 対象 | LF SHA256 |
| --- | --- |
| 要求r1 | 0d29d97db3efd5386f37918d9655123621877f4dae4713f1894140922f02f022 |
| 設計r2（r1から、解放がないとき吸収を呼ばない形へ変更） | b46252e93c4cbc70c525e694188fa75b9af665513f6a7202fe40c976c5edd61a |
| 命名r2（r1へtest名を2つ追加） | cdd6757fb4054720f96a7c7b5ec208114106a3278a7c47739fcd0bf19bfa3cd2 |
| tasks r1（承認hashの規約どおり、完了のcheckbox `[x]`を`[ ]`へ置き換えた内容で計算した値） | 6358806a7ef5e5df7fefd3b2ea2d996dc771c07c0fcf9624c4df3c2179306a7a |
| source全体、`cbf38b6`、288パス | efd414bc52a424e84aff2849fa8e9ee16891cbd4e0892b4c39785fe2b7d02040 |

source hashはtracked Pythonと2goldenの、パス昇順・LF内容のhash（`spec_checks.py identity --rev cbf38b6`の計算）。前spec（`97d9c43`、286パス）から2パス増えた（新しいsourceとtest）。`src`・`tests`の差分は、新しい2ファイルと、保留位置のowner・そのtest・依存境界test・共用scriptの4ファイル。固定旧`748c3aa`から旧実装・tools・2golden・旧回帰testへのdiffは、commit済み・作業ツリーとも空。全pytestの実行時、未コミット差分はない。

全pytestは**10080 passed / 3 skipped / 2 warnings、192.42s、exit 0**。前spec 10031＋新test 47＋保留位置のownerのtest 2＝10080。JUnitは10083 testcase、failure 0、error 0、skip 3。`tests.test_regression`と`tests.test_proposed_regression`はどちらも成功。skip 3件と警告2件は以前のspecと同じ既存のもの。

Ruff check成功、format checkは186 files already formatted。Pyrightは固定venv指定で0 errors/0 warnings/0 informations。pip check成功。

参考: 指摘の反映より前のcommit `86cd7ed`でも全pytestを実行した（10077 passed / 3 skipped。`venv/refactoring-tests/released-pending-sample-assignment-full-86cd7ed.xml`）。判定には使わない。

## 証拠の場所

Git管理外（元checkoutの`venv/refactoring-tests/`、このPCだけ）: `released-pending-sample-assignment-full.xml`/`.log`、`released-pending-sample-assignment-mutation-evidence/assignment/`と`buffer/`（`report.json`と変異ごとのlog）、各レビューの出力。

## 未検証・残る制約

- 学習要求の記録と共同学習の接続（旧`train_step`）、標本1件の処理全体、計算量の診断、保留標本そのものを持つowner、新全体runのgolden一致は未検証。
- 保留標本の並びの検査は、警報のときの応答のmoduleにも同じ趣旨のものがある（重複。IMPROVE-010）。
- 独立レビュー担当による全pytestの再実行は行っていない（2026-10-07のユーザー決定による基準）。Task 1のレビュー担当（Haiku、読取り専用）はtestを実行していない。
- 手順上の事実: sourceとtestをリポジトリ外で下書きし、承認前に作業ツリーの複製で実行した（変異toolの試行を含む）。worktreeへはtest→sourceの順で適用した。変異toolはTask 1の独立レビューより前に実行し、指摘の反映の後に実行し直した。設計r2で足したtest名と、Task 1の指摘で足したtest名は、命名r2として実装の後に再レビューを受けた（命名r1の承認時にはなかった名前）。

## レビュー

Task 1はClaude Haiku 5.5（2回目で承認）、命名r2とTask 2・3は、GPT-6 Lunaが利用上限で使えないためClaude Haiku 5.5が代替して承認し、別sessionのHaiku 5.5（同じく代替）がfeature最終GO（2026-10-09）。経緯と採否は[review.md](review.md)。
