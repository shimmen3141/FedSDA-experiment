# 候補検証の確定に伴う診断通知 — 統合検証

## 対象と判定

検証対象commit: `ef1be82`（本specのsource・testの最終commit。全pytestを実行したcommit。sourceの最終変更は`42fc7b8`で、`ef1be82`はtestだけを追加した）。主担当Claude Code、2026-10-09。実行環境はWindowsの基準環境（固定venv、Python 3.13、torch 2.12.1+cpu、OMP/MKL各1 thread）。判定は末尾の「レビュー」に記録する。

## taskごとの証拠

| Task | 証拠 |
| --- | --- |
| 1 実装 | 実装前RED（変更後のtestは変更前のsourceで12 failed、注入契約testはguardなしで4 failed）。独立レビュー1回目の指摘（保持がないときのowner型拒否のtestがない）でtestを10条件足した。GREENは対象57＋依存境界3043＝3100 passed。経緯はreview.md |
| 2 検出力と独立動作 | 実source変異41/41検出（本specで足した11種＋既存30種の再実行。検査を上流の後へ移す変異3種を含む）、復元後57 passed。fresh CPU 8条件成功。[詳細](mutation-and-cpu-evidence.md) |
| 3 全回帰・品質 | 全pytest 10029 passed / 3 skipped / 2 warnings、exit 0。Ruff・Pyright・pip check成功、固定旧差分は空 |

## 要求対応（7/7）

| 要求 | 実装と検証 |
| --- | --- |
| 1.1 | `advance_held_candidate_validation`が、確定のとき記録→保持の解除の後に`notify_diagnostics_of_training_assignment_change`を1回呼ぶ。2/4class×確定4条件で、通知が1回、渡る帰属変更が確定の結果のものそのもの、通知の時点で記録が追加済み・保持が空。通知の省略・二重実行・解除や記録の前への移動・帰属変更を渡さない・別の診断証拠へ通知する変異を検出 |
| 1.2 | 未到達3標本で診断証拠が不変。保持なしで診断証拠が不変。未到達かどうかを見る前に通知する変異を検出 |
| 1.3 | `finalize_held_incomplete_candidate_validation`は変更なし（引数に診断のownerがない）。終端回収のtestは変更なしで成功 |
| 2.1 | 進行×（保持、記録、診断）と終端回収×（保持、記録）の5組×（別の型、派生型）×（保持あり、保持なし）の20条件で、上流を呼ぶ前に拒否（上流を呼んだら失敗するmock）、例外の文言で対象の引数を確認。診断の型検査を上流の後へ移す・削除・isinstanceへ緩める・保持があるときだけ行う変異を検出 |
| 2.2 | 拒否の文言を「only a response during candidate validation is accepted while a session is held」へ改めた。条件と例外の型は同じで、既存の拒否12条件が保持の不変のまま成功。文言を照合するtestは足していない |
| 3.1 | 2/4class×確定4条件で、診断証拠が実旧の再始動hookの後のAdaHedgeと一致。再始動は採用と他モデルの再利用で1回、現行の維持と棄却で0回 |
| 3.2 | 既存の照合（進行の結果、適応記録、保持、記録→解除の順）は変更なしで成功。既存の変異30種も変更後のsourceで全件検出。全pytest（旧11・最終3goldenを含む）、固定旧差分が空 |

## 同一性と品質

| 対象 | LF SHA256 |
| --- | --- |
| 要求r1 | 940af2b81a5eab2d72ff326c61f43d686f80e38471e6ff12f46f25045c239b48 |
| 設計r2（r1から、8節のownerの型の拒否の条件数と変異の種類を改めた。契約と処理は同じ） | 7b90f6c40ceffc47a6e0a3a6fc70f094fccf6deeb1e05301550e40e3b67656b9 |
| 命名r2（r1へ`session_is_held`を追加） | ed7adaedc0124957898f758ce515ecbbb8b83b9e27132e5e20608f7887a44d01 |
| tasks r2（r1から条件数と変異の種類を改めた。承認hashの規約どおり、tasks.mdの完了のcheckbox `[x]`を`[ ]`へ置き換えた内容で計算した値で、現在のファイルそのもののhashではない） | 45cefee7594285075fd1332577d8b0c69e798f218b16ce9c69c0118aed90e2e8 |
| source全体、`ef1be82`、283パス | 99540715ccb940c52f960a286eb0339996035b38c2207c4e7b3e80e9bc43bf4e |

source hashはtracked Pythonと2goldenの、パス昇順・LF内容のhash（`spec_checks.py identity --rev ef1be82`の計算）。直前の完了spec（`e80b368`）からの`src`・`tests`の差分は、本specの3ファイル（進行の接続、そのtest、依存境界test）だけ。固定旧`748c3aa`から旧実装・tools・2golden・旧回帰testへのdiffは、commit済み・作業ツリーとも空。全pytestの実行時、`src`・`tests`に未コミット差分はない。

全pytestは**10029 passed / 3 skipped / 2 warnings、186.34s、exit 0**。直前の完了spec 10000＋注入契約13＋対象の増分16（41→57）＝10029。JUnitは10032 testcase、failure 0、error 0、skip 3。`tests.test_regression`と`tests.test_proposed_regression`はどちらも成功。skip 3件と警告2件は以前のspecと同じ既存のもの。

Ruff check成功、format checkは182 files already formatted。Pyrightは固定venv指定で0 errors/0 warnings/0 informations。pip check成功。

参考: 指摘の反映より前のcommit `42fc7b8`でも全pytestを実行した（10019 passed / 3 skipped。`venv/refactoring-tests/held-candidate-validation-diagnostic-notification-full-42fc7b8.xml`）。この実行の終わり近くでtestファイルを編集し始めたため、判定には使わない。

## 証拠の場所

Git管理外（元checkoutの`venv/refactoring-tests/`、このPCだけ）: `held-candidate-validation-diagnostic-notification-full.xml`/`.log`、`held-candidate-validation-diagnostic-notification-task1-red-target.log`・`-task1-red-guard.log`・`-task1-green.log`、`held-candidate-validation-diagnostic-notification-mutations.py`と`held-candidate-validation-diagnostic-notification-mutation-evidence/`、`held-candidate-validation-diagnostic-notification-fresh-cpu.py`と`.log`、各レビューの出力。

## 未検証・残る制約

- 警報のない標本での帰属確定と学習、標本1件の処理全体、予測側の警報hook、検出位置の記録、サーバ同期、新全体runのgolden一致（保存する診断の同一性を含む）は未検証。
- 進行の関数の引数が増えたので、過去spec（held-candidate-validation-progress、alarm-occurrence-handling）のfresh CPU script（Git管理外）は現在のsourceでは動かない。当時のcommitの証拠として残し、更新していない。同じ流れは本specのfresh CPU scriptが現在のsourceで実行している。
- 独立レビュー担当による全pytestの再実行は行っていない（2026-10-07のユーザー決定による基準）。Task 1のレビュー担当（Haiku、読取り専用）はtestを実行していない。
- 手順上の事実: 命名の事前登録のためsourceとtestの変更をリポジトリ外で下書きし、承認前に作業ツリーの複製で実行した（変異scriptの試行を含む）。worktreeへはtest→RED→注入契約test→RED→guard→srcの順で適用した。変異とfresh CPU（Task 2の内容）は、Task 1の独立レビューより前に実行し、指摘の反映の後に全件を実行し直した。

## レビュー

Task 1はClaude Haiku 5.5（2回目で承認）、設計・命名・tasksのrevision2とTask 2・3はGPT-6 Lunaが承認（Task 3は2回目で承認。2026-10-09）。別sessionのLunaがfeature最終GO（同日）。経緯と採否は[review.md](review.md)。
