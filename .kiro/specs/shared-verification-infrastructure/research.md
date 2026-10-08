# 検証基盤の整理 — 調査記録

2026-10-09、主担当Claude Code。基点commit `caa5b8d`。固定旧基準`748c3aa`。Windowsの基準環境（Python 3.13、torch 2.12.1+cpu）。

## 経緯

2026-10-09、直近4 specの記録から、かけた時間に対して効果が小さい手順を主担当が7件挙げ、ユーザーが採用した（共通引継ぎ手順の経緯の記録）。文書とtoolの変更は`caa5b8d`で反映済み。本specは、testの追加・変更が要る残りを扱う。

## 依存testについての調査と決定

`tests/refactoring/test_single_run_dependency_boundaries.py`を計測した（2026-10-09、読取りだけ）。

- moduleごとに名前の組で許可集合を書いた分岐が58、別形式の分岐が10。test関数は44、条件は3,043、10,616行。
- 検査の方式がmoduleの群によって違う（`from m import a`を`m.a`だけで見る群と、`m`と`m.a`の両方で見る群がある。`__future__`を許可集合に書く群と書かない群がある）。
- 「許可集合と実際のimportが一致する」ことを機械的に確かめる形へ全moduleを移すには、群ごとに方式を読み解く必要があり、1つのscriptでは移せない。得られるのはtestの件数とfileの大きさの削減で、検出できる不具合は増えない。
- ユーザーの決定（2026-10-09）: 既存の登録と注入契約testは移さず、消さない。新しいmoduleにはsymbolごとの注入契約testを足さない。方式の一本化は行わない（[IMPROVE-009](../../../docs/research/improvement-candidates/improve-009-unify-dependency-boundary-tests.md)に、行う場合の条件を記録）。
- 移行なしで入れられる検査として、「許可集合の各名前が実際にimportされている」（許可が広すぎない）を全moduleへ入れる。試算で該当したのは1件（`candidate_parameter_initialization.py`の`torch.Tensor`。sourceは`import torch`だけで、許可例のtestに`from torch import Tensor`がある）。この許可を足したspecの意図は調べていない（使われていないことだけを確認した）。

## fresh process scriptについて

- これまでの個別script（元checkoutの`venv/refactoring-tests/`、Git管理外）は、specごとに約300行を書き、不具合の発見はなかった。held-candidate-validation-diagnostic-notificationで進行の関数の引数が増え、それより前の2本が動かなくなった。
- 共用scriptは、その2本の流れを1つにまとめたもの。下書きを作業ツリーの複製で実行し、2/4 class×8つの流れが成功した。pytestから別processで実行するtestは約20秒かかる（候補の学習を含むため）。

## 手順上の事実

- sourceの変更がないので、命名表はtestのmodule直下の名前だけ（新しい規則）。下書きは、作業ツリーの複製へ適用してWindowsの基準環境のPythonで実行した（worktreeのtestは変更していない）。
- `spec_checks.py names`へ`--base`を足した（既存ファイルの変更で、足した名前だけを照合する）。tool の変更で、実験のsrc・testは変わらない。
