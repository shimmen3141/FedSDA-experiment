# 検出力の証拠

Windowsの基準環境（Python 3.13、torch 2.12.1+cpu）、test commit `97d9c43`。証拠ファイルは元checkoutの`venv/refactoring-tests/`（Git管理外、このPCだけ）。

## 追加した検証を一時的に壊す確認

script: `venv/refactoring-tests/shared-verification-infrastructure-detection.py`、証拠: `venv/refactoring-tests/shared-verification-infrastructure-detection-evidence/`（確認ごとのlogと`report.json`）。1件ずつファイルを書き換えて対応するtestを実行し、毎回元byteへ戻した。

結果: 6/6で対応するtestが失敗した。復元後は2 passed。

- 共用scriptへ旧実装のimportを足す → `test_fresh_process_smoke_runs_without_legacy_or_test_modules`が失敗。
- 共用scriptから流れを1つ外す（必要な結果が観測されなくなる） → 同じtestが失敗。
- sourceの進行の関数の引数名を変える（共用scriptが古くなった状態。確認の間だけsrcを書き換えて戻した） → 同じtestが失敗。
- 外した使っていない許可（`torch.Tensor`）を戻す → `test_module_allowed_dependencies_are_all_imported_by_the_module`が失敗。
- 別のmoduleの許可集合へ、使っていない名前を1つ足す → 同じtestが失敗。
- 許可集合の読取りが空振りする状態にする → 同じtestが失敗。

## 汎用の変異tool（NEW-002の修正の確認）

```text
python .kiro/settings/scripts/mutation_check.py --source src/federated_learning_experiments/runtime/held_candidate_validation_progress.py --functions apply_alarm_response_to_validation_session_holder,advance_held_candidate_validation,finalize_held_incomplete_candidate_validation --tests tests/refactoring/test_held_candidate_validation_progress.py --evidence <venv/refactoring-tests/shared-verification-infrastructure-mutation-evidence>
```

結果: 29/31検出。復元後は58 passed。`report.json`は`venv/refactoring-tests/shared-verification-infrastructure-mutation-evidence/`。

修正の前（commit `caa5b8d`の時点のtest）は26/31で、未検出5種のうち3種がtestの穴だった（NEW-002: 保持への反映のexact型検査をisinstanceへ緩める2種、終端回収の記録と解除の入替え1種）。この3種は、今回足した条件で検出されるようになった。

未検出の2種は等価と判断した: `apply_alarm_response_to_validation_session_holder`で、応答の再検査（`__post_init__`）と「結果種別とsessionの対応」の検査を、保持の現在の値を読む代入（`held_validation_session = validation_session_holder.held_validation_session`）の後へ移す変異。その代入は読取りだけで状態を更新しないので、検査が代入の前でも後でも、外から観測できる挙動は同じ。

## 共用のfresh process script

`python tests/refactoring/fresh_process_smoke.py`をworktreeルートで単独実行して成功（2/4 class×8つの流れ。警報の5種類と確定の4種類の結果を観測、旧実装とtest moduleの読込みなし）。全pytestからは`test_fresh_process_smoke.py`が別processで実行する（約20秒）。
