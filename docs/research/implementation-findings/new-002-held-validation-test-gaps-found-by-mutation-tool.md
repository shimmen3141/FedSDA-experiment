# NEW-002: 候補検証sessionの保持と進行のtestに、汎用の変異toolで見つかった穴が2件ある

- 発見日: 2026-10-09。対象: `tests/refactoring/test_held_candidate_validation_progress.py`（検証対象のsourceは`src/federated_learning_experiments/runtime/held_candidate_validation_progress.py`、commit `ef1be82`）。発見の経緯: 汎用の変異tool（`.kiro/settings/scripts/mutation_check.py`）を作って既存のmoduleで試したとき。
- 種別: testの検出力の不足（sourceの挙動は設計どおり）。状態: 未修正。

## 内容

toolは3つの関数から31種の変異を作り、26種が検出、5種が未検出だった。未検出5種のうち2種は等価（下）、3種がtestの穴（下の2件。1件目は変異2種）。

1. **保持への反映で、応答と保持のownerのexact型検査をisinstanceへ緩めても検出されない**（`apply_alarm_response_to_validation_session_holder`の2箇所）。拒否のtestは「別の型」の値だけを渡し、派生型の値を渡す条件がない。進行と終端回収のownerの型は、2026-10-09に派生型の条件を足して検出されるようになったが、反映の関数には足していなかった。
2. **終端回収で、適応記録の追加と保持の解除の順を入れ替えても検出されない**（`finalize_held_incomplete_candidate_validation`）。進行のtestは、解除の時点で記録が追加済みであることを記録用wrapperで確かめているが、終端回収のtestは最終状態（記録が旧イベントと一致、保持が空）だけを見ている。要求（held-candidate-validation-progressの4.1「記録の追加は保持の解除より前に行う」）は終端回収にも当てはまる。

どちらも、手書きの変異（held-candidate-validation-progressの30種、held-candidate-validation-diagnostic-notificationの11種）には入っていなかった。

等価と判断した2種: 反映の関数で、応答の再検査（`__post_init__`）と「結果種別とsessionの対応」の検査を、保持の現在の値を読む代入の後へ移す2種（その代入は読取りだけで、状態を更新しない）。`handle_alarm_occurrence`で、保持への反映と、区間解決を局所名へ取る代入を入れ替える・その後へ移す2種も同じ理由で等価（こちらはalarm-occurrence-handlingの12種のうちの未検出2種）。

## 影響

sourceは正しく、過去の検証結果（実旧との対照、全回帰）は変わらない。将来、この2箇所を誤って変更したときに、testが検出しない。

## 扱い

`test_held_candidate_validation_progress.py`を次に変更するspecで、(1)反映の拒否条件へ応答と保持の派生型を足し、(2)終端回収のtestへ、解除の時点で記録が追加済みであることの確認を足して、toolで検出を確かめる。修正commitは未定。
