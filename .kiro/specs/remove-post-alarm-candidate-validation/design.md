# Design Document

## Overview
候補検証の処理を、5段に分けて消す。各段の終わりに、候補検証を使わないgolden caseが値を変えずに通ることを確かめる。消す対象と残す対象の境界は、影響範囲の調査の報告
（[scoping-report.md](scoping-report.md)）の1の表に従う。実装担当は、着手の前に報告を読み、HEADとの違いを確かめる。

## 段の分け方（tasks.mdの順）
1. 残す比較と共有の部品を確定する: 候補の学習を新しいモジュールへ移す（呼び出し側・研究用の道具のpatchの先も直す）。goldenの照合の「外すcase」と「除く項目」の仕組みを先に作る（まだ何も消さないので、除く一覧は空で通る）。
2. 構成と設定の入口を閉じる（要求1）。検証ありの設定を組み立てていたtest・helper・fixtureを、外すか、検証なしへ直す（要求4.5）。
3. 実行と状態を消す（要求2）。
4. 記録・指標・保存の形を消す（要求3）。goldenの除く一覧を確定する（要求4.2）。
5. 研究用の道具・文書を直し、版15を足す（要求4.3・4.4・6）。版15のgoldenの生成は主担当。

段1〜4の各終わりに、版7〜14の残すcaseの数値の照合を通す（版1〜6は、残すFedDrift・FedAvgのcaseと`fedsda-without-candidate-validation`ほかの検証なしのcase）。

## goldenの照合
- 外すcase: 各版で、`build_case_settings`が候補検証ありの方針を選ぶcase。版ごとの一覧をtestに定数で持ち、外す理由（「候補検証の処理を消した。基準版15のspec」）を書く。
  一覧は、保存済みのgoldenの設定の記録から機械的に確かめる（方針の値が検証ありのcaseと一覧が一致することをtestにする）。
- 除く項目: 記録の表`candidate_validation_decision.*`の列、runの指標の項目、設定の項目、実行の段（`incomplete_candidate_validation_finalization`）。それぞれ、保存済みのgoldenの中での位置（鍵の道）で列挙する。
  除いた後に残る差は全部失敗にする。
- 保存した設定と、組み立てた設定の照合（`behavior_golden_cases.py`の1458行付近）も、同じ除く一覧で行う。
- 版15: case名の一覧は版14から4 caseを除いたもの。版15のgoldenを作った後、4.4のtestで版14と比べる。

## 乱数の状態の照合（要求4.7）
- 対象: 版15の18 caseのうちFedSDAの14 case（goldenのcaseの条件の小さいrun）。比較の地点: runの終わり（全clientの終端の処理の後、runの指標の導出の前）。
- 記録するもの: runが借りるPython `Random`の`getstate()`、NumPyの`RandomState`の`get_state()`、torchのCPUの`get_rng_state()`（runの中で使う乱数源の全部。実装担当が、runが使う乱数源を確かめて一覧にする）。
- fixtureはtask 1の開始時のHEADで作り、testのデータとしてcommitする。許容差なし。消す段で一致しなくなったら、その段の変更を直す（fixtureを直さない）。

## 数値の不変の根拠（調査の報告の4）
- 毎標本の検証の進行・終端の処理は、検証が無いとすぐ戻る。呼び出しを消しても、乱数・パラメータ・記録は変わらない見込み。
- 参照モデルの固定（torchの乱数を使う）は、検証の開始の側だけ。
- 候補の生成・学習の回数・初期化・学習の順は、移すときに変えない。
- 実時間は変わってよい。runの指標のうち時間の項目は、今のgoldenの照合の扱いのまま。

## 保存の形とrunの識別子
- FedSDAのrunは、保存した設定の項目が減るので識別子のhashが変わる。前半の`<データ>_interval<間隔>_seed<seed>`は変わらない。
- 過去の結果は読める。集計の道具（`tools/experiments/accuracy_periods.py`、`tools/research_feddrift_delta_comparison.py`ほか）はrunを名前の前半で対応させ、消した項目を読まない（実装担当が確かめ、違う道具があれば報告する）。
- 過去の掃引の置き場所への再開・seedの追加はできない（新しい置き場所に流す）。文書に書く。

## 範囲の外
- 旧実装の変更、`docs/reference/options.md`の手での編集、構成`fedsda-without-candidate-validation`の改名、候補の共有部の更新・警報の区間の選び方などの、共有の値の削除。
