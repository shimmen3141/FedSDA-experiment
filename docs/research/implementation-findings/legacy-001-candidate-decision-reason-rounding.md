# LEGACY-001: 候補採否と区間margin理由の丸め不整合

## 状態と対象

- 発見日: 2026-10-04。種別: 数値境界の診断不整合。状態: 再現済み・未修正。
- 基準: 旧commit748c3aa、forward_persistent。
- 対象: federated_drift_experiment/provisional_model.pyのhas_disjoint_validation_advantageとdisjoint_validation_rejection_reason、clients/fedsda.pyの_finalize_forward_validation。
- 発見spec: [post-alarm-candidate-loss-evaluation](../../../.kiro/specs/post-alarm-candidate-loss-evaluation/README.md)。
- 新実装: 同specで旧演算順と両方向の不一致を維持して移植済み、修正していない。

## 観測と再現

採否はfloat32平均をPython floatへ取り出してcandidate_mean < reference_mean - min_deltaで判定する。
理由は参照−候補の平均差を先にfloat32で計算してから、必要改善量以下か判定する。演算精度/順序の違いで以下を再現した。

| 候補損失列 | 参照損失列 | 必要改善量 | 旧採否 | 旧理由 |
|---|---|---|---|---|
| 0.2, 0.2 | 0.8, 0.8 | 0.60000001 | 棄却 | accepted |
| 0.30000004, 0.30000004 | 0.80000007, 0.80000007 | 0.5000000149011612 | 採用 | first_and_second |

前半一件/後半一件、履歴平均を持たない参照との比較。主担当とLunaが旧finalizeのdecision captureで確認した。
期待する診断は採否と不合格理由の整合。独立margin診断を残すなら採否理由とは意味を分ける。

worktreeルート、共有golden環境で以下を実行する。テストは現在、不整合を保存することを検証する。

```powershell
$env:TMP=(Resolve-Path ../../venv/refactoring-tests).Path
$env:TEMP=$env:TMP
$env:MPLCONFIGDIR=(Resolve-Path ../../venv/matplotlib-cache).Path
../../venv/Scripts/python.exe -m pytest tests/refactoring/test_post_alarm_candidate_loss_evaluation.py -k rounding_boundary -q -p no:cacheprovider
```

[統合証拠](../../../.kiro/specs/post-alarm-candidate-loss-evaluation/integration-validation.md)、移植完了commit60fa55d。全tests2504 passed / 3 skipped、golden更新なし。

## 影響と未確認事項

実モデル操作は採否boolを使う。理由を使う集計/表示に誤解を生む可能性がある。
上の再現条件は最終goldenの通常改善量1e-4とは異なる。通常設定や過去研究実験での発生・研究指標への影響は未確認で、影響があったとは断定しない。

## 修正案と検証

担当: 候補損失評価の判断/診断。修正specは未作成。

1. 採否式を正本として区間合否から理由を作る。採否を維持して診断だけ整合させる案。
2. margin式へ統一する。採否が変わり得るためアルゴリズム変更として判断する案。
3. 独立margin値を別の診断項目へ分離する案。

両方向の再現を残し、修正後の期待理由/採否を別テストで検証する。旧11/最終3goldenの数値・イベント列、既存成果の該当条件と理由を照合し、変わる指標と再実験要否を判断する。環境差を根拠にgoldenを更新しない。
採用案・修正commit・過去成果への影響・最終検証結果は未定。決定後このファイルと一覧を更新する。

## 判断（2026-10-13、Claude Codeの提案をユーザーが了承）

採用（優先度は低い）。修正案1（採否の演算を正本にし、区間ごとの合否から理由を作る）。採否は変えない。計測（2026-10-13、新実装`ab10ad4`へ読取りだけの計数を差し込んだ。提案構成、client 10、標本2500、集約間隔50、sine2・sea4・circle2・mnist2のseed 0と1の計8 run、Windows。8 run・1つの規模だけなので、頻度は目安）では、検証の確定129件で、採否と理由の食い違いは0件。goldenは変わらない見込み（未検証）。

## 時期の決定（2026-10-13、ユーザー）

診断だけの修正だが、理由の文字列は、重要動作のgolden（基準版1〜3）の記録の列に入っている。直すと、古い基準版の照合が通らなくなるので、いまは直さない。**対応しないのではなく、旧実装を外したあと（古い基準版を、旧と同じ扱いで照合し続ける必要が無くなった後）に直す候補**とする。修正の内容は、上の「判断（2026-10-13）」のとおり（採否の演算を正本にし、区間ごとの合否から理由を作る）。
