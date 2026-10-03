# 調査と設計判断

## Summary

旧固定基準748c3aaの最終policyはforward_persistent、CPUfloat32、将来検証10件、改善量1e-4、適合距離0.1。参照順・dtype・margin式の差を保存する。
確認した資料はprovisional_model.pyのForwardCreationPolicy/適合選択/二分判定、clients/fedsda.py:193～287のfinalize、tests/test_proposed_regression.pyとgolden。
研究調査agentは編集なしで旧finalizeへ直接lossを投入して結果を確認した。主担当は旧ソースと既存設定を確認して境界を合成した。

## Research Log

- 初期参照は元Pythonlossのsum最小、入力順tie。事前float32変換は選択を変えうる。
- 適合参照だけ現在利用可能IDへ制限する。履歴欠落を除外し、CPUfloat32平均→Pythonfloatから履歴floatを引き<=距離。現行優先、代替は(mean,id)tuplemin。
- 採否はfloat32平均をPythonfloatへ取り出してreference−deltaと厳密比較。理由はfloat32平均との差を取り出して<=delta。両者の丸め境界は一致しない。
- 実測: candidate(.2,.2)、reference(.8,.8)、delta.60000001は棄却だが旧reason accepted。逆方向もLunaが確認した。
- oracleは旧sessionへlossをappend、SimpleNamespaceで旧finalizeをunbound呼出、decision appendをcaptureして専用例外で中断。モデル操作・登録・学習は実行しない。
- 現行IDは判定時の値で、警報時old_model_idの代用ではない。履歴値は警報時点のsnapshotで、統計n>=2を選ぶ処理は外部。
- readyは>=設定件数、通常は到達直後確定。新入力も>=指定件数にし、余分な検証列を受理する。

## Design Decisions

cc-sddのlight discovery・synthesisと既存命名規約を適用。一般化は収集/forwardを外部に保つloss入力だけの評価APIへ留める。
数値判断はプロジェクト固有で、第三者統計libraryへ置換せず既存torchを使用する。新依存や外部APIは導入しない。
一つのstateless moduleとimmutable結果にまとめる。収集session・汎用factory・abstractinterfaceを増やさない。
必要閾値は明示入力、既存採否設定は唯一所有者のまま。汎用core・設定選択肢は今回変更しない。

## Risks and Mitigations

### 修正候補: 採否と区間margin理由の潜在的不整合

2026-10-04のユーザー確認を受けて記録。旧実装は採否boolと理由で演算精度/順序が異なり、閾値付近で相反する結果を返す。実モデル操作は採否boolに従うが、理由を用いた集計・表示で誤解を招く可能性がある。過去実験での発生有無は未確認。
現在の移植は正常入力の旧結果と両方向の不一致を保存する。今後の修正では、判定式の統一・診断理由だけの整合化のどちらにするかを別の変更で判断し、旧/最終goldenと過去成果物の採否・理由への影響を確認する。丸めを理由にgoldenを更新しない。

旧浮動演算差を「修正」せずdirectoracleで両方向を検証する。理由新名は区間margin診断を表し、採否を逆算しない。
旧refitはここでは再学習でないため正式名に採用しない。旧理由はtest mappingだけへ置きproductionaliasにしない。
dtype/deviceは新APIで明示CPUfloat32、旧oracleのみ一時defaultfloat32と復元。入力・乱数・共有default値不変を検証。
