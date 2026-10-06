# task graph draft

仕様を実装可能な受け入れ単位へ分割する。並列実装なし。

|task|前提|成果と証拠|要件|
|---|---|---|---|
|1|requirements/design/naming承認|入力・コピー・状態・旧実snapshot対照testを先にRED、その後production GREEN。対象pytest/Ruff/PyrightとLuna実diff確認|1.1,1.2,1.3,2.1,2.2,3.1|
|2|task1承認|12条件の実旧/新学習後snapshot→既存initializer→新native復元、全optimizer状態保持。test-onlyなので新production REDはN/A|1.1,1.2,3.1,3.2|
|3|task2承認|exact AST注入RED→guard GREEN、fresh native CPU smoke、全pytest/旧11/最終3golden、品質/diff/hash、統合証拠とLuna確認|3.2,3.3|

feature完了は全task承認/check後の独立Luna統合GOで判断する。旧/golden変更なし。test-only対応を新client/全体run移植完了と扱わない。
