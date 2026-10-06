# 設計: 損失統計のモデルID付替え

revision: 1

## 境界と依存

既存ModelAndClassLossStatisticsStoreが所有する辞書内だけを更新する。
上位が正式IDを決め、各ownerの付替えと現在帰属ID更新・保留解除を組み立てる。
既存依存（stdlibとbounded_loss_momentsの公開2symbol）を増やさない。
store全体のサーバ対応表・統計選択は別spec、他ownerやpendingへ依存しない。
private属性の改ざん・並行変更はpublic契約外。

## Components and Interfaces

`reassign_model_loss_statistics_id(*, original_model_id:int, reassigned_model_id:int)->None`を既存storeへ追加する。
既存`_validate_identifier`で両IDを先に検査する。欠落元はno-op。
元がある場合、内部immutable統計をpopし、変更先へ代入する。
全体/classの統計をコピー・合成・再計算せず、既存の取得時copy境界を維持する。
異なる既存先は位置を維持して丸ごと上書き。未登録先・同IDはpop後の末尾。
検査不正は既存TypeError、項目名と理由を示す。戻り値は常にNone。
正負ID制限・登録成功判定・欠落モデルの再生成・multi-owner transactionは上位責務。

## File Structure Plan

|パス|変更|責務|
|---|---|---|
|src/federated_learning_experiments/learning/loss_statistics/model_and_class_loss_statistics.py|変更|既存ownerの単一ID付替えだけ|
|tests/refactoring/test_loss_statistics_model_id_reassignment.py|新規|順序/衝突/欠落/同ID/拒否、実旧正式登録、保留とのtest-only接続|
|対象spec・steering resume/roadmap|新規/更新|承認/実測/現在地|

AST guardは既存moduleの許可/拒否検査を再実行する。依存を増やさないためguard自体を変更しない。

## 検証・traceability

|要件|検証|
|---|---|
|1.1–1.4|実旧BaseClient.confirm_model_registrationへの負元ID対照、先既存/欠落・元欠落・空・クラス順・0件・件数差上書き。汎用signed/同IDは独立明示期待順|
|2.1|両IDのbool/float/str/None/int派生型を、元有/無・先有/無で拒否、snapshot全値・順序保持、入力検査が欠落判定より先|
|2.2|付替え前取得値の保持、変更先の追加Welford更新・同IDと別ID、返却値を意図的に改変してstore非波及|
|2.3|class2/4×delay1/2の実旧登録→保留→統計更新→ready→確認と、新producer/初期統計/store/pending→現在統計取得→付替え→明示clear。固定parameter全値、現在統計全field、保留を付替えだけでは変更しないことを検証|
|2.4|Python/NumPy/Torch RNG・既定dtype/device保持、既存AST、fresh新CPU snapshot/store/pending起動と旧非import、全pytest旧11/最終3golden・品質・固定旧差分・hash/JUnit|

task1はテストを先にREDにしてからstore APIを追加する。task2はtest-only接続（RED N/A）。task3は統合gate。
仕様の8条件を全て確認し、各taskの独立Lunaレビュー後、別feature最終GOを得る。

## リスクと再検証

同IDは元位置維持のno-opではなく末尾へ移動する。旧pop代入の順序を明示保持する。
衝突は件数最大選択ではなく元統計の上書き。変更する場合は登録/同期の別アルゴリズム変更として扱う。
storeに登録手順を内包せず、現在帰属ID・保留の更新は後続の上位接続へ残す。
