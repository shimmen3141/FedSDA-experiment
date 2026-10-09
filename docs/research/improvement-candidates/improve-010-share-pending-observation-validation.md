# IMPROVE-010. 保留標本の並びの検査を1箇所にまとめる

- 記録日: 2026-10-09。種別: 簡略化（重複の解消）。状態: 未検証・未採用。
- 対象: 新実装`src/federated_learning_experiments/runtime/alarm_buffer_response.py`の`_validate_buffered_alarm_observations`と、`runtime/released_pending_sample_assignment.py`の冒頭の検査。発見spec: released-pending-sample-assignment（基点commit `0f075c0`）。

## 確認した事実

警報のときの応答と、警報のない標本での確定は、どちらも呼出し側から全保留標本（位置・標本・概念ID）のtupleを受け取り、「exact tuple、各要素がexact `IndexedObservedTrainingSample`、位置がbuiltin int、位置の並びが保留位置のownerの並びと一致」を確かめる。前者は非公開のhelper、後者は関数の冒頭に同じ趣旨の検査を持つ（前者は位置が非負であることも見る。保留位置のownerが非負の位置しか持たないので、並びの一致の検査に含まれる）。

追記（2026-10-10、observed-sample-prediction）: 標本1件の型と形の検査（exactな記録、位置が非負のbuiltin int、概念IDが整数またはなし、特徴が1行の2次元、ラベルが1行1列）も、`runtime/observed_sample_processing.py`と`runtime/observed_sample_prediction.py`の2箇所に同じ内容である（予測の関数は単独でも呼べるので、自分でも確かめる）。まとめるなら、この検査も対象にする。

## 案

保留標本そのものを持つowner（標本1件の処理全体のspecで作る見込み）が、位置と標本を一緒に持てば、呼出し側が並びを合わせて渡す必要がなくなり、2箇所の検査は不要になる。ownerを作らない場合は、検査を公開の関数1つへまとめる。

## 挙動への影響

正常な入力での挙動は変わらない。検査の位置と例外の文言が変わりうる。

## 必要な検証

既存の実旧対照と拒否のtestが変わらず成功すること。採否・変更commitは未定。標本1件の処理全体のspecで、保留標本のownerの置き場所と合わせて判断する。
