# 命名 revision 2

|名前|型・役割・更新する状態・区別|
|---|---|
|learning/training/temporary_model_id_allocation.py|サーバ確認前の一時モデルIDの採番。登録（runtime/adopted_candidate_initial_local_registration.py）や正式ID確認ではない|
|TemporaryModelIdAllocator|次に採番する負の一時IDを一つ所有する。クライアントごとに一つ。モデルや登録状態は持たない|
|client_id|exact int非負、初期値の計算に使うクライアント番号。既存data層のclient_idと同じ意味。保持しない|
|next_temporary_model_id|property、次に採番する値。読取りだけ。temporary_model_id（登録APIへ渡す採番済みの値）と区別|
|allocate_temporary_model_id|次IDを返し内部値を1減らす。get_*（非更新取得）ではなく採番による状態更新を表す|
|_next_temporary_model_id|内部状態。次に返す値|

## 似たIDとの区別

|名前|意味|値の範囲・所有|
|---|---|---|
|next_temporary_model_id|まだ採番していない、次に返す一時ID|負。本ownerが所有|
|temporary_model_id|採番済みで、登録APIへ渡すサーバ確認前のクライアント内ID|負。呼出し側の値。登録後は各ownerのkey|
|registered_global_model_id|サーバが採番して通知した正式ID|非負。登録確認APIの引数|
|current_training_model_id|現在の学習帰属ID。一時IDのことも正式IDのこともある|CurrentTrainingModelAssignmentが所有|

## testで使用する名前

|名前|役割|
|---|---|
|build_legacy_temporary_id_client|実旧BaseClientを実__init__で生成する最小fixture（モデル辞書はobject一つ）。初期値と採番を旧コードの実行結果から得る|
|legacy_client|固定旧採番を呼ぶclient|
|temporary_model_id_allocator|新owner|
|client_id / allocation_count / allocation_index|test軸のクライアント番号/採番回数/採番位置|
|expected_temporary_model_ids / actual_temporary_model_ids|実旧/新の採番列|
|invalid_client_id / IntSubclass|拒否入力・派生型|
|source_text / expected_acceptance|AST注入契約|
|test_*|契約を記述するpytest関数|

登録への接続testは、上流の初期ローカル登録testのbuild_initial_registration_oracle・registration_arguments等を同じ役割で再利用する。追加が必要になったら実装前に本表へ戻してレビューする。
