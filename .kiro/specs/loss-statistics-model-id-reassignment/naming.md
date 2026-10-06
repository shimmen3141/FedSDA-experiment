# 命名: 損失統計のモデルID付替え

revision: 1

|名前|役割・型/単位・副作用|
|---|---|
|reassign_model_loss_statistics_id|storeの単一モデル統計のIDを付け替えるkeyword-only操作→None。対応表による選択/統合/正式登録全体と区別|
|original_model_id / reassigned_model_id|付替え元/付替え先のbuiltin int（signed、単位なし）。正式ID採番や負→非負制約を含めない|
|loss_statistics|内部でpopした既存immutable統計。数値更新やcopyは行わない|
|test_loss_statistics_model_id_reassignment.py|本specの実旧対照・拒否・上位接続|
|build_legacy_registration_client|実BaseClientメソッド用のtest-only最小所有状態。他ownerは空、統計だけ明示入力。constructorは省略|
|assert_store_statistics_match_legacy|既存全field比較helperに加えてモデル順も比較|
|test_loss_statistics_id_reassignment_matches_legacy_registration|負元ID・先衝突/欠落・元欠落・同IDを実旧確認へ照合|
|test_loss_statistics_id_reassignment_preserves_signed_id_and_same_id_order|汎用signed IDと同IDの末尾順を期待値で確認|
|test_loss_statistics_id_reassignment_rejects_invalid_ids_without_mutation|両引数・元/先の有無・不正型を拒否し状態保持|
|test_reassigned_loss_statistics_remain_independent_and_updatable|過去取得値/後続観測/取得値改変の非波及|
|test_loss_statistics_id_reassignment_connects_pending_upload_confirmation|producer・初期統計・保留・現在統計・付替え・明示解除を実旧へ照合|
|loss_statistics_store / legacy_client / pending_upload_state|新統計owner/実旧登録oracle/新保留owner|
|initial_loss_statistics_by_model_id / legacy_loss_statistics_by_model_id|new値型/old辞書の初期順序付き統計|
|previous_statistics_snapshot / previous_loss_statistics / reassigned_loss_statistics|更新前一覧/過去取得値/先IDの現在値|
|model_id / invalid_model_id / invalid_parameter_name / source_present / destination_present|ID/拒否値/拒否項目/元の有無/先の有無|
|class_count / upload_delay_round_count / classifier / source_legacy_client / training_batches|既存helperと同じ役割。source_legacy_clientはNN生成ownerで登録oracleと区別|
|initial_parameter_snapshot / input_features / observed_class_labels / per_sample_bounded_losses / initial_loss_statistics|既存producer/loss/initializerの入出力|
|current_loss_statistics / pending_model_upload / previous_random_states / previous_parameter_state|現在統計/借用送信record/共有乱数/parameterとgradの保持検査|
|observed_loss / observed_class_id / expected_model_ids / loss_statistics|観測/正解クラス/期待辞書順/統計値、既存機能と単位同じ|
|DerivedModelId|不正int派生型のtest-only型。受入可能なモデルIDではない|

既存`assert_initial_loss_statistics_match_legacy`、`build_joint_update_oracle_pair`、parameter/grad snapshot比較helperを同じ役割で借用する。
fixture・ループ位置・標準型は既存慣例を使用。公開API/永続状態の追加変更は再レビューへ戻す。
