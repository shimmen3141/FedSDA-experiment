# Research

## 今の実装（2026-10-15、fddd5d2）
- 警報の区間の評価（`resolve_alarm_change_interval`）は、保有モデルの中から区間に合うモデルを選び、選ばれたモデルへ区間の標本を吸収する。選ばれたモデルが今のモデルなら、結果は`alarm_interval_current_model_maintained`（今のモデルを維持）。
- 候補の検証の後（`apply_post_alarm_candidate_validation_resolution`）は、候補を採用、候補を退けて今のモデルへ吸収（`candidate_rejected`）、参照のモデルを再利用（今のモデルなら`current_model_maintained`）のどれか。
- 吸収（`absorb_assigned_training_samples_into_held_model`）は、保存（`ModelTrainingSampleStore`）の列の末尾へ足す。警報の前の区間は、警報の区間の準備（`alarm_training_interval_preparation`）で先に吸収されるので、警報の区間の標本は、その後に足される。
- 本規模の比較で、sea2では警報57.5回のうち、別のモデルの再利用3.1回・候補の採用0.6回で、ほかは今のモデルの維持（または短い警報・候補検証中）だった。
- 保存は、IDの付け替え（`reassign_model_training_samples_id`）と、統合（`remap_model_training_sample_collections`。写像の先ごとに、元の列をモデルの初出順につなぐ）で、列を動かす。

## 判断
- 区切りは、警報の後に同じモデルで続けると決まった2か所だけで付ける。別のモデルへ切り替えたときは、そのモデルの古い標本は同じ概念のもの（再来）なので付けない。候補の採用では、新しいモデルの列は採用の時点から始まるので要らない。
- 区切りは、吸収が成功した後に、吸収の前の長さで付ける（吸収が失敗したときに、区切りだけが残らないように）。
- 統合で複数の列がつながったら、区切りを消す（どこが新しい概念かが決められない）。
- 区切りより前の扱いは、K・αの本規模の掃引（サーバで実行中）を見て選ぶので、使わない形（区切りより後から一様）と、αで減衰させる形の両方を足す。
