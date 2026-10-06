# 根拠と判断

- 旧FedSDAClient._observe_forward_validation(x, y, sample_idx): sessionがなければ0。no_gradで"detection"計算量を記録しcandidate_loss=float(session.candidate.per_sample_error(x,y).mean().item())、session.reference_modelsの各モデルについて計算量記録とfloat(model.per_sample_error(x,y).mean().item())。session.append_losses(candidate_loss, reference_losses)。方針がtrain_reference_shadowsなら全shadowを更新。readyでなければ0、readyなら_finalize_forward_validation(sample_idx)を呼ぶ。
- 2026-10-07のユーザー判断により、参照も学習させる方針（旧shadow_tournament）は当面不要。最終構成のforward_persistentはこの更新を行わない。本specは移植しない。
- 損失の系列・連続する標本位置・規定件数への到達は、移植済みのPostAlarmCandidateLossCollectionが所有する（観測回ごとに全入力を検査してから全系列へ追加し、到達後の追加を拒否する）。採否評価（evaluate_candidate_using_post_alarm_losses）と確定（apply_post_alarm_candidate_validation_resolution）も完了済み。本specは評価と収集の間だけを組み立て、到達後の評価・確定を自動では呼ばない。上位が到達を見て判定recordの保存等を挟めるようにする。
- 旧は1標本（特徴[1,F]、ラベル[1,1]）で呼ばれ、平均は1要素の値そのもの。新は1標本を契約とし、既存の損失評価が返す1要素をitem()でfloatにする。複数行は拒否する。
- 旧の参照モデルは警報時点の値を複製した独立モデル（旧_snapshot_reference_models）で、sessionの間は更新されない。候補は警報区間で学習済みの独立モデル。本specは分類器を受け取るだけで、生成・複製・更新を行わない。生成は候補sessionの開始のspecで扱う。
- 評価順: 旧は候補→参照（session.reference_modelsの辞書順）。forwardは乱数を消費せずモデルも変えないので、順序は結果へ影響しない。新は候補→渡された対応の順で評価する。
- 旧の"detection"計算量記録は診断counterで未移植。本specは記録しない。
- 実旧_observe_forward_validationは、上流の採用oracleの実旧clientと実ForwardValidationSession（損失列を空にし、参照モデルを実旧_snapshot_reference_modelsで作る）で実行できる見込み。規定件数目の呼出しで実旧は確定処理まで進むため、新側は観測→到達→既存の評価→既存の確定をtest-only接続し、最終状態まで照合する。REDより前に確認した: class4・規定4件で、実旧_snapshot_reference_modelsは保有モデルと同じ値・独立した共有部の参照モデル(4,9)を作り、実旧_observe_forward_validationは3件目まで0を返してsessionを保持し、4件目で確定してsessionを破棄した（候補損失0.71〜0.86、判定は棄却create_rejected、現在ID 9のまま）。
