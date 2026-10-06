# 根拠と判断

- 旧BaseClient._absorb_into_store(model_id, data_list): model=self.models[model_id]（未保有はKeyError、空列でも先に評価される）。各要素dについて順に、train_data_store[model_id].append(d)→len(d)>=3なら_record_model_concept(model_id, d[2])（NoneはNo-op）→no_gradで"statistics"計算量記録とl_val=model.get_absolute_error(d[0], d[1])→class_id=int(d[1].view(-1)[0].item())→_update_model_stats(model_id, l_val, class_id=class_id)。
- 呼出しは旧fedsda.pyの前向き検証確定の棄却/再利用/維持分岐、終端の未完了session回収、警報後の複数経路、旧feddrift.py。採用分岐だけは使わない（LEGACY-014）。FIFOから1件ずつ確定する経路（fedsda.py 512-524行）は同じ4更新を統計→標本→概念の順でinlineに行う別実装で、本specの範囲外。
- get_absolute_error: no_gradでper_sample_errorを求め、要素1ならitem()、複数なら平均のfloat。FedSDAの保留標本は1標本ずつ（特徴[1,F]、ラベル[1,1]）で、class_idは先頭ラベル。複数行の要素では損失が平均・classが先頭だけになる。新は1標本1recordを契約とし、複数行のrecordを拒否する。旧feddrift.pyのbatch要素の扱いはFedDrift移植時に確認する。
- 損失は標本ごとに現在のモデルparameterで評価する。吸収中にモデルは更新されないので、全標本の損失を先に評価しても各標本の値は同じ。新は全検証と全損失評価を状態変更より前に行い、その後に標本順で各ownerを更新する。標本ごとの独立forward（batchにまとめない）を維持し、旧と同じ数値を得る。
- 更新の対応: 学習標本はModelTrainingSampleStore.append_model_training_samples（1件ずつ。空列ではモデルの標本列を作らない旧の挙動に合わせ、呼ばない）、概念はModelTrainingAndAssignmentCountsStore.record_assigned_sample_concept（Noneは記録なし）、損失はevaluate_classifier_per_sample_bounded_losses、統計はModelAndClassLossStatisticsStore.record_assigned_loss（未登録モデルは件数0から開始。旧setdefaultと同じ）。
- 旧標本は(特徴, ラベル[, 真の概念])のtuple。新ObservedTrainingSampleは概念を持たないので、同じ長さの概念ID列を別引数で受け取る。真の概念は診断専用で、予測・学習判断へ使わない。
- 旧の"statistics"計算量記録は診断counterで未移植。本specは記録しない。
- 旧は途中の標本で失敗すると、それまでの標本の更新が残る（部分更新）。本specの実装時に実旧で再現し、implementation-findingsへ記録する。
