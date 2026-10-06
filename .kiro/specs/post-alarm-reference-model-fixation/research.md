# 根拠と判断

- 旧FedSDAClient._snapshot_reference_models: self.models.items()の順に、snapshot=self._new_model()（独立した共有部を持つ新しいモデル）、snapshot.set_params(model.get_params())（全parameterの複製を読込み）を行い、{model_id: snapshot}を返す。
- 旧_begin_forward_validation: 候補の生成→学習→参照の複製→（shadow_tournamentだけ参照も学習）→履歴平均の取得→session生成の順。履歴平均は、参照ごとにmodel_stats.get(model_id)があり件数nが2以上ならfloat(stats["mean"])を対応へ入れる（平均0も入れる。件数2未満と統計なしは入れない）。
- 乱数: 旧_new_modelはモデルの初期化でCPUのtorch乱数を消費する。新ResidualAdapterClassifierの生成は旧と同じ順で乱数を消費する（residual-adapter-model-architectureで実旧と照合済み）。設計前の確認で、実旧_snapshot_reference_models（保有モデル4・9）の後の乱数状態が、新の分類器を保有順に同じ構造で生成した後の乱数状態と一致した。値は直後の読込みで上書きされるが、乱数の消費量と順序は後続の処理（候補の学習や以後の標本処理）の数値へ影響するため維持する。optimizerの生成は乱数を消費しない。
- 履歴平均は移植済みのselect_post_alarm_reference_historical_mean_loss（2件以上の保存平均を零も含めて使用）で、モデルの全体統計から選ぶ。
- 新の値の複製は移植済みのsnapshot_classifier_parameters（全parameterの検証と独立clone）を使い、新しい分類器へ標準のload_state_dictで読み込む。構造（構造設定・特徴数・隠れ層幅・クラス数）は保有モデルの分類器から読む。
- 旧は参照モデルにoptimizerも作る（モデル構築の一部）。最終構成では参照を学習しないため使われない。新は参照に個別optimizer管理器を作らない。
- 旧は保有0件で空の対応を返し、後段の確定でmin()が失敗する。新の損失収集は参照ID列が空だと拒否するので、本specは保有0件を入口で拒否する。
- 設計前の確認: 上流の採用oracleの実旧clientで、実旧_begin_forward_validationを候補の学習（_train_new_model）だけ無効化して実行できた。統計を4:n=1/平均0.9、9:n=5/平均0.0にすると、参照ID(4,9)、履歴平均{9: 0.0}、規定件数4、候補は現行モデルと同じ値の独立モデルになった。実旧のsession開始→観測→確定までを通したoracleとして使える。
- 2026-10-07のユーザー判断で旧shadow_tournamentは当面不要。参照を学習させる処理は移植しない。
