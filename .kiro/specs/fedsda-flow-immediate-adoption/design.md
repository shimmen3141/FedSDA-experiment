# Design

## Boundary
変更対象は既存HTMLと本specだけ。refactoringの実装・進行中文書、実験条件・golden・他の図示は変更しない。

## 実装との対応
参照は隣のrefactoring worktreeの現行実装（調査時HEAD 169bec6、既定化作業は進行中）。
- runtime/fedsda_final_configuration_run_settings.py: acceptanceはwithout_validation、candidate共有部はkeep_shared_features_fixed。
- runtime/alarm_change_interval_resolution.py: 再利用評価→初期値選択→train_candidate_on_alarm_change_interval→adopt_candidate_as_current_training_model。同じ応答内で完結しsessionを開始しない。
- learning/training/candidate_epoch_training.py・joint_model_parameter_update.py: 共有部をno_gradで推論しoptimizerを進めず、固有部だけ更新する。早期終了用の内部検証と、削除する将来標本による候補採否検証は別。
- runtime/held_model_training_request_handling.py: 通常共同学習はupdate_shared_features=True。
- learning/training/adopted_candidate_shared_feature_integration.py: 採用時のコピー関数は残るが、候補学習で共有部の値を変えない。図は値の不変を表現し、共有部自体を使わないとは描かない。

## 表示・遷移
既存の候補検証領域の外枠を新モデル作成へ転用し、共有部（固定）・固有部（学習）・即時追加を縦に並べる。固定共有部も順伝播に使うため、候補学習コマでは見せ、更新する固有部と区別した明示を置く。新規候補への入力はFIFO変化区間、追加先は更新用の保有モデル群、標本の確定先はモデル別ストア。

上段はフェーズ1〜4を保持し、5を「新モデル作成」（選択肢なし）、6サーバ共有を保持する。新規経路はC01（変化区間を渡す）→C02（固定共有部で特徴抽出・固有部を学習）→C03（即時採用・標本確定）。既存再利用はA07からサーバ共有へ。R01/R02は新規経路のみ。旧V/F frame・validation選択と専用線を除去し、採用状態はC03到達から決める。時点はfutureではなく現在の標本。

## 検証
変更前の受入検査は新規遷移・旧検証UI不在で失敗することを確認。変更後は全経路でactor/edge参照解決、順序、採用状態、フェーズ対象外、循環を検査する。1280×720・1440×900・390×844の表示と代表コマをブラウザ確認。モデル領域・サーバ関連frameの差分を照合する。実験コードのpytestは対象外。
