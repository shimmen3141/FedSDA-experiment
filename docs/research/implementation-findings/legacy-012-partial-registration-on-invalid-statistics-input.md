# LEGACY-012: 新規モデル登録が途中で失敗すると共有部の上書きと採番消費が残る

- 発見日: 2026-10-07。対象: 旧clients/base.py::_register_trained_new_model、clients/shared_backbone.py::_prepare_model_for_registration、clients/fedsda.py::_finalize_forward_validationの採用分岐、固定基準748c3aa。
- 発見spec: adopted-candidate-initial-local-registration、adopted-candidate-local-adoption。種別: 契約外入力での部分更新。状態: 再現済み・未修正。

## 再現と観測

実旧`SharedBackboneClassConditionalESRFedSDAClient`（保有モデル4・9が共有部を共有、現在ID 9、next_temp_id=-103）へ実ForwardValidationSessionを与え、採用条件（候補損失0.01、参照損失0.9、forward_persistent）を満たした上で、4クラスのsession.training_yの先頭ラベルを範囲外の9.0にして実_finalize_forward_validation(57)を呼ぶ。構築は tests/refactoring/test_adopted_candidate_local_adoption.py の build_local_adoption_oracle(class_count=4) と同じ。
2026-10-07基準venvで実行し、RuntimeError（損失評価のgather）と次の状態を観測した。

- 共有部の値は候補の共有部の値で上書き済み。保有モデル4・9の出力も変わる。
- 候補は共有部へ接続済みで、models=(4, 9, -103)に一時IDで登録済み。
- model_statsには-103がなく(4, 9)のまま。pending_model_paramsはNone。
- next_temp_idは-103から-104へ進む。
- current_model_idは9のまま、_forward_validationのsessionは残る。

旧は、採番→共有部への値反映と接続→models登録→損失評価→統計→送信保留の順に更新し、損失評価より前の更新を巻き戻さない。

## 影響と今回の扱い

範囲外ラベルや空の区間は、通常の実験経路では候補の区間学習が先に失敗するため、登録まで到達しない。正常client経路・過去実験成果への影響は未確認で、過去成果が破損しているとは判断しない。
新実装は、損失評価・初期統計・parameter snapshotを共有反映より前に生成し、入力拒否時は共有部・接続・optimizer・各ownerを変更しない（adopted-candidate-initial-local-registrationの要求2.6/2.7、拒否47条件）。採番も登録の検証が通るまで確定しない（adopted-candidate-local-adoptionの要求2.4、拒否40条件）。正常時の値・状態・採番列は旧と一致する。旧production・goldenは修正しない。

## 将来の修正候補

旧を保守する場合は、登録の入口で特徴・ラベルを検査するか、損失評価を共有部の上書きより前に行う。新実装へ移行するなら旧の修正は不要。担当/修正commitは未定。
