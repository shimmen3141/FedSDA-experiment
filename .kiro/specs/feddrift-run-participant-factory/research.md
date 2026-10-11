# Research & Design Decisions

## Summary
旧experiment._setup_server_and_clientsは共通事前学習→server登録→client順のdeepcopy。新のserver構築はclientを入力に要るためclient→serverの順にする。いずれもモデル生成/forwardを追加しないため正常値・乱数は変わらないことを実旧対照で示す。

## Research Log
- 旧experiment.pyの294〜358行、SimpleMLP、FedDriftClient/Serverと既存新factory/初期学習/client/serverを確認。
- 共通pretrainはindependentモデルにも対応済み。新optimizerは旧NEW_MODEL_LR、初期/再配布optimizerは旧SimpleMLP既定の条件を明示対応する。
- 既存RunParticipantsはFedSDAのstream操作契約で、FedDriftの時刻同期を表現しない。専用集合の方が小さく、偽装cast・新Protocol抽象化を避けられる。
- kiro-spec-init/requirements/design/tasks、kiro-implのmanual --review off、kiro-validate-implとfable-methodを参照。EARS/境界・設計・task graph gateを主担当inlineで実施し、実装前に仕様書群全体の独立レビューで確認する。新ライブラリ・外部APIなし。

## Risks
事前学習後に初めて失敗する設定条件を残さない。条件やfrozenを回避した不正値は最初に検査する。MNISTデータ取得と全体run/goldenは今回の範囲外。

- task実行前の実旧oracle確認: _setup_server_and_clientsをFedDrift ModeSpec、sine2、事前学習7件/1epoch/3batch、client 2で実行し、ID 0と統計n=7を確認。数値手順の複製なし。
