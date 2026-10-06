# LEGACY-013: 使用済みの一時IDへの再登録が既存モデルを黙って置換する

- 発見日: 2026-10-07。対象: 旧clients/base.py::_register_trained_new_model、固定基準748c3aa。
- 発見spec: adopted-candidate-initial-local-registration。種別: 契約外入力での状態置換。状態: 再現済み・未修正。

## 再現と観測

実旧`SharedBackboneClassConditionalESRFedSDAClient`（保有モデル4・-7、-7の統計件数4）へ、別の候補を同じ一時ID -7で実_register_trained_new_model(-7, 候補, 特徴5件, ラベル, pending_ready=False)する。構築は tests/refactoring/test_adopted_candidate_initial_local_registration.py の build_initial_registration_oracle(held_model_ids=(4, -7)) と同じ。
2026-10-07基準venvで実行し、例外なしで、models[-7]が新しい候補へ置き換わり、model_stats[-7]の件数が4から5（新しい初期統計）へ置き換わることを観測した。modelsの順序は(4, -7)のまま。元のモデルと統計は参照を失う。train_data_store等の他の項目は旧IDのまま残る。

## 影響と今回の扱い

旧の一時IDは_alloc_temp_idが単調に減らす値で、同じclient内で同じ値を二度返さない。正常経路で同じ一時IDへ再登録する呼出しは確認していない。正常client経路・過去実験成果への影響は未確認で、過去成果が破損しているとは判断しない。
新実装は、一時IDが学習状態一覧・損失統計・送信保留・学習標本・学習計数・現在の学習帰属IDのいずれかで使用済みなら、状態変更より前にValueErrorで拒否する。旧production・goldenは修正しない。

## 将来の修正候補

旧を保守する場合は、登録の入口で一時IDの未使用を検査する。担当/修正commitは未定。
