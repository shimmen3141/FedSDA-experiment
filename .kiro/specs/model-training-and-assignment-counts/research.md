# 旧根拠と境界

- base.py:62–66、モデル別training_examples/optimizer_steps/concept_countsは別々の初出順を持つdefaultdict。conceptは学習stepでなく帰属標本の真の概念ID診断。
- _record_model_concept:175–182はconcept_id=Noneを無視し、既知conceptをintへ変換して1追加。get_model_concept_countsはコピーを返し、欠落modelの読み取りでは項目を作らない。
- base.py:230–231/shared_backbone.py:382–383で成功したモデル更新へlen(bx)と1stepを追加。shared optimizerのstepとは区別する。候補学習の採用時には_attribute_model_training:275–284で外部計算済みの差分（0可）を加算。
- confirm_model_registration:450–460は元がある辞書だけ先へ加算。先既存は位置維持、新先は末尾、元は削除。概念件数は先concept順を保ち、元だけのconceptを末尾へ加える。
- apply_server_mapping:714–723は3辞書を独立に元順/一回getで再編、同先へ加算。再帰連鎖なし、空concept列/0件の存在も維持。
- experiment.pyで学習量診断と保存、servers/clustering.pyでoracle概念診断に使われる。今回保存/サーバ集計は含めない。
- 同一負ID確認の旧状態破損を実再現し[LEGACY-011](../../../docs/research/implementation-findings/legacy-011-same-id-registration-count-corruption.md)へ記録。新APIは同ID移管を拒否する。正常異ID移管だけ旧対照し、旧production/goldenは固定。
- 診断conceptはsigned builtin int/Noneを明示受理し旧の暗黙int変換・bool受理を持ち込まない。counter増分は非負builtin int、true conceptを予測/割当の判断入力へ流さない。
