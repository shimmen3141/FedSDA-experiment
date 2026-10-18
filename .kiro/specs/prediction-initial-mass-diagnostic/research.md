# Research & Design Decisions
- FixedShareWeightControllerはID集合変更で一様化/分散0、resetを計数する。replayは証拠をclearしてから再生する。このclearを変えず、新ID追加時だけ扱う。
- 監視損失はobserved_sample_processing.pyで現在帰属の更新用モデルから別計算される。混合予測の損失とは別。ただし学習不変は実測で確認する。
- 比率保持のみは既存研究で効果が小さかったが、PとOだけだと比率保持と新ID質量の寄与を区別できないためRを同一条件の対照に置く。
- 既存Tを使い新パラメータを増やさない。新ID1個の初期質量は一様重みの1/Tとする。追加の全質量1/T案より追加数に比例し、Tは既存の2以上に従う。
- cc-sdd init/requirements/design/tasks/impl、fable-method、behavioral-spec-devの契約先行を適用。spec配置/レビュー/承認は既存プロジェクト手順を優先。既存技術内の研究介入で外部API/ライブラリ追加は不要。
