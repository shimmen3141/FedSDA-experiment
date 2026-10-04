# 調査・設計判断

## Integration-focused light discovery
旧models.py:208の_build_component_optimizer、共有model.update/reset、clients/shared_backbone.py:234のjoint経路、既存LocalTrainingSettingsを調査した。新外部依存は追加しない。
最終平時学習は全参加batchの共有forwardとloss×標本数のsum/総数でbackwardし、共有step1回→個別step入力順。単一model.updateだけの移植は最終joint経路を満たさない。
その前提のoptimizer生成を今回の責務にした。学習進行・共有optimizer所有は後続。

## 旧根拠
Adamはlr/weight_decay/amsgrad、SGDはlrだけ。初期stateは空、生成時CPU RNG不変を調査者が実測。
Adam既定betas=(0.9,0.999)、eps=1e-8、SGD momentum/weight_decay=0。最終goldenはAdam/decay0.001/AMSGrad、SINE/SEAのlr0.01、MNIST0.001。
データ別lrの解決・resetは上位が保持し、生成関数は具体値だけ受け取る。

## 主担当synthesis
方式別dataclassに分けるとSGDへ無意味なAdam条件を要求せずに済む。既存field validatorを使うためAMSGradはadam_variantのstandard/amsgradとして表す。
標準optimizer/state_dictを採用しwrapper/registry/state互換readerは不要。二つの設定型と一つの生成関数で閉じる。
主担当保存前gateは9条件・具体path/依存・所有/拒否・実旧比較・後続境界を確認してPASS。

## 空共有部の観測
旧ResidualAdapterMLPへ空幅SharedFeatureBackboneを渡すとoptimizer got an empty parameter listで生成失敗することを調査者が再現した。通常goldenは非空幅。
新モデル構造は空幅identityを受理する。今回builderは空Parameterを拒否するが、後続学習は共有部が空ならoptimizerを生成しない扱いを検討する。旧productionは未変更、過去成果への影響未確認。実証記録は後続の本spec証拠で台帳へ接続する。
