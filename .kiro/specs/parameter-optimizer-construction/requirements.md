# 要求: パラメータoptimizerの設定と生成

## 概要・範囲
上位が指定したモデルParameter集合とoptimizer条件から、新しいoptimizerを一つ生成する。
Adamと学習率だけのSGDを対象とし、Adamでは標準版/AMSGradとweight decayを明示する。データセット別学習率の解決は上位が行う。
前specに合わせCPU float32 strided Parameterに限定する。モデルへのattach、共有optimizerの管理、reset/状態復元、loss/backward/stepの実行、epoch/batch抽出、共同学習/候補進行/全体runは含めない。
空の共有特徴抽出部を学習するとき、上位が共有optimizerを生成しない判断は後続の学習specで扱う。

## 1. 生成結果
1.1 When Adamの条件とParameter列を与えたとき, the Optimizer Builder shall 明示学習率・weight decay・標準版またはAMSGradの条件を用い、その他の条件を旧生成と一致させた新規optimizerを返す。
1.2 When SGDの条件とParameter列を与えたとき, the Optimizer Builder shall 学習率だけを設定した旧と同じmomentumなし・weight decayなしの新規optimizerを返し、Adam専用条件を入力に要求しない。
1.3 When optimizerを生成するとき, the Optimizer Builder shall 渡されたParameterの順序と同一参照を保持し、コピーせず、初期状態が空の独立optimizerを生成し、Parameter値・gradを変更しない。

## 2. 入力・環境
2.1 If 設定の型/選択肢/数値が不正であるとき, then the Optimizer Builder shall 項目と理由を示して生成前に拒否する。学習率とAdam weight decayはboolを除く有限のbuiltin int/floatで0以上、Adam variantはstandard/amsgradの二つとする。
2.2 If Parameter列が非tuple/空、要素がParameter以外、CPU float32 strided以外、または同一Parameterが重複しているとき, then the Optimizer Builder shall 位置または原因を示して生成前に拒否し、入力とgradを変更しない。正常入力のgrad=Noneとrequires_grad=Falseは受理する。
2.3 The Optimizer Builder shall 生成・拒否時にCPU/Python/NumPy乱数、既定dtype/device、grad有効設定を変更しない。

## 3. 移植・接続検証
3.1 When このspecを検証するとき, the Migration Tests shall 実旧生成との全parameter group条件/参照順/初期stateを照合し、上位から同じ勾配を与えた複数更新後の全Parameter値・optimizer stateも一致することを確認する。
3.2 When このspecを検証するとき, the Migration Tests shall 新Residual Adapterモデルから共有抽出部と個別adapter/分類層のParameterを明示選択して生成でき、両集合が重複せず、モデルにoptimizer状態を追加しないことを確認する。
3.3 When このspecを完了するとき, the Migration Tests shall exact依存・旧importなしの独立起動・goldenを含む全testsと旧production/golden無差分を確認し、後続学習との境界を記録する。
