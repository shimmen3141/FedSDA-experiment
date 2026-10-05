# 要求: optimizer状態の所有と明示リセット

## 目的と範囲
固定のparameter列と機能別設定に対して、学習で使うoptimizerとその再生成を管理する。
モデル生成/parameter置換、共有接続、model ID管理、学習・候補・同期・reset時機は上位の責務。
学習率変更、空parameter列のoptimizer、checkpoint、optimizer stateの移送は今回扱わない。

## 1. 生成と借用
- 1.1 When 正常なparameter列と固定設定を渡したとき, the optimizer状態管理器 shall 既存の生成部品と同じAdam/SGDを空の内部stateで生成し、parameterの参照・順序・値・gradを保持する。
- 1.2 When 学習用optimizerを取得したとき, the optimizer状態管理器 shall 現在の同じoptimizer参照を返す。外側からの学習更新を次回取得にも反映する。
- 1.3 If 生成部品で拒否される入力を渡した場合, then the optimizer状態管理器 shall 生成を拒否し、渡されたparameterの値やgradを変更しない。

## 2. 明示リセット
- 2.1 When 明示的にリセットしたとき, the optimizer状態管理器 shall 同じ固定parameter列と設定で新optimizerを生成し、以前のoptimizerの学習stateを引き継がない。parameterの値・gradは変更しない。
- 2.2 When 一つの管理器をリセットしたとき, the optimizer状態管理器 shall 別管理器のoptimizer/stateを変更しない。共有部と概念固有部を別々にリセットできる。
- 2.3 If 再生成が失敗した場合, then the optimizer状態管理器 shall 現在のoptimizer参照とstateを保持する。
- 2.4 When 上位がリセット後に現在のoptimizerを取得したとき, the optimizer状態管理器 shall 新optimizerの参照を返す。既存のHeldModelTrainingBindingなどの借用記録は旧参照を保持する。上位が現在参照で新しい借用記録を作ってから学習する。

## 3. 責務分離
- 3.1 The optimizer状態管理器 shall parameter列を借用し、optimizerを所有する。NNやparameterを新規生成せず、乱数状態を変更しない。
- 3.2 The optimizer状態管理器 shall 学習・モデル再接続・候補採否・同期を自動で実行せず、リセット時機を外側へ残す。
