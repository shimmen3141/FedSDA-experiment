# 命名表: initial-model-pretraining

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## ファイル・型

| 提案名 | 種類 | 役割・所有する状態 | 関連する名前との違い |
|---|---|---|---|
| `initial_model_pretraining_settings` | module（learning/training） | 事前学習の条件の設定型 | `candidate_epoch_training_settings`は、警報の後に作る候補の学習の条件 |
| `InitialModelPretrainingSettings` | frozen dataclass | 事前学習の標本数、epoch数、batchの件数 | `LocalTrainingScheduleSettings`は、標本処理の中のローカル学習の間隔と回数 |
| `initial_model_pretraining` | module（runtime） | 事前学習の関数と、その結果 | `fedsda_run_client`は、この結果からclientを組み立てる |
| `PretrainedInitialModel` | frozen dataclass | 事前学習の結果: 分類器、概念固有部と共有部のoptimizerの状態、損失統計（clientの組立ての引数に対応する） | `HeldModelTrainingState`は、clientが保有するモデル1つ（分類器と概念固有部のoptimizerの状態）。この記録は、clientへ渡す前の、共有部のoptimizerの状態と統計も含む |

## 関数・メソッド

| 提案名 | 入力 | 出力 | 役割 | 状態更新・副作用 |
|---|---|---|---|---|
| `pretrain_initial_model` | 事前学習の設定、モデル構造の設定、隠れ層の幅、クラス数、optimizerの設定、標本生成器、乱数生成器 | `PretrainedInitialModel` | 分類器を作り、概念0の標本で学習し、その標本での損失統計を求める | torchの全体の乱数、標本生成器の乱数、借りた乱数生成器を消費する |

## 判断が必要な点

- 「pretraining」は、旧の用語（`PRETRAIN_*`、`_pretrain_initial_model`）を引き継ぐ。runの最初に1回だけ行う学習で、clientのローカル学習や候補の学習と区別する。
