# 調査と判断

## 旧経路
clients/shared_backbone.pyの_train_heads_togetherはupdates_per_sample * count_multiplier回、
_sample_training_batches→空ならcontinue→共同forward/loss/backward→共有step→参加順の個別stepを繰り返す。
mean経路の個別autograd.gradは診断用であり、適用する勾配は共同lossのbackwardから得る。
clients/base.pyのtrain_step/flush_pending_updatesは回数の算出・pending状態を所有するため後続へ分離する。

## 依存とsynthesis
直前のheld-model-training-batch-samplingとjoint-model-parameter-updateが完成しており、
保有IDをNNとoptimizerへ対応付ける小さな明示記録と反復関数で接続する。
上位の標本ストアや状態を所有するclient、汎用callback/registryは導入しない。
回数は算出済みの非負整数を外側から渡す。settingsへ新しいオプションを追加する作業ではない。

## エラー範囲
回数0は旧range(0)と同じ無操作。負数/bool/小数は新公開入口の契約外として拒否する。
対応IDの構造は抽出前に確認し、Tensor・分類器・optimizerの詳細検査は完成済み部品へ委譲する。
抽出後の更新拒否はRNGを消費し得る。複数回全体のtransaction・rollbackを追加しない。
空共有の旧生成失敗はLEGACY-010で追跡済み。新入口は既存のNone共有optimizer契約を維持する。
現時点で追加の旧正常不具合は観測していない。

## 主担当requirements保存前gate
PASS。全12条件は数値ID/EARSで観測可能な振る舞い、0/空/拒否/正常状態と実旧照合を記述。
型・具体API・ファイル配置はdesign/namingへ分離する。承認はLunaレビュー後にhashを保存する。
