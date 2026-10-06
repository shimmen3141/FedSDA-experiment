# 根拠と判断

- 旧BaseClient.confirm_model_registration: 現在IDが非負ならpending params/stats解除・ready=Trueのみ。負なら保有model pop→正式ID上書き、統計/学習/評価標本pop→上書き、学習量/step/概念計数pop→加算、現在ID更新、pending解除。
- 同じ負IDへの通知は旧計数を破損し得る（既存LEGACY011）。本APIは正式IDを非負とし、負元と同IDへの移管を通常契約から除く。旧productionは修正しない。
- 旧でmodelsから元が欠落するとpending snapshotを新モデルへ復元する分岐がある。新registryには分類器と一致するoptimizer管理器が必要なので、欠落再構築はモデル初期化を行う上位が準備してから確認する。本specは保有済みを明示前提にする。未移植の再構築分岐は後続server/clientの仕様で検証し、本specで旧確認全分岐移植を主張しない。
- 一保留枠は既存ownerの明示解除を使う。今回のAPIは旧と同様、負の現在帰属IDを付替え元にする。pending IDから元を推測する変更はしない。通知を処理する順序・pending/currentの整合性は上位server/clientで保証する。
- 初期登録は旧_register_trained_new_modelと候補採用に、共有再接続・初期統計・parameter snapshot・学習量帰属・現在ID切替えが存在する。初期登録と確認を一つの巨大coordinatorへ含めない。
