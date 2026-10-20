# 命名

- cross-evaluation（既存id維持）／配布されたモデル群: 集約後に受信し、評価・統合後の反映まで保持するモデル群。予測用snapshotではない。
- R05a／評価依頼: 評価するモデルID・対象IDを渡す軽量メッセージのコマ。
- receive-models（要求8で廃止しearly-models/changed-modelsへ置換）／先行配布・変更分再送: サーバ配布から受信モデル枠へ渡すパラメータ線。同じ経路でframeごとに内容を区別。
- evaluation-request／評価依頼（IDのみ）: モデル転送を伴わない依頼線。
- apply-received／反映: 受信済みモデルと変更分を更新用モデルへ反映するクライアント内部の処理。

## サーバ内の処理
- synchronization-condition／評価・統合条件: R04だけ活性にする処理ひし形。
- server-evaluation／評価依頼・結果集計: IDで評価を依頼し統計を受け取るサーバ処理。
- consolidation／モデル統合: 評価結果から統合し、必要時に変更分を送るサーバ処理。
- early-models / changed-models: 集約後の全モデル先行配布／統合後の変更分再送。
- server-order: 常時薄いサーバ内の順序ガイド。通信やデータ移動の線ではない。
