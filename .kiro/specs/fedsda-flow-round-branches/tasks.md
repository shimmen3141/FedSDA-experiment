# タスク

- [x] 1. 実装の登録可否＋複数モデル条件を照合し、仕様レビュー後に2経路・用語・左列と通信線を更新する。
- [x] 2. 全分岐/3画面の経路・既存操作と幾何配置を検証し、独立レビューを反映する。

## 記録
- fedsda_run_server.pyのsynchronize_modelsはnew_model_registration_availableをクラスタリング有効条件へ渡し、server_round_synchronization.pyはさらに集約対象が複数であることを判定する。UIはこれに対応した2種類のラウンドの模式例とする。

- 独立Luna high仕様レビューPASS、指摘なし。旧版のラウンド経路が無条件にクロス評価を含むことで先行受入REDを確認した。
- 用語変更、サーバ外枠+50・新モデル作成内容+50と外枠縮小、関連ポート/内部接続を追随。判定右端から配布し、R04に経路結果を表示。新モデルを登録する例と新規登録なしの通常例へbuildを分岐した。
- 全6分岐×3画面の18経路411コマでissues/pageerror 0。通常経路は評価依頼/評価/統合/変更分再送を省略し、受信・内部反映・copy/replayは保持。判定結果・phase説明・配布線の起点と通常/先行配布ラベル、既存の左右キー/上下分岐/循環/focus/再生/補足を検証した。
- 両経路×3画面の代表コマでcaption・phase文言の枠内と周期説明1行、左列箱と候補枠内、通信/コピー/採用線のラベル非干渉に成功。通常R04と評価ありR05などのPNGを目視した。
- frame契約差分はR04/R05のみ。inputとコマ移動handler全文は変更前一致。通常標本の処理・モデル群/ストア/FIFOの座標は維持。Node構文・diff --check成功。数値実装/goldenは未変更で数値pytest対象外。
- 証拠はGit管理外`venv/flow-browser-check/check-round-branches.cjs`・`round-branches-validation.json`、`round-branches-geometry.cjs`・`round-branches-geometry.json`・`round-branches-final-<幅>-<応答>-<frame>.png`。
- 独立Luna high実装レビューPASS、指摘なし。全経路/配置検査記録と代表PNG、実装条件の模式例という説明を独立確認。ブラウザ独立再実行は未実施。
