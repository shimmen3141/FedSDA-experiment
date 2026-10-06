# 実装task

要求/設計revision1・命名revision2、graph独立PASSが正本。順次実行。

- [x] 1. 一保留枠とラウンド待機をTDDで移植する
  - 先に状態/record/拒否/借用保持/実旧列のtestをREDにし、productionと空packageを実装してGREENへ。
  - 実旧具象FedSDAのdelay1/2/4・負/0/正ID、取得/置換/clear/空とreadyのno-opを照合。保留なしの有効旧残回数は0として比較する（research.md）。不正queue/recordは空/待機/ready状態と入力を保持する。
  - 対象pytest、Ruff/format/Pyright、独立Luna実diff承認をgateにする。
  - Requirements: 1.1,1.2,1.3,1.4,2.1,2.2,2.3,2.4,3.1
- [x] 2. 固定parameterと現在損失統計をtest-only接続する
  - class2/4×delay1/2/4の6条件で実旧register→保留→統計更新→境界→取得と、新producer/loss/初期統計/store/pendingを照合。
  - 登録時snapshot全値、同ID現在統計全field、モデル更新非波及、順序/参照、parameter/grad/乱数保持を確認。production追加なし、RED N/A。
  - 対象pytest/品質/独立Luna承認をgateにする。
  - Requirements: 1.4,3.2
- [x] 3. 依存境界と固定goldenを統合検証する
  - exact依存の許可/拒否注入をRED→guard GREENにし、fresh新CPU producer/loss/store/pending接続と旧非importを確認。
  - 全pytest旧11/最終3golden、Ruff/format/Pyright/pip、固定旧/golden/test差分空、source hash/JUnitを実測記録。独立Luna task承認と主担当gate後に全check。
  - 別feature統合GOで11要件/境界/接続/共有状態/ファイル計画を確認し、再開案内を更新してcommit/push。
  - Requirements: 3.3
