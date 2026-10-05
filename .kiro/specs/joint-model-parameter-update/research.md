# 調査と保存前gate

## 旧経路と主担当synthesis

旧clients/shared_backbone.pyの_train_heads_togetherを調査し、調査agentが固定batchを返す最小clientで実呼出した。単一参加は共有→個別、参加ID列(4,-7)は共有→4→-7、共有更新無効は4→-7、空列は操作なし。ID昇順のautograd.gradはmean経路では診断用で、実適用は入力順の共同損失backwardである。
バッチ抽出とカウンタだけをoracle側で固定/省略し、loss/backward/optimizerは実旧を使用できる。
モデル構築・optimizer生成の次の責務として、一回共同更新を選ぶ。参加選択やoptimizer寿命は取り込まない。

## 契約外入力と正常拡張

調査agentが0行batchのNaNとweight decayによる更新、重複IDの二回step、多クラス小数ラベルの切捨てを実測した。通常の旧サンプリングは正の固定batch長・一意ID・整数ラベルを供給するため、通常実験の不具合と断定しない。新公開入口では事前拒否する。
二値BCEのsoft targetは旧も扱うため有限[0,1]を維持し、多クラスだけ整数値を要求する。旧の入力はCPUfloat32の[N,D]、ラベル[N,1]。
空共有optimizerの生成失敗は既存LEGACY-010に記録済み。新NNの空幅identityでは共有optimizer=Noneとし個別学習を検証する。旧production/過去成果を修正しない。
非空列ではgrad有効を要求する。演算開始後の例外rollbackや極大値の数値安定性までは保証しない。

## 主担当requirements保存前gate

PASS。数値ID、EARS、明示参加/順序/空列/共有凍結/空共有/入力拒否/実旧照合の観測可能な契約を確認した。APIの具体shape/type/配置はdesignへ分離した。承認はLuna独立レビュー後にhashで記録する。
